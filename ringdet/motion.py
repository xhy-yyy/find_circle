"""地面运动：金字塔 LK 前后向校验 + RANSAC 相似变换，优先估计目标附近的运动。"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import cv2


@dataclass
class Motion:
    A: np.ndarray      # 2x3，上一处理帧原图 px -> 当前帧原图 px
    inliers: int

    @property
    def scale(self):
        return math.hypot(self.A[0, 0], self.A[1, 0])

    def apply(self, x, y):
        A = self.A
        return (A[0, 0] * x + A[0, 1] * y + A[0, 2], A[1, 0] * x + A[1, 1] * y + A[1, 2])


class GroundMotion:
    def __init__(self, params):
        self.P = params
        self.prev = None
        self.last_inliers = 0
        self.focus = None
        self.local_ok = False
        self.error = 0.0

    def reset(self):
        self.prev = None
        self.focus = None
        self.local_ok = False
        self.error = 0.0

    def update(self, gray_u8):
        """输入当前帧灰度 uint8，返回 上一帧->当前帧 的 Motion；失败返回 None。"""
        P = self.P
        H, W = gray_u8.shape
        s = min(1.0, P.width / W)
        small = gray_u8 if s >= 1.0 else cv2.resize(
            gray_u8, (int(round(W * s)), int(round(H * s))), interpolation=cv2.INTER_AREA)
        prev, self.prev = self.prev, small
        self.last_inliers = 0
        self.local_ok = False
        self.error = 0.0
        if prev is None or prev.shape != small.shape:
            return None
        pts = cv2.goodFeaturesToTrack(prev, P.max_corners, P.quality, P.min_distance, blockSize=7)
        if pts is None:
            return None
        n_global = len(pts)
        if self.focus is not None:
            x, y, r = self.focus
            h, w = prev.shape
            radius = max(120.0, 5 * r) * s
            x, y = (x + 0.5) * s - 0.5, (y + 0.5) * s - 0.5
            mask = np.zeros_like(prev)
            cv2.rectangle(mask, (max(0, int(x - radius)), max(0, int(y - radius))),
                          (min(w - 1, int(x + radius)), min(h - 1, int(y + radius))), 255, -1)
            extra = cv2.goodFeaturesToTrack(prev, 100, P.quality, 5, mask=mask, blockSize=7)
            if extra is not None:
                pts = np.concatenate((pts, extra))
        lk = dict(winSize=(21, 21), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        nxt, st1, _ = cv2.calcOpticalFlowPyrLK(prev, small, pts, None, **lk)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(small, prev, nxt, None, **lk)
        fb = np.linalg.norm((back - pts).reshape(-1, 2), axis=1)
        good = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < P.fb_max)
        if good.sum() < 6:
            return None
        p0, p1 = pts.reshape(-1, 2)[good], nxt.reshape(-1, 2)[good]
        if self.focus is not None:
            near = np.max(np.abs(p0 - (x, y)), axis=1) <= radius
            if near.sum() >= 8:
                local, local_inliers = cv2.estimateAffinePartial2D(
                    p0[near], p1[near], method=cv2.RANSAC,
                    ransacReprojThreshold=min(P.ransac_px, 1.25), maxIters=1000, confidence=0.99)
                if local is not None:
                    keep = local_inliers.ravel().astype(bool)
                    q0, q1 = p0[near][keep], p1[near][keep]
                    scale = math.hypot(local[0, 0], local[1, 0])
                    if len(q0) >= 8 and len(q0) >= 0.6 * near.sum() and 0.75 <= scale <= 1.35:
                        residual = np.linalg.norm(q0 @ local[:, :2].T + local[:, 2] - q1, axis=1)
                        self.error = float(np.percentile(residual, 90)) / s
                        self.local_ok = True
                        self.last_inliers = len(q0)
                        A = local.astype(np.float64)
                        if s < 1.0:
                            c = 0.5 - 0.5 / s
                            A[:, 2] = A[:, :2] @ np.array([c, c]) + (A[:, 2] + 0.5) / s - 0.5
                        return Motion(A, len(q0))
        global_good = good[:n_global]
        p0 = pts[:n_global].reshape(-1, 2)[global_good]
        p1 = nxt[:n_global].reshape(-1, 2)[global_good]
        if len(p0) < P.min_inliers:
            return None
        A, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC,
                                             ransacReprojThreshold=P.ransac_px,
                                             maxIters=2000, confidence=0.99)
        if A is None:
            return None
        inl = inl.ravel().astype(bool)
        n = int(inl.sum())
        self.last_inliers = n
        if n < P.min_inliers:
            return None
        # 内点不能挤在画面一小块
        h, w = small.shape
        gx = np.clip((p0[inl, 0] / w * P.grid).astype(int), 0, P.grid - 1)
        gy = np.clip((p0[inl, 1] / h * P.grid).astype(int), 0, P.grid - 1)
        if len(set(zip(gx.tolist(), gy.tolist()))) < P.min_cells:
            return None
        residual = np.linalg.norm(p0[inl] @ A[:, :2].T + A[:, 2] - p1[inl], axis=1)
        self.error = float(np.percentile(residual, 90)) / s
        A = A.astype(np.float64)
        # 缩放图与原图的像素中心关系：p_s = (p + 0.5) * s - 0.5，换回原图坐标
        if s < 1.0:
            M = A[:, :2]
            c = 0.5 - 0.5 / s
            A[:, 2] = M @ np.array([c, c]) + (A[:, 2] + 0.5) / s - 0.5
        return Motion(A, n)
