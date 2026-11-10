"""地面运动：金字塔 LK 前后向校验 + RANSAC 相似变换（平移、旋转、统一缩放）。"""
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

    def reset(self):
        self.prev = None

    def update(self, gray_u8):
        """输入当前帧灰度 uint8，返回 上一帧->当前帧 的 Motion；失败返回 None。"""
        P = self.P
        H, W = gray_u8.shape
        s = min(1.0, P.width / W)
        small = gray_u8 if s >= 1.0 else cv2.resize(
            gray_u8, (int(round(W * s)), int(round(H * s))), interpolation=cv2.INTER_AREA)
        prev, self.prev = self.prev, small
        self.last_inliers = 0
        if prev is None or prev.shape != small.shape:
            return None
        pts = cv2.goodFeaturesToTrack(prev, P.max_corners, P.quality, P.min_distance, blockSize=7)
        if pts is None or len(pts) < P.min_inliers:
            return None
        lk = dict(winSize=(21, 21), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        nxt, st1, _ = cv2.calcOpticalFlowPyrLK(prev, small, pts, None, **lk)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(small, prev, nxt, None, **lk)
        fb = np.linalg.norm((back - pts).reshape(-1, 2), axis=1)
        good = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < P.fb_max)
        if good.sum() < P.min_inliers:
            return None
        p0, p1 = pts.reshape(-1, 2)[good], nxt.reshape(-1, 2)[good]
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
        A = A.astype(np.float64)
        # 缩放图与原图的像素中心关系：p_s = (p + 0.5) * s - 0.5，换回原图坐标
        if s < 1.0:
            M = A[:, :2]
            c = 0.5 - 0.5 / s
            A[:, 2] = M @ np.array([c, c]) + (A[:, 2] + 0.5) / s - 0.5
        return Motion(A, n)
