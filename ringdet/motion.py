"""地面运动：金字塔 LK 前后向校验 + RANSAC 相似变换，优先估计目标附近的运动。"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import cv2

from .edcircles import flow_fb


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
        self.points = None
        self.n_global = 0
        self.frames = 0
        self.backend = 'cpu'
        self.backend_error = ''
        self._gpu_prev = None
        self._gpu_cur = None
        self._gpu_points = None
        self._cuda_lk = None
        self._cuda_corners = {}
        if params.backend not in ('auto', 'cpu', 'cuda'):
            raise ValueError('motion.backend 必须是 auto、cpu 或 cuda')
        if params.backend != 'cpu':
            try:
                if cv2.cuda.getCudaEnabledDeviceCount() < 1:
                    raise RuntimeError('没有可用的 CUDA 设备')
                factory = getattr(cv2, 'cuda_SparsePyrLKOpticalFlow', None)
                create = getattr(factory, 'create', None) if factory is not None else None
                if create is None:
                    create = getattr(cv2.cuda, 'SparsePyrLKOpticalFlow_create', None)
                if create is None:
                    raise RuntimeError('OpenCV 未编译 cudaoptflow 模块')
                self._cuda_lk = create(winSize=(21, 21), maxLevel=3, iters=30)
                self._gpu_prev, self._gpu_cur = cv2.cuda_GpuMat(), cv2.cuda_GpuMat()
                self._gpu_points = cv2.cuda_GpuMat()
                self.backend = 'cuda'
            except (cv2.error, AttributeError, RuntimeError) as e:
                self.backend_error = str(e)
                if params.backend == 'cuda':
                    raise RuntimeError(f'CUDA 光流初始化失败：{e}') from e

    def reset(self):
        self.prev = None
        self.focus = None
        self.local_ok = False
        self.error = 0.0
        self.points = None
        self.n_global = 0
        self.frames = 0
        if self._gpu_prev is not None:
            self._gpu_prev.release()
            self._gpu_cur.release()

    def _corners(self, image, limit, distance, bounds=None):
        if self.backend == 'cuda' and hasattr(cv2.cuda, 'createGoodFeaturesToTrackDetector'):
            try:
                key = (limit, distance)
                if key not in self._cuda_corners:
                    self._cuda_corners[key] = cv2.cuda.createGoodFeaturesToTrackDetector(
                        cv2.CV_8UC1, maxCorners=limit, qualityLevel=self.P.quality,
                        minDistance=distance, blockSize=7, useHarrisDetector=False)
                if self._gpu_prev.empty():
                    self._gpu_prev.upload(image)
                gpu = self._gpu_prev
                if bounds is not None:
                    xa, ya, xb, yb = bounds
                    gpu = gpu.rowRange(ya, yb).colRange(xa, xb)
                points = self._cuda_corners[key].detect(gpu)
                return None if points.empty() else points.download().reshape(-1, 1, 2)
            except (cv2.error, AttributeError) as e:
                self.backend_error = str(e)
                if self.P.backend == 'cuda':
                    raise RuntimeError(f'CUDA 角点检测失败：{e}') from e
                self.backend = 'cpu'
        if bounds is not None:
            xa, ya, xb, yb = bounds
            image = image[ya:yb, xa:xb]
        return cv2.goodFeaturesToTrack(image, limit, self.P.quality, distance, blockSize=7)

    def _flow(self, prev, small, pts):
        if self.backend == 'cuda':
            try:
                if self._gpu_prev.empty():
                    self._gpu_prev.upload(prev)
                self._gpu_cur.upload(small)
                self._gpu_points.upload(pts.reshape(1, -1, 2))
                nxt, st1, _ = self._cuda_lk.calc(self._gpu_prev, self._gpu_cur, self._gpu_points, None)
                back, st2, _ = self._cuda_lk.calc(self._gpu_cur, self._gpu_prev, nxt, None)
                out = (nxt.download().reshape(-1, 1, 2), st1.download(),
                       back.download().reshape(-1, 1, 2), st2.download())
                self._gpu_prev, self._gpu_cur = self._gpu_cur, self._gpu_prev
                return out
            except (cv2.error, AttributeError) as e:
                self.backend_error = str(e)
                if self.P.backend == 'cuda':
                    raise RuntimeError(f'CUDA 光流运行失败：{e}') from e
                self.backend = 'cpu'
                self._cuda_lk = None
        native = flow_fb(prev, small, pts)
        if native is not None:
            return native
        lk = dict(winSize=(21, 21), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        nxt, st1, _ = cv2.calcOpticalFlowPyrLK(prev, small, pts, None, **lk)
        if nxt is None or st1 is None:
            return None
        back, st2, _ = cv2.calcOpticalFlowPyrLK(small, prev, nxt, None, **lk)
        return nxt, st1, back, st2

    def update(self, gray_u8):
        """输入当前帧灰度 uint8，返回 上一帧->当前帧 的 Motion；失败返回 None。"""
        P = self.P
        H, W = gray_u8.shape
        s = min(1.0, P.width / W)
        small = gray_u8 if s >= 1.0 else cv2.resize(
            gray_u8, (int(round(W * s)), int(round(H * s))), interpolation=cv2.INTER_LINEAR)
        prev, self.prev = self.prev, small
        self.frames += 1
        self.last_inliers = 0
        self.local_ok = False
        self.error = 0.0
        if prev is None or prev.shape != small.shape:
            self.points = None
            if self._gpu_prev is not None:
                self._gpu_prev.release()
            return None
        pts, n_global = self.points, self.n_global
        refresh = self.frames % max(1, P.refresh_frames) == 0
        if pts is None or n_global < max(P.min_inliers, int(P.refresh_ratio * P.max_corners)) or refresh:
            pts = self._corners(prev, P.max_corners, P.min_distance)
            n_global = 0 if pts is None else len(pts)
        if pts is None:
            if self._gpu_prev is not None:
                self._gpu_prev.release()
            return None
        if self.focus is not None:
            x, y, r = self.focus
            h, w = prev.shape
            radius = max(120.0, 5 * r) * s
            x, y = (x + 0.5) * s - 0.5, (y + 0.5) * s - 0.5
            local = pts[n_global:].reshape(-1, 2)
            enough = np.sum(np.max(np.abs(local - (x, y)), axis=1) <= radius) >= 30
            if refresh or not enough:
                xa, ya = max(0, int(x - radius)), max(0, int(y - radius))
                xb, yb = min(w, int(x + radius) + 1), min(h, int(y + radius) + 1)
                if xb - xa >= 7 and yb - ya >= 7:
                    extra = self._corners(prev, 100, 5, (xa, ya, xb, yb))
                    if extra is not None:
                        extra += np.float32([xa, ya])
                        pts = np.concatenate((pts[:n_global], extra))
        tracked = self._flow(prev, small, pts)
        if tracked is None:
            self.points = None
            return None
        nxt, st1, back, st2 = tracked
        if back is None or st2 is None:
            self.points = None
            return None
        fb = np.linalg.norm((back - pts).reshape(-1, 2), axis=1)
        good = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < P.fb_max)
        self.n_global = int(good[:n_global].sum())
        self.points = nxt[good].copy()
        if good.sum() < 6:
            self.points = None
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
