"""相机内参与畸变。像素坐标约定：左上原点，像素中心在整数坐标。"""
from __future__ import annotations

import numpy as np
import cv2


class Camera:
    def __init__(self, K, dist, size, calibrated):
        self.K = np.asarray(K, np.float64).reshape(3, 3)
        self.dist = np.asarray(dist, np.float64).ravel()
        self.size = (int(size[0]), int(size[1]))   # (W, H)
        self.calibrated = bool(calibrated)
        self._has_dist = bool(np.any(np.abs(self.dist) > 0))

    @property
    def fx(self):
        return float(self.K[0, 0])

    @staticmethod
    def default(width, height, fx=None):
        """未标定：默认水平视场 90°（fx = W/2），主点在画面中心，无畸变。"""
        f = float(fx) if fx else width / 2.0
        K = [[f, 0, (width - 1) / 2.0], [0, f, (height - 1) / 2.0], [0, 0, 1]]
        return Camera(K, np.zeros(5), (width, height), calibrated=False)

    @staticmethod
    def from_yaml(path, size=None):
        """读 ROS camera_info 格式（camera_calibration 包的输出）。

        size 与标定分辨率不同时按比例缩放内参（要求同一传感器区域、只是缩放）。
        """
        import yaml
        with open(path, 'r', encoding='utf-8') as f:
            d = yaml.safe_load(f)
        W0, H0 = int(d['image_width']), int(d['image_height'])
        K = np.array(d['camera_matrix']['data'], np.float64).reshape(3, 3)
        dist = np.array(d['distortion_coefficients']['data'], np.float64)
        if size is not None and (size[0] != W0 or size[1] != H0):
            sx, sy = size[0] / W0, size[1] / H0
            K = K.copy()
            K[0, 0] *= sx
            K[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
            K[1, 1] *= sy
            K[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
            W0, H0 = size
        return Camera(K, dist, (W0, H0), calibrated=True)

    def undistort(self, pts):
        """畸变像素 -> 去畸变像素（同一个 K）。pts: (N,2)"""
        pts = np.asarray(pts, np.float64).reshape(-1, 2)
        if not self._has_dist:
            return pts.copy()
        out = cv2.undistortPoints(pts.reshape(-1, 1, 2), self.K, self.dist, P=self.K)
        return out.reshape(-1, 2)

    def distort(self, pts):
        """去畸变像素 -> 畸变像素。"""
        pts = np.asarray(pts, np.float64).reshape(-1, 2)
        if not self._has_dist:
            return pts.copy()
        n = (pts - self.K[:2, 2]) / np.array([self.K[0, 0], self.K[1, 1]])
        obj = np.hstack([n, np.ones((len(n), 1))]).reshape(-1, 1, 3)
        out, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), self.K, self.dist)
        return out.reshape(-1, 2)

    def bearing(self, pt):
        """畸变像素 -> 去畸变归一化坐标 (x/z, y/z)。"""
        u = self.undistort(np.asarray(pt, np.float64).reshape(1, 2))[0]
        return (float((u[0] - self.K[0, 2]) / self.K[0, 0]), float((u[1] - self.K[1, 2]) / self.K[1, 1]))
