"""一帧图像的检测通道，按需计算。

gray ：亮度，任何地面都能用。
white：min(B, G, R)。石灰三个颜色分量都高，取最小值后几乎不变暗；有颜色的地面（黄沙、红土、草）
       至少有一个分量低，取最小值后明显变暗，环的对比因此变大。灰色地面上与 gray 基本相同。
输入是灰度图时只有 gray。
"""
from __future__ import annotations

import numpy as np
import cv2


def noise_sigma(img):
    """杂波尺度：1.4826 * median(|g - median3(g)|)。"""
    img = np.ascontiguousarray(img, np.float32)
    d = np.abs(img - cv2.medianBlur(img, 3))
    return 1.4826 * float(np.median(d))


def frame_noise(gray):
    """整帧杂波尺度：中心和四个象限中心各取一块，取中位数。"""
    H, W = gray.shape
    b = min(192, H // 3, W // 3)
    vals = []
    for fx, fy in ((0.5, 0.5), (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)):
        x, y = int(fx * W - b / 2), int(fy * H - b / 2)
        vals.append(noise_sigma(gray[y:y + b, x:x + b]))
    return float(np.median(vals))


class Frame:
    def __init__(self, img, noise_floor):
        self._bgr = img if img.ndim == 3 else None
        self._floor = noise_floor
        self._u8 = {'gray': img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)}
        self._f32 = {}
        self._noise = {}

    @property
    def shape(self):
        return self._u8['gray'].shape

    def channels(self, wanted):
        """wanted 里本帧有的通道，顺序不变。"""
        return [c for c in wanted if c == 'gray' or self._bgr is not None]

    def u8(self, ch):
        if ch not in self._u8:
            if ch != 'white' or self._bgr is None:
                raise ValueError(f'没有通道 {ch}')
            b, g, r = cv2.split(self._bgr)
            self._u8[ch] = cv2.min(cv2.min(b, g), r)
        return self._u8[ch]

    def f32(self, ch):
        if ch not in self._f32:
            self._f32[ch] = self.u8(ch).astype(np.float32)
        return self._f32[ch]

    def noise(self, ch):
        if ch not in self._noise:
            self._noise[ch] = max(frame_noise(self.f32(ch)), self._floor)
        return self._noise[ch]
