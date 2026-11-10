"""EDCircles 候选：调用 ED_Lib（Cihan Topal，MIT）的 C++ 实现。

库文件 ringdet/_lib/libedcircles.so 由 ./build.sh 编译（只依赖 OpenCV core/imgproc）。
没有编译时退回 opencv-contrib 的 cv2.ximgproc.EdgeDrawing（同源移植），两个都没有就不可用。
"""
from __future__ import annotations

import ctypes
import math
import os
import sys

import numpy as np
import cv2

_OPS = {'prewitt': 0, 'sobel': 1, 'scharr': 2, 'lsd': 3}
_lib = None
_lib_err = None


def _load():
    global _lib, _lib_err
    if _lib is not None or _lib_err is not None:
        return _lib
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_lib')
    paths = [os.path.join(here, 'libedcircles.so')]
    for p in paths:
        if p and os.path.exists(p):
            try:
                lib = ctypes.CDLL(p)
            except OSError as e:
                _lib_err = f'{p}: {e}'
                continue
            lib.edc_detect.restype = ctypes.c_int
            lib.edc_detect.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_int,
                ctypes.c_void_p, ctypes.c_int]
            lib.edc_version.restype = ctypes.c_char_p
            _lib = lib
            _lib_err = None
            return _lib
    if _lib_err is None:
        _lib_err = '没有找到 ringdet/_lib/libedcircles.so（先运行 ./build.sh）'
    return None


def backend(prefer='auto'):
    """返回实际可用的后端：'edlib' / 'opencv' / ''。"""
    if prefer in ('auto', 'edlib') and _load() is not None:
        return 'edlib'
    if prefer in ('auto', 'opencv') and hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'createEdgeDrawing'):
        return 'opencv'
    return ''


def describe():
    b = backend()
    if b == 'edlib':
        return f"ED_Lib C++（{_lib.edc_version().decode()}）"
    if b == 'opencv':
        return 'cv2.ximgproc.EdgeDrawing（ED_Lib 的 OpenCV 移植）'
    return f'不可用：{_lib_err}'


def _detect_edlib(gray, grad, anchor, sigma, min_path, op, scan, validate):
    g = np.ascontiguousarray(gray, np.uint8)
    H, W = g.shape
    cap = 256
    while True:
        out = np.zeros((cap, 6), np.float64)
        n = _lib.edc_detect(g.ctypes.data, W, H, g.strides[0], _OPS[op], int(grad), int(anchor), int(scan),
                            int(min_path), float(sigma), 1, int(bool(validate)), out.ctypes.data, cap)
        if n < 0:
            raise RuntimeError('edc_detect 内部异常')
        if n <= cap:
            return out[:n]
        cap = n


def _detect_opencv(gray, grad, anchor, sigma, min_path, op, validate):
    ed = cv2.ximgproc.createEdgeDrawing()
    p = cv2.ximgproc_EdgeDrawing_Params()
    p.EdgeDetectionOperator = {'prewitt': 0, 'sobel': 1, 'scharr': 2, 'lsd': 3}[op]
    p.GradientThresholdValue = int(grad)
    p.AnchorThresholdValue = int(anchor)
    p.Sigma = float(sigma)
    p.MinPathLength = int(min_path)
    p.NFAValidation = True
    p.PFmode = bool(validate)
    ed.setParams(p)
    ed.detectEdges(np.ascontiguousarray(gray, np.uint8))
    ed.detectLines()
    e = ed.detectEllipses()
    rows = []
    if e is not None:
        for x, y, r, a, b, ang in e.reshape(-1, 6):
            if r > 0:
                rows.append((x, y, r, r, 0.0, 0.0))
            else:
                rows.append((x, y, a, b, math.radians(ang), 1.0))
    return np.array(rows, np.float64).reshape(-1, 6)


def detect(gray, grad=8, anchor=0, sigma=1.0, min_path=10, op='prewitt', scan=1, validate=False, prefer='auto'):
    """gray: uint8 单通道。返回 (N, 6)：cx, cy, a, b, theta(弧度), kind(0 圆 / 1 椭圆)。

    validate=True 时先用 EDPF 的 Helmholtz 原理筛边缘段（ED_Lib 默认行为），弱边缘会被删掉；
    False 时直接用 ED 的边缘段。圆和椭圆最后都要过 EDCircles 自己的 NFA 验证。
    """
    b = backend(prefer)
    if b == 'edlib':
        return _detect_edlib(gray, grad, anchor, sigma, min_path, op, scan, validate)
    if b == 'opencv':
        return _detect_opencv(gray, grad, anchor, sigma, min_path, op, validate)
    raise RuntimeError(f'EDCircles 不可用：{_lib_err}')


if __name__ == '__main__':
    print(describe(), file=sys.stderr)
