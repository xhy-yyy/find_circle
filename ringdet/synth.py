"""合成测试数据：程序化地面纹理 + 干扰物 + 抗锯齿石灰环 + 相机退化（模糊、噪声、JPEG）。

没有数据集时用它测召回、误检和圆心误差（真值精确已知）。
"""
from __future__ import annotations

import math

import numpy as np
import cv2


def ground_texture(h, w, rng, base=120.0, amp=30.0, octaves=6):
    """多尺度噪声地面。"""
    img = np.zeros((h, w), np.float32)
    total = 0.0
    for o in range(octaves):
        cell = 2 ** (octaves - o)
        gh, gw = h // cell + 3, w // cell + 3
        g = rng.standard_normal((gh, gw)).astype(np.float32)
        up = cv2.resize(g, (gw * cell, gh * cell), interpolation=cv2.INTER_CUBIC)[:h, :w]
        a = 0.7 ** o
        img += a * up
        total += a * a
    return base + amp * img / math.sqrt(total)


def _paint(img, x0, y0, x1, y1, fn, delta, ss=4):
    """在 [x0,x1)x[y0,y1) 内按 fn(X,Y) 的超采样覆盖率加亮度 delta。"""
    H, W = img.shape
    x0, y0 = max(0, int(math.floor(x0))), max(0, int(math.floor(y0)))
    x1, y1 = min(W, int(math.ceil(x1)) + 1), min(H, int(math.ceil(y1)) + 1)
    if x1 <= x0 or y1 <= y0:
        return
    xs = x0 + (np.arange((x1 - x0) * ss) + 0.5) / ss - 0.5
    ys = y0 + (np.arange((y1 - y0) * ss) + 0.5) / ss - 0.5
    X, Y = np.meshgrid(xs, ys)
    cov = fn(X, Y).astype(np.float32).reshape(y1 - y0, ss, x1 - x0, ss).mean(axis=(1, 3))
    img[y0:y1, x0:x1] += delta * cov


def draw_ring(img, cx, cy, r, w, contrast, axes=None, angle_deg=0.0, gaps=()):
    """画石灰环。axes=(a,b) 时画椭圆；gaps: [(起始角, 终止角)] 度，这些角度上没有石灰。"""
    a, b = axes if axes else (r, r)
    t = math.radians(angle_deg)
    ct, st = math.cos(t), math.sin(t)
    rm = (a + b) / 2

    def fn(X, Y):
        dx, dy = X - cx, Y - cy
        qx, qy = dx * ct + dy * st, -dx * st + dy * ct
        d = (np.hypot(qx / a, qy / b) - 1) * rm
        m = np.abs(d) <= w / 2
        if gaps:
            ang = np.degrees(np.arctan2(dy, dx)) % 360
            for g0, g1 in gaps:
                g0, g1 = g0 % 360, g1 % 360
                m &= ~(((ang >= g0) & (ang <= g1)) if g0 <= g1 else ((ang >= g0) | (ang <= g1)))
        return m

    e = max(a, b) + w + 2
    _paint(img, cx - e, cy - e, cx + e, cy + e, fn, contrast)


def draw_disc(img, cx, cy, r, contrast):
    _paint(img, cx - r - 2, cy - r - 2, cx + r + 2, cy + r + 2,
           lambda X, Y: np.hypot(X - cx, Y - cy) <= r, contrast)


def draw_line(img, x0, y0, x1, y1, w, contrast):
    L = math.hypot(x1 - x0, y1 - y0)
    ux, uy = (x1 - x0) / L, (y1 - y0) / L

    def fn(X, Y):
        px, py = X - x0, Y - y0
        along = px * ux + py * uy
        perp = np.abs(-px * uy + py * ux)
        return (along >= 0) & (along <= L) & (perp <= w / 2)

    _paint(img, min(x0, x1) - w - 2, min(y0, y1) - w - 2, max(x0, x1) + w + 2, max(y0, y1) + w + 2, fn, contrast)


def draw_arc(img, cx, cy, r, w, contrast, a0, a1):
    """只画 [a0, a1] 度这一段弧。"""
    draw_ring(img, cx, cy, r, w, contrast, gaps=[(a1, a0)])


def add_speckles(img, rng, n, amp=40.0, sigma=(0.4, 1.2)):
    """碎石：随机亮点暗点。"""
    H, W = img.shape
    for _ in range(n):
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        s = rng.uniform(*sigma)
        a = rng.uniform(-amp, amp)
        e = 3 * s + 1
        _paint(img, x - e, y - e, x + e, y + e,
               lambda X, Y: np.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2 * s * s)), a, ss=2)


def add_distractors(img, rng, r_ref, n_disc=6, n_line=4, n_arc=3, n_blob=4, contrast=(30, 80), avoid=None):
    """白色实心圆、地面白线、半圆弧（同尺寸）、大块亮斑。avoid=(x,y,R) 区域内不放。"""
    H, W = img.shape
    placed = []

    def pos(margin):
        for _ in range(50):
            x, y = rng.uniform(margin, W - margin), rng.uniform(margin, H - margin)
            if avoid is None or math.hypot(x - avoid[0], y - avoid[1]) > avoid[2] + margin:
                return x, y
        return x, y

    for _ in range(n_disc):
        r = rng.uniform(0.3, 1.5) * r_ref
        x, y = pos(r + 4)
        draw_disc(img, x, y, r, rng.uniform(*contrast))
        placed.append(('disc', x, y, r))
    for _ in range(n_line):
        x, y = pos(10)
        ang = rng.uniform(0, math.pi)
        L = rng.uniform(4, 15) * r_ref
        draw_line(img, x, y, x + L * math.cos(ang), y + L * math.sin(ang),
                  rng.uniform(0.08, 0.2) * r_ref, rng.uniform(*contrast))
        placed.append(('line', x, y, L))
    for _ in range(n_arc):
        r = rng.uniform(0.8, 1.25) * r_ref
        x, y = pos(r + 4)
        a0 = rng.uniform(0, 360)
        span = rng.uniform(90, 200)
        draw_arc(img, x, y, r, max(1.0, 0.1 * r), rng.uniform(*contrast), a0, a0 + span)
        placed.append(('arc', x, y, r))
    for _ in range(n_blob):
        sx, sy = rng.uniform(1, 4) * r_ref, rng.uniform(1, 4) * r_ref
        x, y = pos(max(sx, sy))
        _paint(img, x - sx, y - sy, x + sx, y + sy,
               lambda X, Y: (np.abs(X - x) <= sx / 2) & (np.abs(Y - y) <= sy / 2), rng.uniform(*contrast), ss=1)
        placed.append(('blob', x, y, max(sx, sy)))
    return placed


def degrade(img, rng, blur=0.8, noise=2.0, jpeg=85):
    """相机退化：高斯模糊 + 噪声 + JPEG。返回 uint8。"""
    out = img.astype(np.float32)
    if blur > 0:
        out = cv2.GaussianBlur(out, (0, 0), blur)
    if noise > 0:
        out = out + rng.normal(0, noise, out.shape).astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if jpeg:
        ok, buf = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg)])
        out = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return out


def scene(rng, size=(960, 540), r=None, w=None, contrast=None, ring=True, distractors=True,
          speckles=400, blur=0.8, noise=2.0, jpeg=85, gaps=()):
    """一张合成图。返回 (uint8 灰度, 真值 dict 或 None)。"""
    W, H = size
    img = ground_texture(H, W, rng)
    r = r if r is not None else rng.uniform(9, 40)
    w = w if w is not None else max(0.8, rng.uniform(0.05, 0.1) * r)
    contrast = contrast if contrast is not None else rng.uniform(25, 60)
    m = r + 10
    cx, cy = rng.uniform(m, W - m), rng.uniform(m, H - m)
    add_speckles(img, rng, speckles)
    if distractors:
        add_distractors(img, rng, r, avoid=(cx, cy, r + 6) if ring else None)
    gt = None
    if ring:
        draw_ring(img, cx, cy, r, w, contrast, gaps=gaps)
        gt = dict(cx=cx, cy=cy, r=r, w=w, contrast=contrast)
    return degrade(img, rng, blur, noise, jpeg), gt


def descent(rng, n=90, size=(960, 540), r0=10.0, r1=60.0, contrast=40.0, w_ratio=0.08,
            drift=(0.6, 0.25), rot_deg=0.15, jitter=0.4, blur=0.8, noise=2.0, jpeg=85,
            occlude=(), hide=(), exit_frames=0, distractors=True):
    """模拟下降视频：地面纹理和环一起按相似变换运动，半径从 r0 涨到 r1。

    occlude: 这些帧里环被挡住 240°（只剩 1/3）；hide: 这些帧里环完全看不见（强模糊/全遮挡）；
    exit_frames > 0 时最后这么多帧环快速移出画面。
    返回 frames (uint8 灰度列表), gts [(cx, cy, r) 或 None]
    """
    W, H = size
    R_t = r0                      # 纹理上环的半径（纹理 px）；k = r / R_t >= 1，只放大不缩小
    TS = 2048
    tex = ground_texture(TS, TS, rng)
    add_speckles(tex, rng, 3000)
    ring_t = (TS / 2, TS / 2)
    if distractors:
        add_distractors(tex, rng, R_t, n_disc=25, n_line=15, n_arc=10, n_blob=15,
                        avoid=(ring_t[0], ring_t[1], 3 * R_t))
    frames, gts = [], []
    c = np.array([W * 0.45, H * 0.5])
    th = 0.0
    for i in range(n):
        f = i / max(1, n - 1)
        r = r0 * (r1 / r0) ** f
        k = r / R_t
        if exit_frames and i >= n - exit_frames:
            c = c + np.array([W / exit_frames, 0.0])
        c = c + np.array(drift) + rng.normal(0, jitter, 2)
        th += math.radians(rot_deg)
        ct, st = math.cos(th), math.sin(th)
        A = np.array([[k * ct, -k * st, 0], [k * st, k * ct, 0]], np.float64)
        A[:, 2] = c - A[:, :2] @ np.array(ring_t)
        img = cv2.warpAffine(tex, A, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        gaps = ()
        if i in occlude:
            g0 = rng.uniform(0, 360)
            gaps = [(g0, g0 + 240)]
        if i not in hide:
            draw_ring(img, c[0], c[1], r, max(0.8, w_ratio * r), contrast, gaps=gaps)
        frames.append(degrade(img, rng, blur, noise, jpeg))
        inside = bool(c[0] - r >= 0 and c[0] + r <= W - 1 and c[1] - r >= 0 and c[1] + r <= H - 1)
        gts.append((float(c[0]), float(c[1]), float(r), inside))
    return frames, gts
