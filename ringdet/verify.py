"""复核与测量：分扇区径向采样 -> 每条射线找双侧都暗的亮峰 -> RANSAC + Huber 拟圆 -> 门槛。

每条射线上用一维"中间亮带 vs 两侧背景带"模板找峰，模板宽度在几档线宽里
自动选（所有射线取同一档），所以从高空 1 px 细线到近地几十 px 粗线都适用。
"""
from __future__ import annotations

import math
from functools import lru_cache
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import cv2

from .edcircles import ellipse_hypotheses


@dataclass
class Measurement:
    ok: bool
    reason: str
    cx: float = float('nan')      # 原图（畸变）px
    cy: float = float('nan')
    r: float = float('nan')
    ellipse: Optional[Tuple[float, float, float, float, float]] = None   # (cx, cy, a, b, angle_deg)
    line_w: float = float('nan')
    n_rays: int = 0
    n_inside: int = 0
    n_valid: int = 0
    n_inliers: int = 0
    sectors: int = 0
    max_gap: int = 0
    inlier_ratio: float = 0.0
    rms: float = float('nan')
    contrast: float = 0.0
    points: Optional[np.ndarray] = field(default=None, repr=False)   # 内点（原图 px），画图用
    source: str = ''              # 候选来源：'ed' = EDCircles，'mf' = 匹配滤波
    channel: str = 'gray'         # 复核通过的通道
    ed_ok: bool = False           # 有 EDCircles 支持（全图检出，或候选级放大后检出）


def _delta(pair, r):
    return max(pair[0], pair[1] * r)


def _widths(r, ratio):
    lo = max(1.0, ratio[0] * r)
    hi = max(1.5, ratio[1] * r)
    ws = [lo]
    while ws[-1] * math.sqrt(2) <= hi * 1.0001:
        ws.append(ws[-1] * math.sqrt(2))
    return ws


def _bg_len(w):
    return max(2.0, 0.5 * w)


@lru_cache(maxsize=8)
def _angles(n):
    th = np.arange(n) * (2 * math.pi / n)
    return th, np.cos(th), np.sin(th)


def _ray_scores(gray, cx, cy, r, delta, n_rays, V, r_ray=None):
    """沿射线采样并对每档线宽打分。

    r_ray：每条射线的预测半径（椭圆时用），None 时都等于 r。采样半径 s 是相对 r_ray 的偏移。
    返回 dict: thetas, s (相对偏移), iw (找峰窗口下标), inside (射线全在图内),
              scores[j] (n_rays x len(iw))，widths, r_ray
    """
    H, W = gray.shape
    widths = _widths(r, V.width_ratio)
    step = V.step
    th, ct, st = _angles(n_rays)
    rr = np.full(n_rays, float(r)) if r_ray is None else np.asarray(r_ray, np.float64)
    ext = max(w / 2 + V.gap + _bg_len(w) for w in widths) + 2 * step
    lo = max(-(rr.min() - 0.5), -delta - ext)
    s = np.arange(lo, delta + ext + 1e-9, step)
    dist = rr[:, None] + s[None, :]
    xs = (cx + ct[:, None] * dist).astype(np.float32)
    ys = (cy + st[:, None] * dist).astype(np.float32)
    inside = ((xs >= 0) & (xs <= W - 1) & (ys >= 0) & (ys <= H - 1)).all(axis=1)
    prof = cv2.remap(gray, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    cs = np.zeros((n_rays, len(s) + 1), np.float64)
    np.cumsum(prof, axis=1, out=cs[:, 1:])
    iw = np.nonzero((s >= -delta) & (s <= delta))[0]
    n = len(s)
    scores = []
    for w in widths:
        nb = 2 * int(math.floor(w / (2 * step))) + 1           # 奇数，框中心落在采样点上
        ng = int(math.floor(V.gap / step + 0.5))
        nl = max(2, int(math.floor(_bg_len(w) / step + 0.5)))
        a = iw - nb // 2
        def box(lo, ln):
            lo = np.clip(lo, 0, n - ln)
            return (cs[:, lo + ln] - cs[:, lo]) / ln
        line = box(a, nb)
        inner = box(a - ng - nl, nl)
        outer = box(a + nb + ng, nl)
        # 亮物边缘被锐化/JPEG 振铃出的亮边：两侧背景差 |I-O| 是亮线对比的好几倍，整条射线作废
        a, b = line - inner, line - outer
        sc = np.minimum(a, b)
        sc[np.abs(a - b) > V.side_ratio * np.maximum(sc, 0)] = 0.0
        scores.append(sc)
    return dict(thetas=th, ct=ct, st=st, s=s, iw=iw, inside=inside, scores=scores, widths=widths, r_ray=rr)


def _circle3(p):
    """p: (K,3,2) -> 圆心 (K,2)、半径 (K,)；退化的给 nan。"""
    ax, ay = p[:, 0, 0], p[:, 0, 1]
    bx, by = p[:, 1, 0], p[:, 1, 1]
    cx_, cy_ = p[:, 2, 0], p[:, 2, 1]
    d = 2 * (ax * (by - cy_) + bx * (cy_ - ay) + cx_ * (ay - by))
    with np.errstate(divide='ignore', invalid='ignore'):
        a2, b2, c2 = ax ** 2 + ay ** 2, bx ** 2 + by ** 2, cx_ ** 2 + cy_ ** 2
        ux = (a2 * (by - cy_) + b2 * (cy_ - ay) + c2 * (ay - by)) / d
        uy = (a2 * (cx_ - bx) + b2 * (ax - cx_) + c2 * (bx - ax)) / d
    c = np.stack([ux, uy], 1)
    rr = np.hypot(ax - ux, ay - uy)
    bad = np.abs(d) < 1e-9
    c[bad] = np.nan
    rr[bad] = np.nan
    return c, rr


def _huber_circle(pts, c, r, delta, iters=10):
    """几何距离 + Huber 权的 Gauss-Newton。"""
    c = np.array(c, np.float64)
    for _ in range(iters):
        d = pts - c
        dist = np.hypot(d[:, 0], d[:, 1])
        dist = np.maximum(dist, 1e-9)
        res = dist - r
        a = np.abs(res)
        w = np.where(a <= delta, 1.0, delta / np.maximum(a, 1e-12))
        J = np.column_stack([-d[:, 0] / dist, -d[:, 1] / dist, -np.ones(len(pts))])
        JW = J * w[:, None]
        try:
            step = np.linalg.solve(JW.T @ J + 1e-9 * np.eye(3), -JW.T @ res)
        except np.linalg.LinAlgError:
            break
        c += step[:2]
        r += step[2]
        if np.abs(step).max() < 1e-4:
            break
    return c, float(r)


def _fit_circle(pts, tau, iters, rng, r_lim):
    """RANSAC 粗筛 + Huber 精修。r_lim=(lo, hi) 之外的假设不考虑。
    返回 (center, r, inlier_mask) 或 None。"""
    n = len(pts)
    if n < 5:
        return None
    idx = np.argsort(rng.random((iters, n)), axis=1)[:, :3]   # 每行 3 个不重复下标
    c, rr = _circle3(pts[idx])
    ok = np.isfinite(rr) & (rr >= r_lim[0]) & (rr <= r_lim[1])
    if not ok.any():
        return None
    c, rr = c[ok], rr[ok]
    res = np.abs(np.hypot(pts[None, :, 0] - c[:, None, 0], pts[None, :, 1] - c[:, None, 1]) - rr[:, None])
    cnt = (res <= tau).sum(1)
    b = int(np.argmax(cnt))
    inl = res[b] <= tau
    if inl.sum() < 5:
        return None
    cen, r = _huber_circle(pts[inl], c[b], rr[b], tau / 2)
    for _ in range(2):
        if not (r_lim[0] <= r <= r_lim[1]) or not np.all(np.isfinite(cen)):
            return None
        res = np.abs(np.hypot(pts[:, 0] - cen[0], pts[:, 1] - cen[1]) - r)
        inl = res <= tau
        if inl.sum() < 5:
            return None
        cen, r = _huber_circle(pts[inl], cen, r, tau / 2)
    if not (r_lim[0] <= r <= r_lim[1]) or not np.all(np.isfinite(cen)):
        return None
    res = np.abs(np.hypot(pts[:, 0] - cen[0], pts[:, 1] - cen[1]) - r)
    return cen, r, res <= tau


def _sector_stats(pts, cen, n_sec, min_rays):
    """min_rays：扇区内至少这么多内点才算有效。"""
    ang = np.arctan2(pts[:, 1] - cen[1], pts[:, 0] - cen[0])
    sec = ((ang + math.pi) / (2 * math.pi) * n_sec).astype(int) % n_sec
    cnt = np.bincount(sec, minlength=n_sec)
    good = cnt >= min_rays
    if good.all():
        return n_sec, 0
    if not good.any():
        return 0, n_sec
    g = np.roll(good, -int(np.argmax(good)))   # 从一个有效扇区开始，环形最长空段
    gap = run = 0
    for v in g:
        run = 0 if v else run + 1
        gap = max(gap, run)
    return int(good.sum()), gap


def _ellipse_radius(a, b, ang_deg, th):
    """椭圆（半轴 a、b，长轴方向 ang_deg）在方向 th 上中心到边界的距离。"""
    t = th - math.radians(ang_deg)
    return a * b / np.sqrt((b * np.cos(t)) ** 2 + (a * np.sin(t)) ** 2)


def _ellipse_residual(pts, ex, ey, a, b, ang_deg):
    t = math.radians(ang_deg)
    q = pts - (ex, ey)
    qx = q[:, 0] * math.cos(t) + q[:, 1] * math.sin(t)
    qy = -q[:, 0] * math.sin(t) + q[:, 1] * math.cos(t)
    return (np.hypot(qx / a, qy / b) - 1) * (a + b) / 2


def _peaks(gray, cx, cy, r, delta, noise, V, r_ray=None):
    """射线找峰。返回 (m 半成品, px 有效峰点, peak 对比, valid 掩码, n_rays) 或 (m, None...)。"""
    n_rays = V.rays_large if r >= V.rays_large_r else V.rays_small
    d = _ray_scores(gray, cx, cy, r, delta, n_rays, V, r_ray)
    inside = d['inside']
    m = Measurement(False, '', n_rays=n_rays, n_inside=int(inside.sum()))
    if m.n_inside < 8:
        m.reason = 'outside'
        return m, None, None, None, n_rays
    # 所有射线用同一档线宽：取"各射线最高分"中位数最大的那档
    meds = [np.median(sc[inside].max(1)) for sc in d['scores']]
    j = int(np.argmax(meds))
    sc = d['scores'][j]
    k = np.argmax(sc, 1)
    rows = np.arange(n_rays)
    peak = sc[rows, k]
    # 抛物线亚像素
    km, kp = np.clip(k - 1, 0, sc.shape[1] - 1), np.clip(k + 1, 0, sc.shape[1] - 1)
    y0, y1, y2 = sc[rows, km], peak, sc[rows, kp]
    den = y0 - 2 * y1 + y2
    with np.errstate(divide='ignore', invalid='ignore'):
        off = np.where((den < 0) & (k > 0) & (k < sc.shape[1] - 1), 0.5 * (y0 - y2) / den, 0.0)
    off = np.clip(off, -0.5, 0.5)
    dist = d['r_ray'] + d['s'][d['iw'][k]] + off * V.step
    thr = max(V.k_sigma * noise, V.min_contrast)
    valid = inside & (peak >= thr)
    m.n_valid = int(valid.sum())
    m.line_w = d['widths'][j]
    if m.n_valid < 8:
        m.reason = 'few_valid'
        return m, None, None, None, n_rays
    px = np.column_stack([cx + d['ct'] * dist, cy + d['st'] * dist])[valid]
    return m, px, peak[valid], valid, n_rays


def _finish(m, pu, px, pk, inl, cen_u, n_rays, V, cam):
    """内点 -> 扇区、残差、对比、圆心（原图 px）。"""
    m.n_inliers = int(inl.sum())
    m.inlier_ratio = m.n_inliers / m.n_valid
    # 扇区有效要求一半以上射线是内点：纹理里的随机峰偶尔落在圆上，但很难占满半个扇区
    need = max(V.sector_min_rays, int(math.ceil(V.sector_min_frac * n_rays / V.sectors)))
    m.sectors, m.max_gap = _sector_stats(pu[inl], cen_u, V.sectors, need)
    m.contrast = float(np.median(pk[inl]))
    m.points = px[inl]
    c = cam.distort(np.asarray(cen_u, np.float64).reshape(1, 2))[0]
    m.cx, m.cy = float(c[0]), float(c[1])


def _one_pass(gray, cam, cx, cy, r, delta, noise, V, rng):
    """圆模型。"""
    m, px, pk, _, n_rays = _peaks(gray, cx, cy, r, delta, noise, V)
    if px is None:
        return m, None
    pu = cam.undistort(px)
    # 内点容差随半径而不是线宽：手画石灰环的不圆度与半径成比例；
    # 容差必须远小于找峰窗口，否则纹理里随机的峰也能凑出高内点率
    tau = max(V.inlier_tol[0], V.inlier_tol[1] * r)
    fit = _fit_circle(pu, tau, V.ransac_iters, rng, (0.5 * r, 1.5 * r))
    if fit is None:
        m.reason = 'no_fit'
        return m, None
    cen_u, ru, inl = fit
    res = np.hypot(pu[inl, 0] - cen_u[0], pu[inl, 1] - cen_u[1]) - ru
    m.rms = float(np.sqrt(np.mean(res ** 2)))
    m.r = float(ru)
    _finish(m, pu, px, pk, inl, cen_u, n_rays, V, cam)
    return m, cen_u


def _ellipse_pass(gray, cam, m0, noise, V, rng):
    """椭圆模型：斜视时圆环在画面里是椭圆，圆模型会把长短轴两端判成外点。
    先用圆模型的有效点粗拟椭圆，再沿椭圆的预测半径重新找峰、拟合。"""
    if m0.points is None or len(m0.points) < 8:
        return None
    best = None
    for it in range(2):
        if best is None:
            # 初值：圆模型的全部有效峰（不只是内点）重新找一遍，窗口放宽到轴比 0.6
            m, px, pk, _, n_rays = _peaks(gray, m0.cx, m0.cy, m0.r, max(_delta(V.delta_search, m0.r), 0.25 * m0.r), noise, V)
        else:
            ex, ey, a, b, ang = best.ellipse
            th = np.arange(V.rays_large if best.r >= V.rays_large_r else V.rays_small)
            th = th * (2 * math.pi / len(th))
            rr = _ellipse_radius(a, b, ang, th)
            m, px, pk, _, n_rays = _peaks(gray, ex, ey, (a + b) / 2, _delta(V.delta_track, (a + b) / 2), noise, V, rr)
        if px is None or len(px) < 12:
            return best
        pu = cam.undistort(px)
        # RANSAC：5 点椭圆，取内点最多的
        tau = max(V.inlier_tol[0], V.inlier_tol[1] * m0.r)
        best_inl, best_e = None, None
        n = len(pu)
        samples = np.array([rng.choice(n, 5, replace=False) for _ in range(60)], np.int32)
        native = ellipse_hypotheses(pu, samples, m0.r, tau, V.ellipse_min_axis_ratio)
        if native is not None:
            best_e, best_inl = native
        for idx in samples if native is None else ():
            try:
                (ex, ey), (d1, d2), ang = cv2.fitEllipseDirect(pu[idx].astype(np.float32))
            except cv2.error:
                continue
            a, b = d1 / 2, d2 / 2
            if not (np.isfinite(a) and np.isfinite(b)) or min(a, b) < 0.5 * m0.r or max(a, b) > 1.6 * m0.r:
                continue
            if min(a, b) / max(a, b) < V.ellipse_min_axis_ratio:
                continue
            inl = np.abs(_ellipse_residual(pu, ex, ey, a, b, ang)) <= tau
            if best_inl is None or inl.sum() > best_inl.sum():
                best_inl, best_e = inl, (ex, ey, a, b, ang)
        if best_inl is None or best_inl.sum() < 12:
            return best
        # 内点上最小二乘精修两次
        inl = best_inl
        for _ in range(2):
            (ex, ey), (d1, d2), ang = cv2.fitEllipseDirect(pu[inl].astype(np.float32))
            a, b = d1 / 2, d2 / 2
            if min(a, b) <= 0 or min(a, b) / max(a, b) < V.ellipse_min_axis_ratio:
                return best
            inl = np.abs(_ellipse_residual(pu, ex, ey, a, b, ang)) <= tau
            if inl.sum() < 12:
                return best
        res = _ellipse_residual(pu[inl], ex, ey, a, b, ang)
        m.rms = float(np.sqrt(np.mean(res ** 2)))
        m.r = float((a + b) / 2)
        _finish(m, pu, px, pk, inl, (ex, ey), n_rays, V, cam)
        m.ellipse = (m.cx, m.cy, float(a), float(b), float(ang))
        best = m
    return best


def measure(gray, cam, cx, cy, r, noise, V, mode='lock', delta=None, seed=0):
    """在 (cx, cy, r) 附近测量圆环。

    mode='lock'：首锁门槛，(cx, cy, r) 是搜索候选；
    mode='track'：跟踪门槛，(cx, cy, r) 是预测值。
    """
    if delta is None:
        delta = _delta(V.delta_search if mode == 'lock' else V.delta_track, r)
    rng = np.random.default_rng(seed)
    m, _ = _one_pass(gray, cam, cx, cy, r, delta, noise, V, rng)
    if m.reason:
        return m
    # 第二遍：射线从新圆心出发与圆环垂直；窗口保持不变，内点率才有区分力
    m2, _ = _one_pass(gray, cam, m.cx, m.cy, m.r, delta, noise, V, rng)
    if not m2.reason:
        m = m2
    # 圆模型扇区不够或残差偏大时试椭圆模型（斜视）；椭圆扇区更多才采用
    if V.use_ellipse and (m.sectors < V.sectors or m.rms > _delta(V.lock_max_rms, m.r)):
        me = _ellipse_pass(gray, cam, m, noise, V, rng)
        if me is not None and (me.sectors, -me.rms) > (m.sectors, -m.rms):
            m = me
    # 门槛
    lock = mode == 'lock'
    min_sec = V.lock_sectors if lock else V.track_sectors
    max_gap = V.lock_max_gap if lock else V.track_max_gap
    shift_lim = V.lock_shift * r if lock else _delta(V.track_shift, r)
    r_tol = V.lock_radius_tol if lock else V.track_radius_tol
    if m.sectors < min_sec:
        m.reason = f'sectors {m.sectors}<{min_sec}'
    elif m.max_gap > max_gap:
        m.reason = f'gap {m.max_gap}>{max_gap}'
    elif m.inlier_ratio < (V.lock_min_inlier_ratio if lock else V.min_inlier_ratio):
        m.reason = f'inliers {m.inlier_ratio:.2f}'
    elif m.rms > _delta(V.lock_max_rms if lock else V.max_rms, m.r):
        m.reason = f'rms {m.rms:.2f}'
    elif abs(m.r - r) > r_tol * r:
        m.reason = f'radius {m.r:.1f} vs {r:.1f}'
    elif math.hypot(m.cx - cx, m.cy - cy) > shift_lim:
        m.reason = f'shift {math.hypot(m.cx - cx, m.cy - cy):.1f}'
    else:
        m.ok = True
    return m
