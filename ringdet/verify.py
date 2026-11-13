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

from .edcircles import ellipse_fit, huber_circle, peak_pick, ransac_circle, ray_scores


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


def _ray_scores(gray, cx, cy, r, delta, n_rays, V, r_ray=None, widths=None):
    """沿射线采样并对每档线宽打分。

    r_ray：每条射线的预测半径（椭圆时用），None 时都等于 r。采样半径 s 是相对 r_ray 的偏移。
    widths：只试这些线宽（快路径用上一帧实测线宽），None 时按 r 生成全部候选。
    返回 dict: thetas, s (相对偏移), iw (找峰窗口下标), inside (射线全在图内),
              scores[j] (n_rays x len(iw))，widths, r_ray
    """
    H, W = gray.shape
    widths = _widths(r, V.width_ratio) if widths is None else [float(w) for w in widths]
    # 采样步长随找峰窗口放大：每条射线在窗口内保持约 80 个采样点。
    # 固定 0.5px 时 r=190、delta=57 一条射线要采 300+ 点，大环复核耗时随 r 线性增长。
    step = max(V.step, float(delta) / 40.0)
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
    iw = np.nonzero((s >= -delta) & (s <= delta))[0]
    n = len(s)
    scores = ray_scores(prof, iw, widths, step, V.gap, V.side_ratio, inside) if V.backend != 'python' else None
    meds = None
    if scores is not None:
        scores, meds = scores
    cs = None
    if scores is None:
        cs = np.zeros((n_rays, len(s) + 1), np.float64)
        np.cumsum(prof, axis=1, out=cs[:, 1:])
        scores = []
    for w in widths if cs is not None else ():
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
    return dict(thetas=th, ct=ct, st=st, s=s, iw=iw, inside=inside, scores=scores, widths=widths,
                r_ray=rr, meds=meds, step=step)


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


def _huber_circle(pts, c, r, delta, iters=10, native=True):
    """几何距离 + Huber 权的 Gauss-Newton。"""
    fitted = huber_circle(pts, c, r, delta, iters) if native else None
    if fitted is not None:
        return fitted
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


@lru_cache(maxsize=256)
def _circle_samples(n, iters, state, increment, has_uint32, uinteger):
    generator = np.random.Generator(np.random.PCG64())
    generator.bit_generator.state = dict(bit_generator='PCG64', state=dict(state=state, inc=increment),
                                        has_uint32=has_uint32, uinteger=uinteger)
    samples = np.argsort(generator.random((iters, n)), axis=1)[:, :3].copy()
    samples.setflags(write=False)
    return samples, generator.bit_generator.state


def _fit_circle(pts, tau, iters, rng, r_lim, native=True):
    """RANSAC 粗筛 + Huber 精修。r_lim=(lo, hi) 之外的假设不考虑。
    返回 (center, r, inlier_mask) 或 None。"""
    n = len(pts)
    if n < 5:
        return None
    if native and type(rng.bit_generator) is np.random.PCG64:
        state = rng.bit_generator.state
        idx, end_state = _circle_samples(n, iters, state['state']['state'], state['state']['inc'],
                                         state['has_uint32'], state['uinteger'])
        rng.bit_generator.state = end_state
    else:
        idx = np.argsort(rng.random((iters, n)), axis=1)[:, :3]
    if native:
        fitted = ransac_circle(pts, idx, tau, r_lim[0], r_lim[1])
        if fitted is not False:
            return fitted
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
    cen, r = _huber_circle(pts[inl], c[b], rr[b], tau / 2, native=native)
    for _ in range(2):
        if not (r_lim[0] <= r <= r_lim[1]) or not np.all(np.isfinite(cen)):
            return None
        res = np.abs(np.hypot(pts[:, 0] - cen[0], pts[:, 1] - cen[1]) - r)
        inl = res <= tau
        if inl.sum() < 5:
            return None
        cen, r = _huber_circle(pts[inl], cen, r, tau / 2, native=native)
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


def _peaks(gray, cx, cy, r, delta, noise, V, r_ray=None, widths=None):
    """射线找峰。返回 (m 半成品, px 有效峰点, peak 对比, valid 掩码, n_rays) 或 (m, None...)。"""
    n_rays = V.rays_large if r >= V.rays_large_r else V.rays_small
    d = _ray_scores(gray, cx, cy, r, delta, n_rays, V, r_ray, widths)
    inside = d['inside']
    m = Measurement(False, '', n_rays=n_rays, n_inside=int(inside.sum()))
    if m.n_inside < 8:
        m.reason = 'outside'
        return m, None, None, None, n_rays
    # 所有射线用同一档线宽：取"各射线最高分"中位数最大的那档
    if d['meds'] is None:
        meds = [np.median(sc[inside].max(1)) for sc in d['scores']]
    else:
        meds = d['meds']
    j = int(np.argmax(meds))
    step = d['step']
    sc = d['scores'][j]
    thr = max(V.k_sigma * noise, V.min_contrast)
    picked = peak_pick(sc, d['r_ray'], d['s'][d['iw']], step, thr, inside) if V.backend != 'python' else None
    if picked is not None:
        _, peak, dist, valid = picked
    else:
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
        dist = d['r_ray'] + d['s'][d['iw'][k]] + off * step
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


def _one_pass(gray, cam, cx, cy, r, delta, noise, V, rng, widths=None):
    """圆模型。"""
    m, px, pk, _, n_rays = _peaks(gray, cx, cy, r, delta, noise, V, widths=widths)
    if px is None:
        return m, None
    pu = cam.undistort(px)
    # 内点容差随半径而不是线宽：手画石灰环的不圆度与半径成比例；
    # 容差必须远小于找峰窗口，否则纹理里随机的峰也能凑出高内点率
    tau = max(V.inlier_tol[0], V.inlier_tol[1] * r)
    fit = _fit_circle(pu, tau, V.ransac_iters, rng, (0.5 * r, 1.5 * r), native=V.backend != 'python')
    if fit is None:
        m.reason = 'no_fit'
        return m, None
    cen_u, ru, inl = fit
    res = np.hypot(pu[inl, 0] - cen_u[0], pu[inl, 1] - cen_u[1]) - ru
    m.rms = float(np.sqrt(np.mean(res ** 2)))
    m.r = float(ru)
    _finish(m, pu, px, pk, inl, cen_u, n_rays, V, cam)
    return m, cen_u


@lru_cache(maxsize=128)
def _ellipse_samples(n, state, increment, has_uint32, uinteger):
    generator = np.random.Generator(np.random.PCG64())
    generator.bit_generator.state = dict(bit_generator='PCG64', state=dict(state=state, inc=increment),
                                        has_uint32=has_uint32, uinteger=uinteger)
    samples = np.array([generator.choice(n, 5, replace=False) for _ in range(60)], np.int32)
    samples.setflags(write=False)
    return samples, generator.bit_generator.state


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
        # RANSAC：5 点椭圆，取内点最多的；原生路径把内点精修也一次算完
        tau = max(V.inlier_tol[0], V.inlier_tol[1] * m0.r)
        n = len(pu)
        if V.backend != 'python' and type(rng.bit_generator) is np.random.PCG64:
            state = rng.bit_generator.state
            samples, end_state = _ellipse_samples(n, state['state']['state'], state['state']['inc'],
                                                  state['has_uint32'], state['uinteger'])
            rng.bit_generator.state = end_state
        else:
            samples = np.array([rng.choice(n, 5, replace=False) for _ in range(60)], np.int32)
        native = ellipse_fit(pu, samples, m0.r, tau, V.ellipse_min_axis_ratio) if V.backend != 'python' else None
        if native is not None:
            rc, best_e, inl = native
            if rc < 12:              # 没有合格假设（<12 内点）或精修失败：沿用上一轮结果
                return best
            ex, ey, a, b, ang = best_e
        else:
            best_inl = None
            for idx in samples:
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
                    best_inl = inl
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


def _gate_reason(m, cx, cy, r, V, mode):
    lock = mode == 'lock'
    min_sec = V.lock_sectors if lock else V.track_sectors
    max_gap = V.lock_max_gap if lock else V.track_max_gap
    min_inliers = V.lock_min_inlier_ratio if lock else V.min_inlier_ratio
    shift_lim = V.lock_shift * r if lock else _delta(V.track_shift, r)
    r_tol = V.lock_radius_tol if lock else V.track_radius_tol
    if mode == 'local':
        min_sec = max(V.track_sectors, V.lock_sectors - 2)
        max_gap = min(V.track_max_gap, V.lock_max_gap + 2)
        min_inliers = max(V.min_inlier_ratio, V.lock_min_inlier_ratio - .10)
    if lock and r < 20:
        # 半径 20 px 以下的环周长只有 40~120 px：16 个扇区每个只分到 3~8 条射线，
        # 一条射线偏 1 px 整个扇区就作废；快速下降时有运动模糊，实测干净的环
        # （内点率 1.0、RMS 0.23）也只有 9~12 个扇区，按 14 要求会永远锁不上。
        # 扇区门槛随半径折减，内点率、RMS、半径/圆心一致性仍然照旧。
        min_sec = max(V.track_sectors, int(round(V.lock_sectors * max(0.7, r / 20.0))))
        # 缺的扇区常常是连着的（高光/模糊），小环允许空一段，最多 1/4 圈。
        max_gap = max(max_gap, min(V.sectors // 4, V.sectors - min_sec + 1))
    if m.sectors < min_sec:
        return f'sectors {m.sectors}<{min_sec}'
    if m.max_gap > max_gap:
        return f'gap {m.max_gap}>{max_gap}'
    if m.inlier_ratio < min_inliers:
        return f'inliers {m.inlier_ratio:.2f}'
    if m.rms > _delta(V.lock_max_rms if lock else V.max_rms, m.r):
        return f'rms {m.rms:.2f}'
    if abs(m.r - r) > r_tol * r:
        return f'radius {m.r:.1f} vs {r:.1f}'
    shift = math.hypot(m.cx - cx, m.cy - cy)
    if shift > shift_lim:
        return f'shift {shift:.1f}'
    return ''


def measure(gray, cam, cx, cy, r, noise, V, mode='lock', delta=None, seed=0, circle_first=False,
            quick=False, widths=None):
    """在 (cx, cy, r) 附近测量圆环。

    mode='lock'：首锁门槛，(cx, cy, r) 是搜索候选；
    mode='track'：跟踪门槛，(cx, cy, r) 是预测值。
    mode='local'：可靠局部光流内的复核。
    quick=True：只做一遍圆拟合、单档线宽、不试椭圆（稳定跟踪的快路径，失败时调用方再走完整复核）。
    """
    if V.backend not in ('auto', 'python'):
        raise ValueError('verify.backend 必须是 auto 或 python')
    if delta is None:
        delta = _delta(V.delta_search if mode in ('lock', 'local') else V.delta_track, r)
    rng = np.random.default_rng(seed)
    m, _ = _one_pass(gray, cam, cx, cy, r, delta, noise, V, rng, widths=widths)
    if m.reason:
        return m
    if not quick:
        # 第二遍：射线从新圆心出发与圆环垂直；窗口保持不变，内点率才有区分力
        m2, _ = _one_pass(gray, cam, m.cx, m.cy, m.r, delta, noise, V, rng)
        if not m2.reason:
            m = m2
    reason = _gate_reason(m, cx, cy, r, V, mode)
    # 局部光流可靠且圆拟合合格时，不必再拟合椭圆。
    circle_ok = not reason and (mode == 'local' or (
        circle_first and m.sectors >= V.lock_sectors and m.max_gap <= V.lock_max_gap
        and m.inlier_ratio >= V.lock_min_inlier_ratio and m.rms <= _delta(V.lock_max_rms, m.r)))
    if not quick and V.use_ellipse and not circle_ok and (m.sectors < V.sectors or m.rms > _delta(V.lock_max_rms, m.r)):
        me = _ellipse_pass(gray, cam, m, noise, V, rng)
        if me is not None and (me.sectors, -me.rms) > (m.sectors, -m.rms):
            m = me
            reason = _gate_reason(m, cx, cy, r, V, mode)
    m.reason, m.ok = reason, not reason
    return m
