"""候选搜索：EDCircles（ED_Lib）为主，多尺度环带匹配滤波（分块 + FFT）补高空小环和断续的环。

打分 S = min(M - I, M - O) / max(sigma, noise_floor)，且 |I - O| > side_ratio * S 时置 0
M/I/O 是中间亮带、内背景带、外背景带的平均亮度。取两侧对比的较小值，
实心亮斑（I 也亮）和单侧亮边（I 或 O 亮）都接近 0；
|I - O| 条件去掉亮物边缘被锐化/JPEG 振铃出的亮边。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import cv2

from . import edcircles as edc
from .frame import noise_sigma


@dataclass
class Candidate:
    x: float          # 原图 px
    y: float
    r: float
    score: float
    level: int


@dataclass
class Job:
    level: int
    x0: int           # 取块左上角（层内 px，含 margin，可以在图外）
    y0: int
    w: int
    h: int
    margin: int
    valid: Tuple[int, int, int, int]   # 层内有效区 (x0, y0, x1, y1)，左闭右开
    radii: Tuple[float, ...]           # 层内半径


def _annulus(r_in, r_out, half, ss=8):
    """抗锯齿圆环：每像素 ss*ss 超采样的面积占比。"""
    off = (np.arange(ss) + 0.5) / ss - 0.5
    sub = (np.arange(-half, half + 1)[:, None] + off[None, :]).ravel()
    yy, xx = np.meshgrid(sub, sub, indexing='ij')
    d = np.hypot(xx, yy)
    m = ((d >= r_in) & (d < r_out)).astype(np.float64)
    n = 2 * half + 1
    return m.reshape(n, ss, n, ss).mean(axis=(1, 3))


def ring_bands(r, line_ratio, m_min, side_min):
    """返回 (内带, 中间带, 外带) 的边界半径。"""
    wm = max(m_min, line_ratio * r)
    sb = max(side_min, line_ratio * r)
    a1, a2 = r - wm / 2, r + wm / 2
    return max(0.0, a1 - sb), a1, a2, a2 + sb


def ring_kernels(r, line_ratio, m_min, side_min):
    """返回 (k_MI, k_MO, half)，k_MI = 中间带均值核 - 内带均值核。"""
    a0, a1, a2, a3 = ring_bands(r, line_ratio, m_min, side_min)
    half = int(math.ceil(a3)) + 1
    kI, kM, kO = (_annulus(a0, a1, half), _annulus(a1, a2, half), _annulus(a2, a3, half))
    kI /= kI.sum()
    kM /= kM.sum()
    kO /= kO.sum()
    return kM - kI, kM - kO, half


def get_patch(img, x0, y0, w, h):
    """取 [x0, x0+w) x [y0, y0+h)，图外部分镜像填充。"""
    H, W = img.shape[:2]
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(W, x0 + w), min(H, y0 + h)
    if xb <= xa or yb <= ya:
        return None
    patch = img[ya:yb, xa:xb]
    t, b, l, r = ya - y0, y0 + h - yb, xa - x0, x0 + w - xb
    if t or b or l or r:
        patch = cv2.copyMakeBorder(patch, t, b, l, r, cv2.BORDER_REFLECT_101)
    return patch


class RingSearcher:
    def __init__(self, params):
        self.P = params
        self._spec = {}

    # ---------- 规划 ----------
    def radii(self, r_lo, r_hi):
        s = self.P.search.r_step
        n = max(1, int(math.floor(math.log(r_hi / r_lo) / math.log(s))) + 1)
        rs = r_lo * s ** np.arange(n)
        if rs[-1] < r_hi / math.sqrt(s):
            rs = np.append(rs, r_hi)
        return rs

    def _split_levels(self, radii):
        lv = {}
        top = self.P.search.level_r_max
        for r in radii:
            L = 0
            while r > top * (2 ** L) * 1.0001:
                L += 1
            lv.setdefault(L, []).append(r / 2 ** L)
        return lv

    def _margin(self, radii_level):
        S = self.P.search
        ratio = self.P.ring.line_width_m / (self.P.ring.diameter_m / 2)
        a3 = ring_bands(max(radii_level), ratio, S.m_band_min, S.side_band_min)[3]
        return int(math.ceil(a3)) + 2

    def plan(self, region, img_size, r_lo, r_hi):
        """region: 原图 (x0, y0, x1, y1)。返回覆盖该区域、该半径范围的全部分块。"""
        W, H = img_size
        jobs = []
        for L, rl in sorted(self._split_levels(self.radii(r_lo, r_hi)).items()):
            k = 2 ** L
            WL, HL = (W + k - 1) // k, (H + k - 1) // k
            rx0 = max(0, int(math.floor(region[0] / k)))
            ry0 = max(0, int(math.floor(region[1] / k)))
            rx1 = min(WL, int(math.ceil(region[2] / k)) + 1)
            ry1 = min(HL, int(math.ceil(region[3] / k)) + 1)
            if rx1 <= rx0 or ry1 <= ry0:
                continue
            m = self._margin(rl)
            tw = min(self.P.search.tile, cv2.getOptimalDFTSize(rx1 - rx0 + 2 * m))
            th = min(self.P.search.tile, cv2.getOptimalDFTSize(ry1 - ry0 + 2 * m))
            vw, vh = tw - 2 * m, th - 2 * m
            n = self.P.search.radii_per_job
            chunks = [tuple(rl[i:i + n]) for i in range(0, len(rl), n)]
            for ty in range(ry0, ry1, vh):
                for tx in range(rx0, rx1, vw):
                    valid = (tx, ty, min(tx + vw, rx1), min(ty + vh, ry1))
                    for ch in chunks:
                        jobs.append(Job(L, tx - m, ty - m, tw, th, m, valid, ch))
        return jobs

    # ---------- 执行 ----------
    def _spectrum(self, r, shape):
        key = (shape, round(r, 4))
        sp = self._spec.get(key)
        if sp is None:
            S = self.P.search
            ratio = self.P.ring.line_width_m / (self.P.ring.diameter_m / 2)
            kmi, kmo, half = ring_kernels(r, ratio, S.m_band_min, S.side_band_min)
            h, w = shape
            buf = np.zeros((h, w, 2), np.float32)
            n = 2 * half + 1
            buf[:n, :n, 0] = kmi
            buf[:n, :n, 1] = kmo
            buf = np.roll(buf, (-half, -half), axis=(0, 1))   # 核中心放到 (0,0)
            sp = cv2.dft(buf, flags=cv2.DFT_COMPLEX_OUTPUT)
            self._spec[key] = sp
        return sp

    def score_maps(self, level_img, job):
        """返回 (best_score, best_radius_index, sigma)，均为块内坐标。"""
        patch = get_patch(level_img, job.x0, job.y0, job.w, job.h)
        if patch is None:
            return None
        patch = np.ascontiguousarray(patch, np.float32)
        vx0, vy0, vx1, vy1 = job.valid
        core = patch[vy0 - job.y0:vy1 - job.y0, vx0 - job.x0:vx1 - job.x0]
        sigma = max(noise_sigma(core), self.P.search.noise_floor)
        F = cv2.dft(patch, flags=cv2.DFT_COMPLEX_OUTPUT)
        best = np.full(patch.shape, -np.inf, np.float32)
        arg = np.zeros(patch.shape, np.int16)
        for i, r in enumerate(job.radii):
            out = cv2.idft(cv2.mulSpectrums(F, self._spectrum(r, patch.shape), 0),
                           flags=cv2.DFT_SCALE)
            a, b = out[..., 0], out[..., 1]          # M-I, M-O；|I-O| = |a-b|
            s = np.minimum(a, b)
            s[np.abs(a - b) > self.P.search.side_ratio * np.maximum(s, 0)] = 0.0
            upd = s > best
            best[upd] = s[upd]
            arg[upd] = i
        return best / sigma, arg, sigma

    def run(self, pyr, job):
        S = self.P.search
        res = self.score_maps(pyr[job.level], job)
        if res is None:
            return []
        score, arg, _ = res
        k = 2 ** job.level
        vx0, vy0, vx1, vy1 = job.valid
        ox, oy = vx0 - job.x0, vy0 - job.y0
        sv = score[oy:oy + vy1 - vy0, ox:ox + vx1 - vx0]
        av = arg[oy:oy + vy1 - vy0, ox:ox + vx1 - vx0]
        dil = cv2.dilate(sv, np.ones((3, 3), np.uint8))
        ys, xs = np.nonzero((sv >= dil) & (sv >= S.min_score))
        if len(xs) == 0:
            return []
        vals = sv[ys, xs]
        order = np.argsort(-vals)
        cell = max(4, S.cell // k)
        per_cell = {}
        kept = []
        for i in order:
            x, y = xs[i] + vx0, ys[i] + vy0
            r = job.radii[av[ys[i], xs[i]]]
            c = (x // cell, y // cell)
            if per_cell.get(c, 0) >= S.per_cell:
                continue
            if any((x - q[0]) ** 2 + (y - q[1]) ** 2 < (0.5 * max(r, q[2])) ** 2 for q in kept):
                continue
            per_cell[c] = per_cell.get(c, 0) + 1
            kept.append((x, y, r, float(vals[i])))
            if len(kept) >= S.per_tile:
                break
        return [Candidate(x * k, y * k, r * k, s, job.level) for x, y, r, s in kept]


ED_SCORE = 100.0   # EDCircles 候选的分数：圆心半径已经很准，排在匹配滤波候选前面复核


@dataclass
class EdJob:
    cell: Tuple[int, int, int, int]   # 圆心落在这个格子里的结果才保留（原图 px，左闭右开）
    crop: Tuple[int, int, int, int]   # 实际跑 EDCircles 的范围：格子外扩 margin，裁到图内


def plan_ed(region, img_size, r_hi, cell_size):
    """把搜索区域切成不重叠的格子；每个格子外扩 1.25*r_hi + 8 px 跑 EDCircles，
    圆心在格子里的环一定完整落在外扩范围内。"""
    W, H = img_size
    x0, y0 = max(0, int(math.floor(region[0]))), max(0, int(math.floor(region[1])))
    x1, y1 = min(W, int(math.ceil(region[2]))), min(H, int(math.ceil(region[3])))
    if x1 <= x0 or y1 <= y0:
        return []
    m = int(math.ceil(1.25 * r_hi)) + 8
    nx = max(1, int(math.ceil((x1 - x0) / cell_size)))
    ny = max(1, int(math.ceil((y1 - y0) / cell_size)))
    xs = np.linspace(x0, x1, nx + 1).round().astype(int)
    ys = np.linspace(y0, y1, ny + 1).round().astype(int)
    jobs = []
    for j in range(ny):
        for i in range(nx):
            cell = (int(xs[i]), int(ys[j]), int(xs[i + 1]), int(ys[j + 1]))
            crop = (max(0, cell[0] - m), max(0, cell[1] - m), min(W, cell[2] + m), min(H, cell[3] + m))
            jobs.append(EdJob(cell, crop))
    return jobs


def run_ed(gray_u8, job, r_lo, r_hi, S):
    """在 job.crop 上跑 EDCircles（ED_Lib），按半径范围和轴比筛选，返回候选。"""
    x0, y0, x1, y1 = job.crop
    d = edc.detect(gray_u8[y0:y1, x0:x1], grad=S.ed_gradient, anchor=S.ed_anchor, sigma=S.ed_sigma,
                   min_path=S.ed_min_path, validate=S.ed_validate, prefer=S.ed_backend)
    out = []
    for cx, cy, a, b, _, _ in d:
        cx, cy = cx + x0, cy + y0
        r = (a + b) / 2
        if not (r_lo / 1.15 <= r <= r_hi * 1.15):
            continue
        if min(a, b) < S.ed_min_axis_ratio * max(a, b):   # 太扁的不是下视的圆环
            continue
        if not (job.cell[0] <= cx < job.cell[2] and job.cell[1] <= cy < job.cell[3]):
            continue
        out.append(Candidate(float(cx), float(cy), float(r), ED_SCORE, -1))
    return out


def confirm_ed(gray_u8, x, y, r, S):
    """在 (x, y, r) 周围的小块上放大后跑 EDCircles，找到圆心、半径都吻合的圆就返回 True。

    高空细环在原图上只有 1 px 宽、半径 8 px 左右，边缘段太短太弱，全图跑 EDCircles 连不成圆；
    放大到半径约 S.ed_local_target_r 后边缘变长，才能连起来。只放大候选周围一小块，耗时很少。
    """
    k = min(S.ed_local_max_zoom, max(1.0, S.ed_local_target_r / max(r, 1.0)))
    R = int(math.ceil(S.ed_local_margin * r)) + 4
    xi, yi = int(round(x)), int(round(y))
    H, W = gray_u8.shape
    x0, y0, x1, y1 = xi - R, yi - R, xi + R + 1, yi + R + 1
    crop = get_patch(gray_u8, x0, y0, x1 - x0, y1 - y0)
    if crop is None:
        return False
    big = cv2.resize(np.ascontiguousarray(crop), None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    tol_c = max(S.ed_local_center_tol[0], S.ed_local_center_tol[1] * r)
    for cx, cy, a, b, _, _ in edc.detect(big, grad=S.ed_gradient, anchor=S.ed_anchor, sigma=S.ed_sigma,
                                          min_path=S.ed_min_path, validate=S.ed_validate, prefer=S.ed_backend):
        if min(a, b) < S.ed_min_axis_ratio * max(a, b):
            continue
        cx = (cx + 0.5) / k - 0.5 + x0
        cy = (cy + 0.5) / k - 0.5 + y0
        rr = (a + b) / 2 / k
        if math.hypot(cx - x, cy - y) <= tol_c and abs(rr - r) <= S.ed_local_radius_tol * r:
            return True
    return False


def build_pyramid(gray, levels):
    pyr = [gray]
    for _ in range(levels):
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr


def merge_candidates(cands, limit=None):
    """跨块、跨层、跨来源合并：同一位置（距离 < 0.5r）且半径相近（差 < 30%）只留最高分。"""
    out = []
    for c in sorted(cands, key=lambda c: -c.score):
        if any((c.x - o.x) ** 2 + (c.y - o.y) ** 2 < (0.5 * max(c.r, o.r)) ** 2
               and abs(c.r - o.r) < 0.3 * max(c.r, o.r) for o in out):
            continue
        out.append(c)
        if limit and len(out) >= limit:
            break
    return out
