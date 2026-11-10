"""状态机 SEARCH -> TRACK -> COAST -> REACQUIRE，对外状态 DETECTED / PREDICTED / LOST。

每帧：估计地面运动 T -> 用 T 搬运点击框、候选、目标 -> 跟踪复核 或 搜索+候选确认 -> 输出。
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple

import numpy as np
import cv2

from .camera import Camera
from .motion import GroundMotion
from .params import Params
from . import edcircles as edc
from .frame import Frame, frame_noise  # noqa: F401  frame_noise 供工具脚本导入
from .search import RingSearcher, build_pyramid, merge_candidates, plan_ed, run_ed, confirm_ed, ED_SCORE
from .verify import measure, _delta

SEARCH, TRACK, COAST, REACQUIRE = 'SEARCH', 'TRACK', 'COAST', 'REACQUIRE'
DETECTED, PREDICTED, LOST = 'DETECTED', 'PREDICTED', 'LOST'


@dataclass
class Result:
    frame_id: int
    stamp: float
    state: str                                   # DETECTED / PREDICTED / LOST
    phase: str                                   # 内部状态
    target_id: int
    center_px: Optional[Tuple[float, float]]     # 原图 px
    bearing: Optional[Tuple[float, float]]       # 去畸变归一化坐标 (x/z, y/z)
    radius_px: Optional[float]
    ellipse: Optional[Tuple[float, ...]]         # (cx, cy, a, b, angle_deg)
    sectors: int
    rms_px: Optional[float]
    line_w_px: Optional[float]
    age_ms: Optional[float]                      # 距上次 DETECTED
    ambiguous: bool
    click_px: Optional[Tuple[float, float]]      # 点击点在当前帧的位置（光流搬运）；首锁后、未点击时为 None
    click_bearing: Optional[Tuple[float, float]]
    motion_ok: bool                              # 本帧地面光流是否有效
    debug: dict = field(default_factory=dict)

    def to_dict(self, debug=False):
        d = asdict(self)
        if not debug:
            d.pop('debug')
        return d


class _Cand:
    """确认中的候选。"""

    def __init__(self, m, stamp, window):
        self.cx, self.cy, self.r = m.cx, m.cy, m.r
        self.hits = deque([True], maxlen=window)
        self.meas = m
        self.last_pass = stamp
        self.ed_hits = int(m.ed_ok)            # 有几帧得到 EDCircles 支持（全图检出或候选级放大检出）


def measure_channels(fr, cam, cx, cy, r, V, mode, delta=None):
    """按 V.channels 的顺序在各通道复核，第一个通过的返回；都不过时返回第一个通道的结果。"""
    first = None
    for ch in fr.channels(V.channels):
        m = measure(fr.f32(ch), cam, cx, cy, r, fr.noise(ch), V, mode=mode, delta=delta)
        m.channel = ch
        if m.ok:
            return m
        if first is None:
            first = m
    return first


def ed_support(fr, m, S, V):
    """复核通过的结果 m 有没有 EDCircles 支持：全图检出的直接算；否则在各通道上做候选级放大检测。"""
    if m.source == 'ed':
        return True
    if not S.ed_local:
        return False
    return any(confirm_ed(fr.u8(ch), m.cx, m.cy, m.r, S) for ch in fr.channels(V.channels))


class RingTracker:
    def __init__(self, params=None, camera=None):
        self.P = params or Params()
        self.cam = camera
        self.searcher = RingSearcher(self.P)
        self.motion = GroundMotion(self.P.motion)
        self.height = None
        self.target_id = 0
        self.reset()

    # ---------- 外部输入 ----------
    def reset(self):
        self.phase = SEARCH
        self.cands = []
        self.tgt = None            # [cx, cy, r]，当前帧坐标
        self.tgt_meas = None
        self.last_detect = None
        self.click = None
        self.reacq = None
        self.r_floor = None        # 丢失后的半径下限（见 track.lost_r_floor）
        self.cursor = 0
        self.ed_cursor = 0
        self.n_frames = 0
        self._pending_click = None
        self.motion.reset()

    def set_click(self, x, y):
        """G30 点击，已换算成原图 px。下一帧生效。"""
        self._pending_click = (float(x), float(y))

    def set_height(self, h):
        """可靠离地高度（m），只用来缩小半径搜索范围；None 表示没有。"""
        self.height = float(h) if h and h > 0 else None

    # ---------- 主流程 ----------
    def process(self, frame, stamp, frame_id=0):
        t0 = time.perf_counter()
        fr = Frame(frame, self.P.search.noise_floor)
        gray_u8 = fr.u8('gray')
        H, W = gray_u8.shape
        if self.cam is None:
            self.cam = Camera.default(W, H)
        elif self.cam.size != (W, H):
            raise ValueError(f'相机参数分辨率 {self.cam.size} 与图像 {(W, H)} 不一致')
        self.n_frames += 1
        noise = fr.noise('gray')
        T = self.motion.update(gray_u8)
        self._propagate(T)
        if self._pending_click is not None:
            self._apply_click(stamp)
        self.out = LOST
        self.ambiguous = False
        self.dbg = dict(noise=noise, motion_inliers=self.motion.last_inliers,
                        motion_ok=T is not None, found=[], region=None)
        if self.phase in (TRACK, COAST):
            self._step_track(fr, T, stamp)
        if self.phase in (SEARCH, REACQUIRE):
            self._step_search(fr, T, stamp)
        self.dbg['cands'] = [(c.cx, c.cy, c.r, int(sum(c.hits))) for c in self.cands]
        self.dbg['ms'] = (time.perf_counter() - t0) * 1000
        return self._result(frame_id, stamp)

    def _propagate(self, T):
        if T is None:
            return
        s = T.scale
        for c in self.cands:
            c.cx, c.cy = T.apply(c.cx, c.cy)
            c.r *= s
        if self.tgt is not None:
            x, y = T.apply(self.tgt[0], self.tgt[1])
            self.tgt = [x, y, self.tgt[2] * s]
        if self.click is not None:
            self.click['x'], self.click['y'] = T.apply(self.click['x'], self.click['y'])
            self.click['side'] *= s
            self.click['scale'] *= s       # 点击以来地面放大了多少倍（下降）
        if self.reacq is not None:
            self.reacq['x'], self.reacq['y'] = T.apply(self.reacq['x'], self.reacq['y'])
            self.reacq['r'] *= s

    def _apply_click(self, stamp):
        x, y = self._pending_click
        self._pending_click = None
        if self.phase in (TRACK, COAST) and self.tgt is not None:
            if math.hypot(x - self.tgt[0], y - self.tgt[1]) <= max(2 * self.tgt[2], 50):
                return                  # 点的就是当前目标，继续跟
        self.phase = SEARCH
        self.cands = []
        self.reacq = None
        self.r_floor = None
        self.tgt = None
        self.cursor = 0
        self.ed_cursor = 0
        self.click = dict(x=x, y=y, side=self.P.search.click_box, level=0, t_level=stamp, scale=1.0)

    # ---------- 跟踪 ----------
    def _step_track(self, fr, T, stamp):
        V, TP = self.P.verify, self.P.track
        cx, cy, r = self.tgt
        since = stamp - self.last_detect
        if T is None:
            delta = _delta(V.delta_search, r)
        elif self.phase == COAST:
            delta = min(_delta(V.delta_search, r), _delta(V.delta_track, r) * (1 + since / TP.coast_s))
        else:
            delta = None
        m = measure_channels(fr, self.cam, cx, cy, r, V, 'track', delta)
        self.dbg['track_meas'] = m
        if m.ok:
            self.phase = TRACK
            self.tgt = [m.cx, m.cy, m.r]
            self.tgt_meas = m
            self.last_detect = stamp
            self.out = DETECTED
        elif T is None or since > TP.coast_s:
            self.phase = REACQUIRE
            self.reacq = dict(start=stamp, x=cx, y=cy, r=r)
            self.cands = []
            self.cursor = 0
            self.ed_cursor = 0
        else:
            self.phase = COAST
            self.out = PREDICTED

    # ---------- 搜索与确认 ----------
    def _region(self, W, H):
        TP = self.P.track
        ref = ref_r = None
        if self.phase == REACQUIRE:
            q = self.reacq
            el = self._stamp - q['start']
            if el > TP.reacq_giveup_s:
                self.r_floor = q['r'] * TP.lost_r_floor
                self.phase = SEARCH
                self.reacq = None
                self.cands = []
                self.cursor = 0
                self.ed_cursor = 0
            else:
                k = TP.reacq_radius if el <= TP.reacq_expand_s else 2 * TP.reacq_radius
                r_lo, r_hi = q['r'] * (1 - k), q['r'] * (1 + k)
                if el <= TP.reacq_expand_s:
                    h = TP.reacq_box_r * q['r'] / 2
                    region = (q['x'] - h, q['y'] - h, q['x'] + h, q['y'] + h)
                else:
                    region = (0, 0, W, H)
                return region, r_lo, r_hi, (q['x'], q['y']), q['r']
        S = self.P.search
        if self.height:
            r0 = self.cam.fx * (self.P.ring.diameter_m / 2) / self.height
            r_lo, r_hi = r0 * S.height_band[0], r0 * S.height_band[1]
        else:
            r_lo, r_hi = S.r_min, S.r_max
            if self.click is not None:
                # 没有高度时，半径上限随点击以来的地面放大倍数增大：高空看不清、降低后才出现的环也在范围内。
                # 下限不跟着放大，否则点击时小于 r_min 的环会一直追不上下限。
                r_hi = min(S.r_max * max(1.0, self.click['scale']), 0.5 * math.hypot(W, H))
        if self.r_floor:
            # 最大可见半径约为画面对角线的一半，超过它的环已经看不全
            r_lo = max(r_lo, self.r_floor)
            r_hi = max(r_hi, min(1.6 * self.r_floor, 0.5 * math.hypot(W, H)))
        region = (0, 0, W, H)
        ref = (W / 2, H / 2)          # 没有点击时：飞机基本在目标上方，画面中心作参考
        if self.click is not None:
            c = self.click
            ref = (c['x'], c['y'])
            h = c['side'] * (2 ** c['level']) / 2
            region = (c['x'] - h, c['y'] - h, c['x'] + h, c['y'] + h)
        elif S.click_only and self.phase == SEARCH and not self.r_floor:
            region = (0, 0, 0, 0)     # 还没点击：不搜索
        return region, r_lo, r_hi, ref, ref_r

    def _same(self, c, x, y, r):
        TP = self.P.track
        return (math.hypot(c.cx - x, c.cy - y) <= _delta(TP.assoc, max(c.r, r))
                and abs(c.r - r) <= TP.assoc_radius * max(c.r, r))

    def _step_search(self, fr, T, stamp):
        P, V, TP = self.P, self.P.verify, self.P.track
        S = P.search
        H, W = fr.shape
        self._stamp = stamp
        region, r_lo, r_hi, ref, ref_r = self._region(W, H)
        region = (max(0, region[0]), max(0, region[1]), min(W, region[2]), min(H, region[3]))
        self.dbg['region'] = region
        self.dbg['r_range'] = (r_lo, r_hi)
        any_pass = False

        # 1. 已有候选：搬运后在当前帧复核；还没有 EDCircles 支持的，每帧再试一次候选级放大检测
        for c in self.cands:
            d = _delta(V.delta_search if T is None else V.delta_track, c.r)
            m = measure_channels(fr, self.cam, c.cx, c.cy, c.r, V, 'lock', d)
            c.hits.append(m.ok)
            if m.ok:
                c.cx, c.cy, c.r, c.meas, c.last_pass = m.cx, m.cy, m.r, m, stamp
                any_pass = True
                if c.ed_hits == 0 and ed_support(fr, m, S, V):
                    c.ed_hits = 1

        # 2. 搜索：每帧先跑一格 EDCircles（各通道轮流），再跑匹配滤波分块直到用完时间预算（每样至少一块）；
        #    关掉匹配滤波时预算全给 EDCircles。新候选都走同一套复核。
        found = []
        ed_chans = fr.channels(V.channels)
        t0 = time.perf_counter()
        ed_jobs = plan_ed(region, (W, H), r_hi, S.ed_cell) if S.use_edcircles and edc.backend(S.ed_backend) else []
        mf_jobs = self.searcher.plan(region, (W, H), r_lo, r_hi) if S.use_matched_filter else []
        n_ed = n_mf = 0
        while n_ed < len(ed_jobs):
            self.ed_cursor %= len(ed_jobs)
            ch = ed_chans[(self.n_frames + self.ed_cursor // len(ed_jobs)) % len(ed_chans)]
            found += run_ed(fr.u8(ch), ed_jobs[self.ed_cursor], r_lo, r_hi, S)
            self.ed_cursor += 1
            n_ed += 1
            if mf_jobs or (time.perf_counter() - t0) * 1000 >= S.budget_ms:
                break
        if mf_jobs:
            pyr = build_pyramid(fr.f32('gray'), max(j.level for j in mf_jobs))
            while n_mf < len(mf_jobs):
                self.cursor %= len(mf_jobs)
                found += self.searcher.run(pyr, mf_jobs[self.cursor])
                self.cursor += 1
                n_mf += 1
                if (time.perf_counter() - t0) * 1000 >= S.budget_ms:
                    break
        self.dbg['jobs'] = dict(ed=(n_ed, len(ed_jobs)), mf=(n_mf, len(mf_jobs)))
        found = merge_candidates(found)
        self.dbg['found'] = [(f.x, f.y, f.r) for f in found]
        n_ver = 0
        for f in found:
            if n_ver >= TP.max_verify:
                break
            same = [c for c in self.cands if self._same(c, f.x, f.y, f.r)]
            if same:
                if f.score >= ED_SCORE:
                    for c in same:
                        c.ed_hits += 1
                continue
            n_ver += 1
            m = measure_channels(fr, self.cam, f.x, f.y, f.r, V, 'lock')
            m.source = 'ed' if f.score >= ED_SCORE else 'mf'
            if not m.ok or not (r_lo / 1.15 <= m.r <= r_hi * 1.15):
                continue
            if any(self._same(c, m.cx, m.cy, m.r) for c in self.cands):
                continue
            m.ed_ok = ed_support(fr, m, S, V)
            self.cands.append(_Cand(m, stamp, TP.confirm_window))
            any_pass = True

        # 3. 清理：连续 3 帧不过、或太久没过的丢掉；两个候选测到同一个环的合并
        keep = []
        for c in sorted(self.cands, key=lambda c: (-sum(c.hits), -c.last_pass)):
            h = list(c.hits)
            if len(h) >= 3 and not any(h[-3:]):
                continue
            if stamp - c.last_pass > TP.cand_timeout_s:
                continue
            if any(self._same(k, c.cx, c.cy, c.r) for k in keep):
                continue
            keep.append(c)
        self.cands = keep[:TP.max_cands]

        # 4. 点击框逐级扩大
        if self.phase == SEARCH and self.click is not None:
            ck = self.click
            if any_pass:
                ck['t_level'] = stamp
            elif stamp - ck['t_level'] >= P.search.escalate_s and ck['level'] < P.search.click_max_levels:
                ck['level'] += 1
                ck['t_level'] = stamp
                self.cursor = 0
                self.ed_cursor = 0

        # 5. 首锁 / 重捕确认。首锁要求有 EDCircles 支持（全图检出或候选级放大检出）：
        #    只靠匹配滤波通过的大多是纹理小圈。重捕有最后位置约束，不要求。
        need_ed = self.phase == SEARCH and TP.lock_need_edcircles and S.use_edcircles and bool(ed_jobs)
        q = [c for c in self.cands if c.hits[-1] and sum(c.hits) >= TP.confirm_hits
             and (not need_ed or c.ed_hits > 0)]
        win = self._choose(q, ref, ref_r)
        if win is not None:
            if self.phase == SEARCH:
                self.target_id += 1
            self.phase = TRACK
            self.tgt = [win.cx, win.cy, win.r]
            self.tgt_meas = win.meas
            self.last_detect = stamp
            self.cands = []
            self.reacq = None
            self.click = None
            self.out = DETECTED
        elif len(q) >= 2:
            self.ambiguous = True

    def _choose(self, q, ref, ref_r):
        TP = self.P.track
        if not q:
            return None
        if len(q) == 1:
            return q[0]
        if ref is None:
            return None
        dist = [math.hypot(c.cx - ref[0], c.cy - ref[1]) for c in q]
        if self.phase == REACQUIRE and ref_r:
            near = [c for c, d in zip(q, dist) if d <= TP.prefer_dist_r * ref_r]
            if len(near) == 1:
                return near[0]
        order = np.argsort(dist)
        if dist[order[0]] < TP.prefer_ratio * dist[order[1]]:
            return q[order[0]]
        return None

    # ---------- 输出 ----------
    def _result(self, frame_id, stamp):
        center = bearing = radius = ell = rms = lw = None
        sectors = 0
        if self.out == DETECTED:
            m = self.tgt_meas
            center, radius, ell = (m.cx, m.cy), m.r, m.ellipse
            sectors, rms, lw = m.sectors, m.rms, m.line_w
        elif self.out == PREDICTED:
            center, radius = (self.tgt[0], self.tgt[1]), self.tgt[2]
        if center is not None:
            bearing = self.cam.bearing(center)
        age = None if self.last_detect is None else (stamp - self.last_detect) * 1000
        click = click_b = None
        if self.click is not None:
            click = (float(self.click['x']), float(self.click['y']))
            click_b = self.cam.bearing(click)
        return Result(frame_id, stamp, self.out, self.phase, self.target_id, center, bearing,
                      radius, ell, sectors, rms, lw, age, self.ambiguous, click, click_b,
                      bool(self.dbg['motion_ok']), self.dbg)


def detect_frame(img, cam, params, region=None, r_range=None, limit=200):
    """单张图：全区域搜索 + 单帧复核（没有多帧确认）。img 为 BGR 或灰度。
    返回 [(Candidate, Measurement)], 灰度噪声。m.ed_ok 表示有 EDCircles 支持，与跟踪器的首锁条件一致。"""
    P, S, V = params, params.search, params.verify
    fr = Frame(img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8), S.noise_floor)
    H, W = fr.shape
    region = region or (0, 0, W, H)
    r_lo, r_hi = r_range or (S.r_min, S.r_max)
    found = []
    if S.use_edcircles and edc.backend(S.ed_backend):
        for ch in fr.channels(V.channels):
            for j in plan_ed(region, (W, H), r_hi, S.ed_cell):
                found += run_ed(fr.u8(ch), j, r_lo, r_hi, S)
    if S.use_matched_filter:
        s = RingSearcher(P)
        jobs = s.plan(region, (W, H), r_lo, r_hi)
        if jobs:
            pyr = build_pyramid(fr.f32('gray'), max(j.level for j in jobs))
            found += [c for j in jobs for c in s.run(pyr, j)]
    out = []
    for c in merge_candidates(found, limit):
        m = measure_channels(fr, cam, c.x, c.y, c.r, V, 'lock')
        m.source = 'ed' if c.score >= ED_SCORE else 'mf'
        if m.ok:
            m.ed_ok = S.use_edcircles and bool(edc.backend(S.ed_backend)) and ed_support(fr, m, S, V)
        out.append((c, m))
    return out, fr.noise('gray')
