"""状态机 SEARCH -> TRACK -> COAST -> REACQUIRE，对外状态 DETECTED / PREDICTED / LOST。

每帧：估计地面运动 T -> 用 T 搬运点击框、候选、目标 -> 跟踪复核 或 搜索+候选确认 -> 输出。
"""
from __future__ import annotations

import math
import time
from collections import deque, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple

import numpy as np
import cv2

from .camera import Camera
from .motion import GroundMotion
from .params import Params
from . import edcircles as edc
from .frame import Frame, frame_noise, noise_sigma  # noqa: F401  frame_noise 供工具脚本导入
from .search import Candidate, RingSearcher, EdJob, build_pyramid, merge_candidates, plan_ed, run_ed, confirm_ed, ED_SCORE
from .verify import Measurement, measure, _delta

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
        self.created = stamp
        self.ed_hits = int(m.ed_ok)            # 有几帧得到 EDCircles 支持（全图检出或候选级放大检出）
        self.from_click = False                # 由点击直测创建（操作员指定，首锁可豁免 ED 支持）
        self.geometry_hits = 0
        self.reference = None
        self.drift = np.zeros(2)


def measure_channels(fr, cam, cx, cy, r, V, mode, delta=None, local=False, circle_first=False,
                     quick=False, widths=None, channels=None):
    """按 V.channels 的顺序在各通道复核，第一个通过的返回；都不过时返回第一个通道的结果。"""
    first = None
    for ch in (channels or fr.channels(V.channels)):
        noise = fr._floor if local else fr.noise(ch)
        if local:
            H, W = fr.shape
            half = int(math.ceil(2 * r + _delta(V.delta_search, r) + 8))
            patch = fr.f32(ch)[max(0, int(cy) - half):min(H, int(cy) + half + 1),
                               max(0, int(cx) - half):min(W, int(cx) + half + 1)]
            if min(patch.shape) >= 3:
                sample = patch[::2, ::2] if patch.size > 4096 else patch
                if sample.size > 16384 and min(sample.shape) >= 96:
                    h, w = sample.shape
                    tiles = [sample[y:y + 48, x:x + 48]
                             for y in (h // 4 - 24, 3 * h // 4 - 24)
                             for x in (w // 4 - 24, 3 * w // 4 - 24)]
                    sample = np.concatenate(tiles, axis=0)
                noise = max(noise_sigma(sample), fr._floor)
        fr.last_noise = noise
        m = measure(fr.f32(ch), cam, cx, cy, r, noise, V, mode=mode, delta=delta,
                    circle_first=circle_first, quick=quick, widths=widths)
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
    if not S.ed_local or m.r > S.ed_local_max_r:
        # 半径 200 px 时本地 ED 要跑 20~40 ms（主线程），而这么大的环全图 ED 本来就能连成圆，
        # 本地再放大一次既慢又没意义。
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
        self._search_pool = None
        self._search_future = None
        self._search_meta = None
        self._last_search_kind = None
        self._template_seeds = None
        self._gpu_matcher = None
        self._gpu_match_roi = self._gpu_match_template = None
        self._gpu_match_response = None
        self._gpu_peak_mask = None
        self._gpu_reduce_ok = True
        self._gpu_peak_routes = {}
        self.template_backend = 'cpu'
        self.template_backend_error = ''
        self.template_peak_backend = 'cpu'
        self.template_peak_error = ''
        if self.P.track.template_backend not in ('auto', 'cpu', 'cuda'):
            raise ValueError('track.template_backend 必须是 auto、cpu 或 cuda')
        self.reset()

    # ---------- 外部输入 ----------
    def reset(self):
        if self._search_meta is not None:
            self._search_meta['valid'] = False
        self.phase = SEARCH
        self.cands = []
        self.tgt = None            # [cx, cy, r]，当前帧坐标
        self.tgt_meas = None
        self.measured_radius = None
        self._measure_width_ratio = None
        self._measure_channel = None
        self.last_detect = None
        self.click = None
        self.reacq = None
        self.r_floor = None        # 丢失后的半径下限（见 track.lost_r_floor）
        self.cursor = 0
        self.focus_cursor = 0
        self.ed_cursor = 0
        self.n_frames = 0
        self._pending_click = None
        self._click_probe = []
        self._probe_spread = False
        self._click_stamp = -1e9   # 上一次收到点击的时间（用于过滤远程连发）
        self._last_click = None    # 上一次有效点击的位置（同上）
        self.miss_streak = 0
        self._mf_cost_ms = 6.0     # 单个同步匹配滤波任务的实测耗时（指数滑动平均），用来判断预算够不够
        self._verify_cost_ms = 3.0  # 一次完整候选复核的实测耗时，同上
        self._tpl_cost_ms = 1.0    # 一次同步模板匹配的实测耗时，用来决定跟丢当帧走不走快路径
        self.template = None
        self.template_angle = 0.0
        self.keyframe = None
        self.references = []
        self.reference_radius = None
        self.identity_hits = 0
        self._identity_score = -1.0
        self._identity_frame = -10 ** 9
        self.recent_identity_hits = 0
        self.stable_track_hits = 0
        self.coast_motion_lost = False
        self.template_hold = 0
        self.recovery_candidate = None
        self.pending = []
        self.learned = None
        self.online = None
        self._template_frame = -1
        self._template_result = None
        self._template_scale_index = 0
        self._appearance_result = None
        self._match_cache = OrderedDict()
        self._match_cache_bytes = 0
        self._gpu_templates = OrderedDict()
        self._gpu_template_bytes = 0
        self._gpu_disks = OrderedDict()
        self.motion.reset()

    def close(self):
        if self._search_pool is not None:
            if self._search_future is not None:
                self._search_future.cancel()
            self._search_pool.shutdown(wait=True)
            self._search_pool = None
        self._search_future = None
        self._search_meta = None
        self._gpu_templates.clear()
        self._gpu_template_bytes = 0
        self._gpu_disks.clear()

    def warmup(self):
        S = self.P.search
        if S.use_matched_filter and S.fft_cache_mb > 0:
            side = S.click_tile + 64
            for radii in self.searcher._split_levels(self.searcher.radii(
                    min(S.r_min, S.click_r_min), max(S.r_min, min(S.r_max, 60)))).values():
                for radius in radii:
                    self.searcher._spectrum(radius, (side, side))
        if self.motion.backend == 'cuda':
            image = np.zeros((180, 320), np.uint8)
            self.motion._corners(image, self.P.motion.max_corners, self.P.motion.min_distance)
            points = np.array([[[64., 64.]], [[128., 96.]], [[200., 120.]]], np.float32)
            self.motion._flow(image, image, points)
            self.motion.reset()
        if (self.P.track.template_backend != 'cpu' and (self.motion.backend == 'cuda'
                or self.P.track.template_backend == 'cuda' or cv2.cuda.getCudaEnabledDeviceCount() > 0)):
            height = 320 if self.P.track.template_cuda_min_area > 180 * 320 else 180
            image = np.random.default_rng(0).integers(0, 256, (height, 320), dtype=np.uint8)
            self._match_templates(image, [image[20:37, 30:47].copy()] * 3, peak_radius=3)
            if self.template_backend == 'cuda' and self.P.track.compact_gpu_matching and self._gpu_reduce_ok:
                generator = np.random.default_rng(1)
                for bucket, shape in enumerate(((160, 192), (256, 320), (448, 512))):
                    image = generator.integers(0, 256, shape, dtype=np.uint8)
                    template = image[20:45, 30:55].copy()
                    image[-55:-30, -65:-40] = np.clip(template.astype(np.int16)
                        + generator.integers(-20, 21, template.shape), 0, 255).astype(np.uint8)
                    self._match_templates(image, [template])
                    if self.template_backend != 'cuda':
                        break
                    scores = self._gpu_match_response
                    try:
                        reference = self._match_peaks(scores.download(), 3)
                        compact = self._match_peaks(scores, 3, gpu=True)
                        equivalent = (compact is not None and np.allclose(compact[:2], reference[:2], atol=1e-5)
                                      and len(compact[2]) == len(reference[2])
                                      and all(a[1] == b[1] and abs(a[0] - b[0]) <= 1e-5
                                              for a, b in zip(compact[2], reference[2])))
                        gpu_times, cpu_times = [], []
                        for _ in range(3) if equivalent else ():
                            started = time.perf_counter()
                            self._match_peaks(scores, 3, gpu=True)
                            gpu_times.append(time.perf_counter() - started)
                            started = time.perf_counter()
                            self._match_peaks(scores.download(), 3)
                            cpu_times.append(time.perf_counter() - started)
                        self._gpu_peak_routes[bucket] = bool(equivalent
                            and np.median(gpu_times) < .95 * np.median(cpu_times))
                    except (cv2.error, AttributeError, TypeError, RuntimeError) as error:
                        self._gpu_reduce_ok = False
                        self.template_peak_error = str(error)
                        self._gpu_peak_routes[bucket] = False
                        break
                if self._gpu_peak_routes:
                    self.template_peak_backend = 'auto'

    def _poll_search(self, stamp):
        if self._search_future is None or not self._search_future.done():
            return []
        future, meta = self._search_future, self._search_meta
        self._search_future = self._search_meta = None
        try:
            found = future.result()
        except Exception as e:
            self.dbg['ed_error'] = str(e)
            return []
        template = meta.get('kind') == 'template'
        if not meta['valid'] or stamp - meta['stamp'] > (0.30 if meta.get('kind') in ('mf', 'template') else 0.35):
            return []
        A = meta['A']
        scale = math.hypot(A[0, 0], A[1, 0])
        for c in found:
            c.x, c.y = A[:2, :2] @ (c.x, c.y) + A[:2, 2]
            c.r *= scale
        if template:
            self._template_seeds = found
            return []
        return found

    def _submit_search(self, kind, stamp, A, fn, *args):
        if self._search_future is not None:
            return False
        if self._search_pool is None:
            self._search_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ring-search')
        self._search_meta = dict(valid=True, stamp=stamp, A=A, kind=kind)
        self._search_future = self._search_pool.submit(fn, *args)
        self._last_search_kind = kind
        return True

    def _submit_ed(self, fr, job, ch, r_lo, r_hi, zoom, stamp):
        if self._search_future is not None:
            return False
        x0, y0, x1, y1 = job.crop
        crop = fr.u8(ch)[y0:y1, x0:x1].copy()
        cell = (job.cell[0] - x0, job.cell[1] - y0, job.cell[2] - x0, job.cell[3] - y0)
        local = EdJob(cell, (0, 0, x1 - x0, y1 - y0))
        A = np.array([[1., 0., x0], [0., 1., y0], [0., 0., 1.]])
        return self._submit_search('ed', stamp, A, run_ed, crop, local, r_lo, r_hi, self.P.search, zoom)

    def _run_mf(self, image, ch, job):
        self.searcher.begin_frame()
        fr = Frame(image, self.P.search.noise_floor)
        # 用 uint8 金字塔：score_maps 反正会把块转 float32，省掉整帧转换与带宽
        return self.searcher.run(build_pyramid(fr.u8(ch), job.level), job)

    def _mf_budget_ok(self, deadline):
        """同步匹配滤波任务是否还装得进这一帧的预算。

        单个任务在大块+多半径时可达 30~50 ms，光判断"现在还没到 deadline"没用，
        必须在开始前就留够它的耗时，否则这一帧就会被拖长。
        """
        return (deadline - time.perf_counter()) * 1000 >= 0.8 * self._mf_cost_ms

    def _mf_job(self, pyr, job):
        """主循环里同步跑一个匹配滤波任务，并记录实测耗时。"""
        t = time.perf_counter()
        found = self.searcher.run(pyr, job)
        cost = (time.perf_counter() - t) * 1000
        self._mf_cost_ms = 0.7 * self._mf_cost_ms + 0.3 * cost
        return found

    def set_click(self, x, y):
        """G30 点击，已换算成原图 px。下一帧生效。"""
        self._pending_click = (float(x), float(y))

    def set_height(self, h):
        """可靠离地高度（m），只用来缩小半径搜索范围；None 表示没有。"""
        self.height = float(h) if h and h > 0 else None

    # ---------- 主流程 ----------
    def process(self, frame, stamp, frame_id=0):
        t0 = time.perf_counter()
        self._stamp = stamp
        self._deadline = t0 + self.P.track.frame_budget_ms / 1000
        self._identity_cache = {}
        if not self.P.search.async_search:
            self.searcher.begin_frame()
        fr = Frame(frame, self.P.search.noise_floor)
        gray_u8 = fr.u8('gray')
        H, W = gray_u8.shape
        if self.cam is None:
            self.cam = Camera.default(W, H)
        elif self.cam.size != (W, H):
            raise ValueError(f'相机参数分辨率 {self.cam.size} 与图像 {(W, H)} 不一致')
        self.n_frames += 1
        previous = None if self.tgt is None else tuple(self.tgt)
        noise = fr._floor
        self.motion.focus = self.tgt
        if self.tgt is None and self.click is not None:
            self.motion.focus = (self.click['x'], self.click['y'], self.P.search.click_tile / 5)
        T = self.motion.update(gray_u8)
        self._propagate(T)
        if T is None and previous is not None and self.learned is not None:
            dt = stamp - self.learned['frame_stamp']
            age = stamp - self.last_detect
            # 光流失效后继续按学习速度外推整个重捕窗口（原来 0.3s 就冻住，
            # 快速下降时环 0.3s 已经跑出几百像素，重捕框还在原地等）。
            if self.learned['hits'] >= 3 and 0 < dt <= 0.1 and age <= self.P.track.reacq_giveup_s:
                velocity = self.learned['velocity']
                # 丢得越久速度先验越不可靠，按 age 指数衰减，避免一直外推到画面外。
                decay = math.exp(-max(0.0, age) / max(0.3, self.P.track.coast_s * 2))
                scale = math.exp(float(np.clip(velocity[2] * dt, -0.1, 0.1)))
                self.tgt = [previous[0] + velocity[0] * decay * dt,
                            previous[1] + velocity[1] * decay * dt, previous[2] * scale]
                if self.reacq is not None:
                    self.reacq.update(x=self.tgt[0], y=self.tgt[1], r=self.tgt[2])
        if self.learned is not None:
            self.learned['frame_stamp'] = stamp
        if self._pending_click is not None:
            self._apply_click(stamp)
        self.out = LOST
        self.ambiguous = False
        self._appearance_result = None
        self.dbg = dict(noise=noise, motion_inliers=self.motion.last_inliers,
                        motion_ok=T is not None, local_motion=self.motion.local_ok,
                        motion_error=self.motion.error, found=[], region=None, rejected={})
        self._template_seeds = None
        self.pending.extend(self._poll_search(stamp))
        if T is not None and previous is not None:
            self.dbg['motion_step_px'] = math.dist(T.apply(previous[0], previous[1]), previous[:2])
        if self.phase in (TRACK, COAST):
            t_step = time.perf_counter()
            self._step_track(fr, T, stamp)
            self.dbg['t_track_ms'] = round((time.perf_counter() - t_step) * 1000, 2)
        if self.phase in (SEARCH, REACQUIRE):
            t_step = time.perf_counter()
            self._step_search(fr, T, stamp)
            self.dbg['t_search_ms'] = round((time.perf_counter() - t_step) * 1000, 2)
        if self.out != DETECTED:
            self.recent_identity_hits = 0
        elif self._search_meta is not None:
            self._search_meta['valid'] = False
        self.dbg['noise'] = fr.last_noise
        self.dbg['motion_backend'] = self.motion.backend
        self.dbg['template_backend'] = self.template_backend
        self.dbg['mf_cost_ms'] = round(self._mf_cost_ms, 2)
        self.dbg['verify_cost_ms'] = round(self._verify_cost_ms, 2)
        self.dbg['tpl_cost_ms'] = round(self._tpl_cost_ms, 2)
        self.dbg['template_peak_backend'] = self.template_peak_backend
        self.dbg['template_peak_error'] = self.template_peak_error
        self.dbg['online_active'] = self.P.track.online_learning and self.online is not None
        self.dbg['online_updates'] = 0 if self.online is None else self.online['updates']
        self.dbg['online_enabled'] = self.P.track.online_learning
        self.dbg['online_min_radius'] = self.P.track.online_min_radius
        self.dbg['online_age_frames'] = None if self.online is None else self.n_frames - self.online['frame']
        self.dbg['template_hold'] = self.template_hold
        self.dbg['reference_count'] = len(self.references)
        self.dbg['verify_sectors'] = self.P.verify.sectors
        c = self.recovery_candidate
        if c is not None and c.frame == self.n_frames:
            self.dbg['confirmation'] = (len(c.hits), c.needed, bool(c.ed_hits))
        elif self.cands:
            current = [c for c in self.cands if c.last_pass == stamp]
            if current:
                c = max(current, key=lambda c: (bool(c.ed_hits) if self.phase == SEARCH else True, sum(c.hits)))
                self.dbg['confirmation'] = (int(sum(c.hits)), c.needed, bool(c.ed_hits))
        self.dbg['cands'] = [(c.cx, c.cy, c.r, int(sum(c.hits))) for c in self.cands]
        self.dbg['ms'] = (time.perf_counter() - t0) * 1000
        return self._result(frame_id, stamp)

    def _propagate(self, T):
        if T is None:
            if self._search_meta is not None and self._search_meta.get('kind') not in ('mf', 'template'):
                self._search_meta['valid'] = False
            return
        if self._search_meta is not None:
            A = np.eye(3)
            A[:2] = T.A
            self._search_meta['A'] = A @ self._search_meta['A']
        s = T.scale
        if self.reference_radius is not None:
            self.reference_radius *= s
        if self.measured_radius is not None:
            self.measured_radius *= s
        for c in self.cands:
            c.cx, c.cy = T.apply(c.cx, c.cy)
            c.r *= s
            c.drift = T.A[:, :2] @ c.drift
            if c.reference is not None:
                c.reference['angle'] += math.degrees(math.atan2(T.A[1, 0], T.A[0, 0]))
        for c in self.pending:
            c.x, c.y = T.apply(c.x, c.y)
            c.r *= s
        if self.tgt is not None:
            x, y = T.apply(self.tgt[0], self.tgt[1])
            self.tgt = [x, y, self.tgt[2] * s]
        if self.template is not None:
            self.template_angle += math.degrees(math.atan2(T.A[1, 0], T.A[0, 0]))
        if self.keyframe is not None:
            self.keyframe['angle'] += math.degrees(math.atan2(T.A[1, 0], T.A[0, 0]))
        if self.online is not None:
            self.online['angle'] += math.degrees(math.atan2(T.A[1, 0], T.A[0, 0]))
        for ref in self.references:
            ref['angle'] += math.degrees(math.atan2(T.A[1, 0], T.A[0, 0]))
        if self.recovery_candidate is not None:
            c = self.recovery_candidate
            c.cx, c.cy = T.apply(c.cx, c.cy)
            c.r *= s
        if self.click is not None:
            self.click['x'], self.click['y'] = T.apply(self.click['x'], self.click['y'])
            self.click['side'] *= s
            self.click['scale'] *= s       # 点击以来地面放大了多少倍（下降）
        if self.reacq is not None:
            self.reacq['x'], self.reacq['y'] = T.apply(self.reacq['x'], self.reacq['y'])
            self.reacq['r'] *= s

    def _apply_click(self, stamp):
        if self._search_meta is not None:
            self._search_meta['valid'] = False
        x, y = self._pending_click
        self._pending_click = None
        # 远程连接/按住不放会逐帧连发点击，位置几乎不动。同一串里只认第一次：
        # 每帧重置一次就永远攒不够确认帧（实测连发 89 帧，首锁从 2 帧拖到 4.4 秒）。
        # 用"时间 + 距离"一起判断，点错了在别处重新点时仍然照常生效。
        diag = math.hypot(*self.cam.size) if self.cam is not None else 0.0
        if (diag and stamp - self._click_stamp < 0.3 and self._last_click is not None
                and math.hypot(x - self._last_click[0], y - self._last_click[1]) <= 0.02 * diag):
            self._click_stamp = stamp
            return
        self._click_stamp = stamp
        self._last_click = (x, y)
        # 还在搜索同一片区域时重复点击也不重置进度，只更新点击点。
        if (diag and self.phase == SEARCH and self.click is not None
                and stamp - self.click.get('t', -1e9) <= 2.0
                and math.hypot(x - self.click['x'], y - self.click['y']) <= 0.02 * diag):
            # 保留首锁进度，但在点击点重新排一次复核（上一轮的阶梯可能已经扫完了）。
            self.click['x'], self.click['y'], self.click['t'] = x, y, stamp
            if not self.cands:
                self._click_probe = [(x, y, False, None)]
                self._probe_spread = False
            return
        self._click_probe = [(x, y, False, None)]   # 下一帧先在点击点复核，不必等 ED/MF 出候选
        self._probe_spread = False            # 点击点没命中后，是否已排入邻域重试
        self._reprobe_rounds = 0              # 点击候选丢失后围绕最后位置重试过几轮
        self.phase = SEARCH
        self.cands = []
        self.reacq = None
        self.r_floor = None
        self.tgt = None
        self.measured_radius = None
        self.template = None
        self.template_angle = 0.0
        self.keyframe = None
        self.references = []
        self.reference_radius = None
        self.identity_hits = 0
        self._identity_score = -1.0
        self._identity_frame = -10 ** 9
        self.recent_identity_hits = 0
        self.stable_track_hits = 0
        self.coast_motion_lost = False
        self.template_hold = 0
        self.recovery_candidate = None
        self.pending = []
        self.learned = None
        self.online = None
        self._template_frame = -1
        self._template_result = None
        self._template_scale_index = 0
        self._appearance_result = None
        self._match_cache.clear()
        self._match_cache_bytes = 0
        self._gpu_templates.clear()
        self._gpu_template_bytes = 0
        self._gpu_disks.clear()
        self.cursor = 0
        self.focus_cursor = 0
        self.ed_cursor = 0
        self.click = dict(x=x, y=y, side=self.P.search.click_box, level=0, t_level=stamp, scale=1.0,
                          t=stamp)

    # ---------- 跟踪 ----------
    def _identity_patch(self, fr, m):
        half = max(16, int(math.ceil(2.5 * m.r)))
        H, W = fr.shape
        if min(m.cx - half, m.cy - half, W - 1 - m.cx - half, H - 1 - m.cy - half) < 0:
            half = max(16, int(math.ceil(1.5 * m.r)))
        if min(m.cx - half, m.cy - half, W - 1 - m.cx - half, H - 1 - m.cy - half) < 0:
            return None
        size = 2 * half + 1
        if size > 96:
            # 参考图本来就存成 96×96：直接按目标分辨率采样，省掉“大图块 + INTER_AREA”
            patch = cv2.getRectSubPix(fr.u8('gray'), (96, 96), (m.cx, m.cy))
        else:
            patch = cv2.getRectSubPix(fr.u8('gray'), (size, size), (m.cx, m.cy))
        return dict(image=patch, size=size, r=m.r, angle=0.0, stamp=self._stamp)

    def _remember(self, fr, m):
        TP, V = self.P.track, self.P.verify
        self._template_scale_index = 0
        if m.sectors < 12 or m.inlier_ratio < 0.75:
            self.recent_identity_hits = 0
            return
        strong = m.sectors >= V.lock_sectors and m.inlier_ratio >= V.lock_min_inlier_ratio
        if not self.template_hold and strong:
            observation = np.array([m.cx, m.cy, math.log(m.r)])
            shape, alpha = m.line_w / m.r, TP.learn_rate
            if self.learned is None:
                self.learned = dict(observation=observation, stamp=self._stamp, frame_stamp=self._stamp,
                                    velocity=np.zeros(3), hits=1, width=shape, width_var=0.0)
            else:
                history = self.learned
                dt = self._stamp - history['stamp']
                if 0 < dt <= 0.2:
                    velocity = (observation - history['observation']) / dt
                    history['velocity'] = (1 - alpha) * history['velocity'] + alpha * velocity
                    history['hits'] += 1
                else:
                    history['velocity'][:] = 0
                    history['hits'] = 1
                error = shape - history['width']
                history['width'] += alpha * error
                history['width_var'] = (1 - alpha) * (history['width_var'] + alpha * error ** 2)
                history.update(observation=observation, stamp=self._stamp, frame_stamp=self._stamp)
        # 稳定跟踪时身份比对不必每帧做：复用上一帧分数，按 identity_every 定期重算；
        # 上次分数不达标或测量不够强时仍然每帧重算，避免身份漂移。
        every = max(1, int(getattr(TP, 'identity_every', 1)))
        stable = (not self.template_hold and strong
                  and m.sectors >= V.lock_sectors
                  and m.inlier_ratio >= V.lock_min_inlier_ratio)
        due = self.n_frames - self._identity_frame >= every
        if every == 1 or due or not stable or self._identity_score < TP.identity_score:
            score = self._reference_score(fr, m)
            self._identity_score = score
            self._identity_frame = self.n_frames
        else:
            score = self._identity_score
        self.dbg['target_identity_score'] = score
        self.dbg['identity_cached'] = self._identity_frame != self.n_frames
        # 旧参考失配时，短期适应仍须由连续可靠测量支撑。
        recent_score = -1.0
        continuous = (strong and m.max_gap <= V.lock_max_gap
                and m.rms <= _delta(V.lock_max_rms, m.r) and self.dbg.get('motion_ok')
                and not self.template_hold and self.keyframe is not None
                and self._stamp - self.keyframe['stamp'] <= TP.coast_s
                and self.dbg.get('measurement_shift', math.inf) <= max(3.0, 0.15 * m.r,
                                                                       2 * self.dbg.get('motion_error', 0))
                and self.learned is not None and self.learned['hits'] >= 3)
        if score < TP.identity_score and continuous:
            recent_score = self._reference_score(fr, m, [self.keyframe])
            self.dbg['recent_identity_score'] = recent_score
        consistent = recent_score >= max(0.9, TP.identity_score)
        self.recent_identity_hits = self.recent_identity_hits + 1 if consistent else 0
        if self.template_hold:
            return
        ref = self._identity_patch(fr, m)
        if ref is None:
            return
        if strong and (score >= TP.identity_score or continuous):
            self.keyframe = ref.copy()
        if score < TP.identity_score and not consistent:
            return
        patch, size = ref['image'], ref['size']
        confidence = float(np.clip((max(score, recent_score) - TP.identity_score) / max(1e-6, 1 - TP.identity_score), .1, 1))
        confidence *= min(1.0, m.sectors / V.sectors) * min(1.0, m.inlier_ratio / V.lock_min_inlier_ratio)
        alpha = TP.learn_rate * confidence
        if self.template is not None:
            n = patch.shape[0]
            old = cv2.resize(self.template['image'], (n, n), interpolation=cv2.INTER_LINEAR)
            if abs(self.template_angle) > 0.5:
                A = cv2.getRotationMatrix2D(((n - 1) / 2, (n - 1) / 2), -self.template_angle, 1)
                old = cv2.warpAffine(old, A, (n, n), borderMode=cv2.BORDER_REFLECT_101)
            extent = (size / m.r) / (self.template['size'] / self.template['r'])
            rate = min(TP.learn_rate, .05) if consistent else TP.learn_rate
            # 采样范围改变后，旧图块不能按相同物理尺度混合。
            adaptive = (cv2.addWeighted(old, 1 - rate, patch, rate, 0)
                        if abs(math.log(extent)) <= math.log(1.08) else patch.copy())
        else:
            adaptive = patch.copy()
        self.template = dict(image=adaptive, size=size, r=m.r)
        self.template_angle = 0.0
        promote = (self.recent_identity_hits >= TP.template_update_hits
                   or score >= TP.identity_score and self.identity_hits >= TP.template_update_hits)
        if not self.references or strong and promote:
            if len(self.references) < 4:
                self.references.append(ref)
            else:
                key, recent = self.references[1], self.references[-2]
                if (abs(math.log(recent['r'] / key['r'])) >= math.log(1.08)
                        or abs(recent['angle'] - key['angle']) >= 10):
                    key = recent
                self.references = [self.references[0], key, self.references[-1], ref]
            self.reference_radius = m.r
            self.identity_hits = 0
            self.recent_identity_hits = 0
        if strong and TP.online_learning and m.r >= TP.online_min_radius:
            self._update_online(fr, m, alpha)

    @staticmethod
    def _online_features(gray, white):
        gray, white = gray.astype(np.float32) / 255, white.astype(np.float32) / 255
        contrast = gray - cv2.GaussianBlur(gray, (0, 0), 1.5)
        scale = np.sqrt(cv2.GaussianBlur(contrast * contrast, (0, 0), 2) + .0004)
        dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) / 8
        dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) / 8
        return np.stack((np.clip(contrast / scale, -3, 3),
                         np.clip((white - cv2.GaussianBlur(white, (0, 0), 1.5)) / scale, -3, 3),
                         np.clip(cv2.magnitude(dx, dy) / scale, 0, 3)))

    @staticmethod
    def _response_psr(response, point, valid):
        x, y = point
        background = valid.copy()
        background[max(0, y - 4):y + 5, max(0, x - 4):x + 5] = False
        values = response[background]
        if values.size < 16:
            return 0.0
        return float((response[y, x] - np.mean(values)) / max(float(np.std(values)), 1e-6))

    def _update_online(self, fr, m, alpha):
        model, TP = self.online, self.P.track
        if model is not None and self.n_frames - model['frame'] < TP.online_update_frames:
            return
        H, W = fr.shape
        side = max(16, int(round(6 * m.r)))
        if min(m.cx - side / 2, m.cy - side / 2, W - 1 - m.cx - side / 2, H - 1 - m.cy - side / 2) < 0:
            return
        n = 48
        images = []
        for ch in ('gray', 'white' if fr._bgr is not None else 'gray'):
            image = cv2.getRectSubPix(fr.u8(ch), (side, side), (m.cx, m.cy))
            image = cv2.resize(image, (n, n), interpolation=cv2.INTER_AREA)
            if model is not None and abs(model['angle']) > .5:
                A = cv2.getRotationMatrix2D(((n - 1) / 2, (n - 1) / 2), model['angle'], 1)
                image = cv2.warpAffine(image, A, (n, n), borderMode=cv2.BORDER_REFLECT_101)
            images.append(image)
        spectrum = np.fft.fft2(self._online_features(*images), norm='ortho')
        if model is None:
            yy, xx = np.indices((n, n))
            radius = n / 6
            distance = np.hypot(np.minimum(yy, n - yy), np.minimum(xx, n - xx))
            support = np.where(distance < 1.65 * radius,
                               .15 + .85 * np.exp(-((distance - radius) / (.3 * radius)) ** 2), 0)
            target = np.exp(-((xx - (n - 1) / 2) ** 2 + (yy - (n - 1) / 2) ** 2) / (2 * 1.25 ** 2))
            model = dict(angle=0.0, mask=support, target=np.fft.fft2(target, norm='ortho'),
                         spectrum=spectrum, filter=np.zeros_like(spectrum), updates=0)
        else:
            response = np.fft.ifft2(np.sum(spectrum * model['filter'], axis=0), norm='ortho').real
            point = cv2.minMaxLoc(response)[3]
            psr = self._response_psr(response, point, np.ones(response.shape, bool))
            self.dbg['online_update_psr'] = psr
            if psr < TP.online_psr or math.hypot(point[0] - (n - 1) / 2, point[1] - (n - 1) / 2) > 3:
                return
            rate = min(alpha, .05) * min(1.0, psr / (2 * TP.online_psr))
            model['spectrum'] = (1 - rate) * model['spectrum'] + rate * spectrum
        spectrum = model['spectrum']
        power = np.sum(np.abs(spectrum) ** 2, axis=0)
        filt, dual, penalty = model['filter'].copy(), np.zeros_like(spectrum), 1.0
        # 大图块提供真实背景负样本，空间约束使滤波器集中在环带及少量邻域。
        for _ in range(4):
            prior = filt - dual
            auxiliary = prior + np.conj(spectrum) * (model['target'] - np.sum(spectrum * prior, axis=0)) / (power + penalty)
            spatial = np.fft.ifft2(auxiliary + dual, norm='ortho').real
            spatial *= penalty * model['mask'] ** 2 / (.05 + penalty * model['mask'] ** 2)
            filt = np.fft.fft2(spatial, norm='ortho')
            dual += auxiliary - filt
            penalty *= 5
        offsets = np.arange(-14, 15)
        model['kernel'] = (spatial[:, (-offsets[:, None]) % n, (-offsets[None, :]) % n] / n).astype(np.float32)
        model['filter'] = filt
        response = np.fft.ifft2(np.sum(spectrum * filt, axis=0), norm='ortho').real
        model['peak'] = float(np.max(response))
        model['frame'] = self.n_frames
        model['updates'] += 1
        self.online = model
        self.dbg['online_updates'] = model['updates']

    def _online_peaks(self, fr, cx, cy, r, half):
        model, TP = self.online, self.P.track
        if not TP.online_learning or model is None or r < TP.online_min_radius:
            return []
        H, W = fr.shape
        peaks = []
        kernels = model['kernel']
        if abs(model['angle']) > .5:
            n = kernels.shape[1]
            A = cv2.getRotationMatrix2D(((n - 1) / 2, (n - 1) / 2), -model['angle'], 1)
            kernels = np.stack([cv2.warpAffine(k, A, (n, n)) for k in kernels])
        scale = 8 / r
        extent = min(half, 192 / scale)
        x0, y0 = max(0, int(cx - extent)), max(0, int(cy - extent))
        x1, y1 = min(W, int(cx + extent) + 1), min(H, int(cy + extent) + 1)
        if x1 <= x0 or y1 <= y0 or min((x1 - x0) * scale, (y1 - y0) * scale) <= 30:
            return []
        images = [cv2.resize(fr.u8(ch)[y0:y1, x0:x1], None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_AREA) for ch in ('gray', 'white' if fr._bgr is not None else 'gray')]
        features = self._online_features(*images)
        response = sum(cv2.filter2D(f, cv2.CV_32F, k) for f, k in zip(features, kernels))
        valid = np.zeros(response.shape, bool)
        valid[14:-14, 14:-14] = True
        search = np.where(valid, response, -np.inf)
        for _ in range(2):
            _, value, _, point = cv2.minMaxLoc(search)
            psr = self._response_psr(response, point, valid)
            if value < max(.03, .2 * model['peak']) or psr < TP.online_psr:
                break
            x, y = point
            peaks.append((min(1.0, value / max(model['peak'], 1e-6)),
                          x0 + (x + .5) / scale - .5, y0 + (y + .5) / scale - .5, r))
            self.dbg['online_psr'] = max(self.dbg.get('online_psr', 0), psr)
            cv2.circle(search, point, 8, -np.inf, -1)
        return peaks

    def _reference_score(self, fr, m, references=None):
        references = self.references if references is None else references
        if not references:
            return 1.0
        key = (m.cx, m.cy, m.r, tuple(id(ref) for ref in references))
        if key in self._identity_cache:
            return self._identity_cache[key]
        H, W = fr.shape
        gray = fr.u8('gray')
        scores = []
        patches = {}
        for ref in references:
            best = -1.0
            n = ref['image'].shape[0]
            angle = ref['angle'] if abs(ref['angle']) > 0.5 else 0.0
            ratio_scale = m.r / max(ref['r'], 1e-6)
            tested = set()
            for ratio in (1.0, 0.9, 1.1):
                # 当前尺度下参考模板对应的物理范围（px）。封顶会把大环截断，
                # 所以取块后缩到模板尺度再比（大环时取块仍是主要成本，但结果正确）。
                size = max(9, min(1024, int(round(n * ratio_scale * ratio))))
                padding = max(2, int(math.ceil(0.08 * m.r)))
                count = size + 2 * padding
                half = (count - 1) / 2
                if min(m.cx - half, m.cy - half, W - 1 - m.cx - half, H - 1 - m.cy - half) < 0:
                    continue
                if (count, size) in tested:
                    continue
                tested.add((count, size))
                margin = max(1, int(round(padding * n / size)))
                out_size = count if size == n else n + 2 * margin
                key_patch = (count, out_size, angle)
                patch = patches.get(key_patch)
                if patch is None:
                    patch = cv2.getRectSubPix(gray, (count, count), (m.cx, m.cy))
                    if size != n:
                        patch = cv2.resize(patch, (out_size, out_size),
                                           interpolation=cv2.INTER_AREA if size > n else cv2.INTER_LINEAR)
                    if angle:
                        A = cv2.getRotationMatrix2D(((out_size - 1) / 2, (out_size - 1) / 2), angle, 1)
                        patch = cv2.warpAffine(patch, A, (out_size, out_size), borderMode=cv2.BORDER_REFLECT_101)
                    patches[key_patch] = patch
                best = max(best, cv2.minMaxLoc(cv2.matchTemplate(patch, ref['image'], cv2.TM_CCOEFF_NORMED))[1])
            scores.append(best)
        score = max(scores)
        self._identity_cache[key] = score
        return score

    def _match_peaks(self, scores, radius, gpu=False):
        if gpu:
            functions = ('sum', 'sqrSum', 'minMaxLoc', 'inRange', 'bitwise_not', 'bitwise_and', 'countNonZero')
            if (not self._gpu_reduce_ok or self.P.track.template_score <= 0
                    or not all(hasattr(cv2.cuda, name) for name in functions)):
                return None
            limit = float(np.finfo(np.float32).max)
            valid = cv2.cuda.inRange(scores, (-limit,), (limit,))
            scores.setTo(0, cv2.cuda.bitwise_not(valid))
            w, h = scores.size()
            mean = float(cv2.cuda.sum(scores)[0]) / (h * w)
            variance = float(cv2.cuda.sqrSum(scores)[0]) / (h * w) - mean * mean
            deviation = math.sqrt(max(variance, 0.0))
            # 高均值低方差的响应图保留 CPU 统计，避免平方差消减。
            if (not math.isfinite(mean + deviation) or deviation < 1e-5
                    or variance <= 1e-4 * mean * mean):
                return None
        else:
            mean, deviation = cv2.meanStdDev(scores)
            mean, deviation = float(mean[0, 0]), float(deviation[0, 0])
        peaks, mask = [], None
        for _ in range(2):
            if gpu:
                _, score, _, point = cv2.cuda.minMaxLoc(scores, mask=mask) if mask is not None else cv2.cuda.minMaxLoc(scores)
            else:
                _, score, _, point = cv2.minMaxLoc(scores)
            if score < self.P.track.template_score:
                break
            if gpu:
                equal = cv2.cuda.inRange(scores, (score,), (score,))
                if mask is not None:
                    equal = cv2.cuda.bitwise_and(equal, mask)
                # 同分峰仍由原响应图决定行优先顺序。
                if cv2.cuda.countNonZero(equal) != 1 or abs((score - mean) / deviation - self.P.track.appearance_psr) < 1e-3:
                    return None
            peaks.append((score, point))
            if gpu and len(peaks) == 1:
                if self._gpu_peak_mask is None:
                    self._gpu_peak_mask = cv2.cuda_GpuMat()
                mask = self._gpu_peak_mask
                mask.create(h, w, cv2.CV_8UC1)
                mask.setTo(255)
                disk = self._gpu_disks.get(radius)
                if disk is None:
                    circle = np.zeros((2 * radius + 1, 2 * radius + 1), np.uint8)
                    cv2.circle(circle, (radius, radius), radius, 255, -1)
                    disk = cv2.cuda_GpuMat()
                    disk.upload(circle)
                    while len(self._gpu_disks) >= 16:
                        self._gpu_disks.popitem(last=False)
                    self._gpu_disks[radius] = disk
                x, y = point
                xa, ya, xb, yb = max(0, x - radius), max(0, y - radius), min(w, x + radius + 1), min(h, y + radius + 1)
                dx, dy = xa - x + radius, ya - y + radius
                mask.rowRange(ya, yb).colRange(xa, xb).setTo(0,
                    disk.rowRange(dy, dy + yb - ya).colRange(dx, dx + xb - xa))
            elif not gpu:
                cv2.circle(scores, point, radius, -1, -1)
        return mean, deviation, peaks

    def _match_templates(self, roi, templates, peak_radius=None):
        """共享图像频谱和积分图，计算各模板的归一化互相关。"""
        if not templates:
            return []
        h, w = roi.shape
        TP = self.P.track
        backend = TP.template_backend
        self.template_backend = 'cpu'
        self.template_peak_backend = 'cpu'
        if (backend != 'cpu' and self._gpu_matcher is not False and roi.dtype == np.uint8
                and min(h, w) >= 128 and (backend == 'cuda' or (h * w >= TP.template_cuda_min_area
                    and (self.motion.backend == 'cuda' or cv2.cuda.getCudaEnabledDeviceCount() > 0)))):
            try:
                if self._gpu_matcher is None:
                    factory = getattr(cv2.cuda, 'createTemplateMatching', None)
                    if factory is None or cv2.cuda.getCudaEnabledDeviceCount() < 1:
                        raise RuntimeError('OpenCV 未提供可用的 CUDA 模板匹配')
                    self._gpu_matcher = factory(cv2.CV_8UC1, cv2.TM_CCOEFF_NORMED)
                    self._gpu_match_roi, self._gpu_match_template = cv2.cuda_GpuMat(), cv2.cuda_GpuMat()
                    self._gpu_match_response = cv2.cuda_GpuMat()
                self._gpu_match_roi.upload(np.ascontiguousarray(roi))
                maps, compact_count = [], 0
                for template in templates:
                    if np.ptp(template) == 0:
                        scores = np.ones((h - template.shape[0] + 1, w - template.shape[1] + 1), np.float32)
                        maps.append(scores if peak_radius is None else self._match_peaks(scores, peak_radius))
                        continue
                    key = (template.shape, template.dtype.str, template.tobytes())
                    cached = self._gpu_templates.get(key)
                    if cached is None:
                        capacity = min(16, max(0, self.P.search.fft_cache_mb / 2)) * 1024 ** 2
                        if template.nbytes <= capacity:
                            while self._gpu_templates and (len(self._gpu_templates) >= 32
                                    or self._gpu_template_bytes + template.nbytes > capacity):
                                _, (old_size, _) = self._gpu_templates.popitem(last=False)
                                self._gpu_template_bytes -= old_size
                            device_template = cv2.cuda_GpuMat()
                            device_template.upload(np.ascontiguousarray(template))
                            self._gpu_templates[key] = (template.nbytes, device_template)
                            self._gpu_template_bytes += template.nbytes
                        else:
                            device_template = self._gpu_match_template
                            device_template.upload(np.ascontiguousarray(template))
                    else:
                        _, device_template = cached
                        self._gpu_templates.move_to_end(key)
                    device_scores = self._gpu_matcher.match(self._gpu_match_roi, device_template, self._gpu_match_response)
                    response_size = (h - template.shape[0] + 1) * (w - template.shape[1] + 1)
                    bucket = int(response_size > 32768) + int(response_size > 131072)
                    if (peak_radius is not None and self.P.track.compact_gpu_matching
                            and self._gpu_peak_routes.get(bucket, True)):
                        try:
                            summary = self._match_peaks(device_scores, peak_radius, gpu=True)
                        except (cv2.error, AttributeError, TypeError, RuntimeError) as error:
                            self._gpu_reduce_ok = False
                            self.template_peak_error = str(error)
                            summary = None
                        if summary is not None:
                            maps.append(summary)
                            compact_count += 1
                            continue
                    scores = device_scores.download()
                    np.nan_to_num(scores, copy=False, nan=0, posinf=0, neginf=0)
                    maps.append(scores if peak_radius is None else self._match_peaks(scores, peak_radius))
                self.template_backend = 'cuda'
                self.template_peak_backend = ('cuda' if compact_count == len(templates)
                                              else 'mixed' if compact_count else 'cpu')
                return maps
            except (cv2.error, AttributeError, RuntimeError) as e:
                self.template_backend_error = str(e)
                self._gpu_matcher = False
                if backend == 'cuda':
                    raise RuntimeError(f'CUDA 模板匹配失败：{e}') from e
        return self._match_templates_cpu(roi, templates, peak_radius, cache=True)

    def _match_templates_cpu(self, roi, templates, peak_radius, cache=False):
        if not templates:
            return []
        h, w = roi.shape
        fh = cv2.getOptimalDFTSize(((h + 63) // 64) * 64)
        fw = cv2.getOptimalDFTSize(((w + 63) // 64) * 64)
        buf = np.zeros((fh, fw), np.float32)
        buf[:h, :w] = roi
        spectrum = cv2.dft(buf, flags=cv2.DFT_COMPLEX_OUTPUT)
        sums, squares = cv2.integral2(roi, sdepth=cv2.CV_64F, sqdepth=cv2.CV_64F)
        denominators = {}
        maps = []
        for template in templates:
            th, tw = template.shape
            key = (fh, fw, template.shape, template.dtype.str, template.tobytes()) if cache else None
            cached = self._match_cache.get(key) if cache else None
            if cached is None:
                centered = template.astype(np.float32) - float(np.mean(template))
                norm = float(np.linalg.norm(centered))
                buf.fill(0)
                buf[:th, :tw] = centered
                kernel = cv2.dft(buf, flags=cv2.DFT_COMPLEX_OUTPUT)
                if cache:
                    capacity = min(16, max(0, self.P.search.fft_cache_mb / 2)) * 1024 ** 2
                    while self._match_cache and self._match_cache_bytes + kernel.nbytes > capacity:
                        _, (_, old) = self._match_cache.popitem(last=False)
                        self._match_cache_bytes -= old.nbytes
                    if kernel.nbytes <= capacity:
                        self._match_cache[key] = (norm, kernel)
                        self._match_cache_bytes += kernel.nbytes
            else:
                norm, kernel = cached
                self._match_cache.move_to_end(key)
            shape = (th, tw)
            if shape not in denominators:
                count = th * tw
                total = sums[th:, tw:] - sums[:-th, tw:] - sums[th:, :-tw] + sums[:-th, :-tw]
                sq = squares[th:, tw:] - squares[:-th, tw:] - squares[th:, :-tw] + squares[:-th, :-tw]
                denominators[shape] = np.sqrt(np.maximum(sq - total * total / count, 0))
            if norm < 1e-6:
                maps.append(np.ones((h - th + 1, w - tw + 1), np.float32))
                continue
            cross = cv2.idft(cv2.mulSpectrums(spectrum, kernel, 0, conjB=True),
                            flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)[:h - th + 1, :w - tw + 1]
            den = denominators[shape] * norm
            scores = np.zeros_like(cross)
            np.divide(cross, den, out=scores, where=den > 1e-6)
            np.clip(scores, -1, 1, out=scores)
            maps.append(scores)
        return maps if peak_radius is None else [self._match_peaks(scores, peak_radius) for scores in maps]

    def _recovery_identity(self, fr, m):
        if not self.references:
            m.identity_score = 1.0
            return True
        score = self._reference_score(fr, m)
        m.identity_score = score
        self.dbg['identity_score'] = score
        TP, V = self.P.track, self.P.verify
        searching = self.phase in (REACQUIRE, SEARCH)
        # 半径不再参与身份判定：尺度由光流传播、模板按当前半径重采样，
        # 只要外观相关度过门槛、位置来自同一块地面，就认为是同一个环。
        # 重捕分级：预测点附近的环身份门槛放低（位置连续性本身就是强证据），远处仍从严
        near = (self.tgt is not None
                and math.hypot(m.cx - self.tgt[0], m.cy - self.tgt[1]) <= max(4.0, 0.35 * m.r))
        thr = TP.identity_score
        if searching:
            thr = max(TP.template_score, 0.65 if near else 0.72)
        if score >= thr:
            return True
        if self.learned is not None:
            tolerance = max(0.06, 3 * math.sqrt(self.learned['width_var']))
            if abs(m.line_w / m.r - self.learned['width']) > tolerance:
                return False
        return (score >= TP.template_score - 0.15 and m.sectors >= V.lock_sectors
                and m.inlier_ratio >= V.lock_min_inlier_ratio and m.max_gap <= V.lock_max_gap
                and m.rms <= _delta(V.lock_max_rms, m.r))

    def _confirm_recovery(self, m, stamp, local=False):
        TP = self.P.track
        needed = TP.reacq_confirm_hits if m.identity_score >= TP.identity_score else max(TP.reacq_confirm_hits, TP.reacq_weak_hits)
        if local:
            needed = min(needed, TP.local_reacq_hits)
        c = self.recovery_candidate
        if c is not None and c.frame == self.n_frames:
            return len(c.hits) >= c.needed
        if c is None or c.frame != self.n_frames - 1 or not self._same(c, m.cx, m.cy, m.r):
            c = _Cand(m, stamp, max(TP.reacq_confirm_hits, TP.reacq_weak_hits))
            c.needed = needed
            self.recovery_candidate = c
        else:
            c.hits.append(True)
            c.cx, c.cy, c.r = m.cx, m.cy, m.r
            c.meas = m
            c.needed = max(c.needed, needed)
        c.frame = self.n_frames
        self.dbg['reacq_hits'] = len(c.hits)
        return len(c.hits) >= c.needed

    def _accept_recovery(self, m, stamp):
        self.dbg['target_identity_score'] = m.identity_score
        self.phase = TRACK
        self.stable_track_hits = 0
        self.miss_streak = 0
        self.coast_motion_lost = False
        self.tgt = [m.cx, m.cy, m.r]
        self.tgt_meas = m
        self.measured_radius = m.r
        self.last_detect = stamp
        self.out = DETECTED
        self.reacq = None
        self.cands = []
        self.pending = []
        self.recovery_candidate = None
        self._template_scale_index = 0
        self.template_hold = self.P.track.template_update_hits
        self.identity_hits = 0
        self.recent_identity_hits = 0

    def _appearance_score(self, fr, m):
        if self.template is None:
            return 1.0
        size = max(9, int(round(self.template['size'] * m.r / self.template['r'])))
        H, W = fr.shape
        half = (size - 1) / 2
        if min(m.cx - half, m.cy - half, W - 1 - m.cx - half, H - 1 - m.cy - half) < 0:
            return -1.0
        patch = cv2.getRectSubPix(fr.u8('gray'), (size, size), (m.cx, m.cy))
        n = self.template['image'].shape[0]
        patch = cv2.resize(patch, (n, n), interpolation=cv2.INTER_AREA)
        if abs(self.template_angle) > 0.5:
            A = cv2.getRotationMatrix2D(((n - 1) / 2, (n - 1) / 2), self.template_angle, 1)
            patch = cv2.warpAffine(patch, A, (n, n), borderMode=cv2.BORDER_REFLECT_101)
        return float(cv2.matchTemplate(patch, self.template['image'], cv2.TM_CCOEFF_NORMED)[0, 0])

    def _recover_shift(self, fr, m, cx, cy, r):
        radius_error = m.reason.startswith('radius ')
        scale_ok = (self.reference_radius is not None
                    and abs(m.r - self.reference_radius) <= 0.2 * self.reference_radius
                    and m.sectors >= self.P.verify.lock_sectors
                    and m.inlier_ratio >= self.P.verify.lock_min_inlier_ratio)
        if (self.template is not None and (m.reason.startswith('shift ') or (radius_error and scale_ok))
                and m.sectors >= 12 and m.inlier_ratio >= 0.75
                and math.hypot(m.cx - cx, m.cy - cy) <= max(8.0, 0.75 * r)
                and self._appearance_score(fr, m) >= self.P.track.template_score
                and (not radius_error or self._reference_score(fr, m) >= self.P.track.identity_score)):
            m.ok, m.reason = True, ''

    def _template_measure(self, fr, stamp):
        if self._template_frame == self.n_frames:
            return self._template_result
        self._template_frame = self.n_frames
        self._template_result = None
        if self.template is None or self.tgt is None:
            return None
        cx, cy, r = self.tgt
        TP, V = self.P.track, self.P.verify
        c = self.recovery_candidate
        confirmed_radius = None
        if (c is not None and c.frame == self.n_frames - 1
                and (self.coast_motion_lost or self.phase == REACQUIRE)
                and c.meas.identity_score >= 0.9 and c.meas.sectors >= V.lock_sectors
                and c.meas.inlier_ratio >= V.lock_min_inlier_ratio):
            cx, cy, r = c.cx, c.cy, c.r
            confirmed_radius = c.r
        # 弱测量可能缩小半径，重捕保留随地面运动传播的可靠尺度。
        r = max(r, self.reference_radius or r)
        async_tpl = self.P.search.async_search or self.P.search.async_template
        if async_tpl:
            finished = self._template_seeds is not None
            seeds = [(f.score, f.x, f.y, f.r) for f in self._template_seeds or ()]
            self._template_seeds = None
            if c is not None and c.frame == self.n_frames - 1 and c.meas.source == 'template':
                seeds.append((1., c.cx, c.cy, c.r))
            if seeds:
                m = self._recovery_from_peaks(fr, seeds, r)
                if m is not None:
                    m.source = 'template'
                    self._template_result = m
                    return m
                if self.ambiguous:
                    return None
            if finished or (self._search_meta is not None and self._search_meta.get('kind') == 'template'):
                return None
        age = max(0.0, stamp - self.last_detect)
        base = max(TP.reacq_box_min / 2, TP.reacq_box_r * r / 2)
        uncertainty = min(base, 0.5 * self.dbg.get('motion_step_px', 0) + 3 * self.dbg.get('motion_error', 0))
        if not self.dbg.get('motion_ok'):
            uncertainty = max(uncertainty, 0.5 * base)
        half = base * min(3.0, 1 + age) + uncertainty
        if (0 < age <= TP.recovery_local_s and self.dbg.get('motion_ok')
                and not self.coast_motion_lost and self.dbg.get('motion_error', 0) <= max(1.0, .25 * r)):
            half = min(half, max(48.0, 4 * r) + uncertainty)
        H, W = fr.shape
        x0, y0 = max(0, int(cx - half)), max(0, int(cy - half))
        x1, y1 = min(W, int(cx + half) + 1), min(H, int(cy + half) + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        roi = fr.u8('gray')[y0:y1, x0:x1]
        if async_tpl and self._search_future is not None:
            return None          # 后台任务槽被占用，这一帧不排队
        online_peaks = self._online_peaks(fr, cx, cy, r, half)
        online_candidate = None
        online_checked = bool(online_peaks and self.dbg.get('online_psr', 0) >= 2 * TP.online_psr)
        if online_checked:
            online_candidate = self._recovery_from_peaks(fr, online_peaks, r, strict=True)
            if self.ambiguous:
                return None
            if online_candidate is not None and online_candidate.identity_score >= .9:
                self.dbg['recovery_source'] = 'online'
                self._template_result = online_candidate
                return online_candidate
        memories = [(self.template, self.template_angle)]
        if self.keyframe is not None:
            memories.insert(0, (self.keyframe, self.keyframe['angle']))
        if self.references:
            memories.append((self.references[-1], self.references[-1]['angle']))
            if len(self.references) > 1:
                memories.append((self.references[0], self.references[0]['angle']))
        unique, seen = [], set()
        for memory, angle in memories:
            if id(memory['image']) not in seen:
                unique.append((memory, angle))
                seen.add(id(memory['image']))
        memories = unique
        size = self.template['size'] * r / self.template['r']
        raw_area = roi.size              # 缩放前的搜索面积，用来判断这次匹配是否值得放后台
        scale = min(1.0, 96 / max(size * 1.1, 1), 512 / max(roi.shape))
        if scale < 1:
            roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        peaks, patches, sizes, appearance = [], [], [], []
        patch_indexes = {}
        ratios = (1.0, 0.9, 1.1)
        if TP.stagger_recovery_scales:
            # 每次保留所有外观参考，只轮换尺度。复核中的候选优先用实测半径。
            ratios = (confirmed_radius / r if confirmed_radius is not None else ratios[self._template_scale_index],)
        for memory, angle in memories:
            template = memory['image']
            if abs(angle) > 0.5:
                n = template.shape[0]
                # 搜索模板按整度缓存；最终身份分和圆心测量仍使用当前原图。
                A = cv2.getRotationMatrix2D(((n - 1) / 2, (n - 1) / 2), -round(angle), 1)
                template = cv2.warpAffine(template, A, (n, n), borderMode=cv2.BORDER_REFLECT_101)
            size = memory['size'] * r / memory['r']
            for ratio in ratios:
                n = max(9, int(round(size * ratio * scale)))
                if n > min(roi.shape):
                    continue
                patch = cv2.resize(template, (n, n), interpolation=cv2.INTER_LINEAR)
                key = (n, patch.tobytes())
                index = patch_indexes.get(key)
                if index is None:
                    index = len(patches)
                    patch_indexes[key] = index
                    patches.append(patch)
                sizes.append((index, n, ratio, memory is self.keyframe))
        peak_radius = max(3, int(round(r * scale)))
        # 小任务（测试图/320p）直接在主线程做，省线程开销；大搜索块（1080p 重捕）放后台，
        # 主循环当帧先输出光流预测，下一帧再收结果。
        background = (async_tpl and TP.template_backend != 'cuda'
                      and raw_area * max(1, len(patches)) >= 1_000_000)
        if not background and async_tpl and TP.template_backend != 'cuda':
            # 已经在主线程上量到过很贵的匹配（慢机器上 240x240 x 4 模板要 20~40 ms）：
            # 之后一律放后台，主循环当帧先输出光流预测，下一帧再收结果。
            background = self._tpl_cost_ms > max(20.0, 0.8 * TP.frame_budget_ms)
        if not background and self.phase in (TRACK, COAST) and len(patches) > len(memories):
            # 跟丢当帧在主线程做：每个参考只保留 1.0 尺度。
            # 一次 240x240 窗口 x 12 个尺度实测 18~30 ms，正好卡在丢锁那一帧；
            # 多尺度完整找回留到 REACQUIRE。
            keep = sorted({index for index, _, ratio, _ in sizes if ratio == 1.0})
            remap = {old: i for i, old in enumerate(keep)}
            patches = [patches[i] for i in keep]
            sizes = [(remap[index], n, ratio, clear) for index, n, ratio, clear in sizes if index in remap]
        if background:
            A = np.array([[1., 0., x0], [0., 1., y0], [0., 0., 1.]])
            self._submit_search('template', stamp, A, self._match_template_seeds,
                                roi.copy(), patches, sizes, peak_radius, scale, r)
            if TP.stagger_recovery_scales:
                self._template_scale_index = (self._template_scale_index + 1) % 3
            return None
        t_tpl = time.perf_counter()
        summaries = self._match_templates(roi, patches, peak_radius=peak_radius)
        self._tpl_cost_ms = 0.6 * self._tpl_cost_ms + 0.4 * (time.perf_counter() - t_tpl) * 1000
        for index, n, ratio, clear in sizes:
            mean, deviation, matches = summaries[index]
            for score, point in matches:
                x = x0 + (point[0] + n / 2) / scale - 0.5
                y = y0 + (point[1] + n / 2) / scale - 0.5
                peaks.append((score, x, y, r * ratio))
                if clear:
                    psr = (score - mean) / max(deviation, 1e-6)
                    appearance.append((score, x, y, psr))
        self._template_result = self._recovery_from_peaks(fr, peaks, r)
        if self._template_result is None and not self.ambiguous and online_peaks:
            if not online_checked:
                online_candidate = self._recovery_from_peaks(fr, online_peaks, r, strict=True)
            if not self.ambiguous:
                self._template_result = online_candidate
                if online_candidate is not None:
                    self.dbg['recovery_source'] = 'online'
        if TP.stagger_recovery_scales and self._template_result is None and not self.ambiguous:
            self._template_scale_index = (self._template_scale_index + 1) % 3
        if (self._template_result is None and not self.ambiguous and self.keyframe is not None
                and age <= TP.appearance_coast_s):
            ranked = []
            for peak in sorted(appearance, reverse=True):
                if not any(math.hypot(peak[1] - p[1], peak[2] - p[2]) <= max(4, r) for p in ranked):
                    ranked.append(peak)
            if (ranked and ranked[0][0] >= TP.appearance_score
                    and (len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= TP.identity_margin)):
                score, x, y, psr = ranked[0]
                motion_agrees = (not self.dbg.get('motion_ok') or self.coast_motion_lost
                                 or math.dist((x, y), self.tgt[:2]) <= max(3.0, 0.25 * r,
                                                                          2 * self.dbg.get('motion_error', 0)))
                if psr >= TP.appearance_psr and motion_agrees:
                    hypothesis = Measurement(False, 'appearance', cx=x, cy=y, r=r)
                    ref = self.keyframe
                    side = max(9, int(round(ref['size'] * r / ref['r'])))
                    n = ref['image'].shape[0]
                    patch = cv2.getRectSubPix(fr.u8('gray'), (side, side), (x, y))
                    patch = cv2.resize(patch, (n, n), interpolation=cv2.INTER_AREA)
                    if abs(ref['angle']) > .5:
                        A = cv2.getRotationMatrix2D(((n - 1) / 2, (n - 1) / 2), ref['angle'], 1)
                        patch = cv2.warpAffine(patch, A, (n, n), borderMode=cv2.BORDER_REFLECT_101)
                    # 环带核心单独复核，避免周边纹理主导定位。
                    yy, xx = np.indices((n, n))
                    distance = np.hypot(xx - (n - 1) / 2, yy - (n - 1) / 2)
                    radius = ref['r'] * n / ref['size']
                    weight = np.where(distance <= 1.5 * radius,
                                      .2 + np.exp(-((distance - radius) / max(1, .3 * radius)) ** 2), 0)
                    first = ref['image'].astype(np.float32)
                    first -= np.sum(first * weight) / np.sum(weight)
                    second = patch.astype(np.float32)
                    second -= np.sum(second * weight) / np.sum(weight)
                    den = math.sqrt(float(np.sum(first * first * weight) * np.sum(second * second * weight)))
                    ring_score = float(np.sum(first * second * weight)) / max(den, 1e-6)
                    if ring_score >= 0.5 and self._reference_score(fr, hypothesis) >= TP.template_score:
                        self._appearance_result = hypothesis
                        self.dbg.update(appearance_ncc=score, appearance_psr=psr, appearance_ring=ring_score)
        return self._template_result

    def _match_template_seeds(self, roi, patches, sizes, peak_radius, scale, r):
        summaries = self._match_templates_cpu(roi, patches, peak_radius)
        return [Candidate((x + n / 2) / scale - .5, (y + n / 2) / scale - .5, r * ratio, score, 0)
                for index, n, ratio, _ in sizes for score, (x, y) in summaries[index][2]]

    def _recovery_from_peaks(self, fr, peaks, r, strict=False):
        TP, V = self.P.track, self.P.verify
        checked, valid = [], []
        min_sectors = V.lock_sectors if strict else max(V.track_sectors, 12 if self.phase == REACQUIRE else 10)
        for score, x, y, radius in sorted(peaks, reverse=True):
            # 模板响应图的峰可能十几个；大环时每个复核都要几毫秒，
            # 只复核最显著的几个，剩下的留给下一帧，避免一帧里堆出几十毫秒。
            # 这里不用帧预算截断：慢机器上一帧只能复核一个时会漏掉"两个长得一样的候选"，
            # 反而可能锁到诱饵上；3 个的上限已经够便宜。
            if len(checked) >= 3:
                break
            if any(math.hypot(x - px, y - py) <= max(3, r) for px, py in checked):
                continue
            checked.append((x, y))
            m = measure_channels(fr, self.cam, x, y, radius, V, 'track', local=True,
                                 quick=len(checked) > 1)
            self.dbg['recover_meas'] = m
            if strict and (m.inlier_ratio < V.lock_min_inlier_ratio or m.max_gap > V.lock_max_gap
                           or m.rms > _delta(V.lock_max_rms, m.r)):
                continue
            if (m.reason.startswith('shift ') and m.sectors >= min_sectors and m.inlier_ratio >= 0.75
                    and math.hypot(m.cx - x, m.cy - y) <= max(8.0, 0.75 * r)):
                m.ok, m.reason = True, ''
            if (m.ok and m.sectors >= min_sectors and abs(m.r - r) <= 0.3 * r
                    and self._recovery_identity(fr, m)
                    and not any(self._same(old, m.cx, m.cy, m.r) for _, old in valid)):
                valid.append((score, m))
        valid.sort(key=lambda item: (item[1].identity_score, item[0]), reverse=True)
        if not valid:
            return None
        if len(valid) > 1 and valid[0][1].identity_score - valid[1][1].identity_score < TP.identity_margin:
            self.ambiguous = True
            return None
        self.dbg['template_score'] = valid[0][0]
        return valid[0][1]

    def _local_motion_reliable(self):
        return self.dbg['local_motion'] and self.dbg['motion_error'] <= 1.5

    def _begin_reacquire(self, stamp, flow_only=False):
        cx, cy, r = self.tgt
        self.phase = REACQUIRE
        self.coast_motion_lost = False
        self.stable_track_hits = 0
        self.reacq = dict(start=stamp, x=cx, y=cy, r=r, flow_only=flow_only)
        self.cands = []
        self.cursor = 0
        self.ed_cursor = 0
        self.pending = []
        self.identity_hits = 0
        self.template_hold = self.P.track.template_update_hits

    def _step_track(self, fr, T, stamp):
        V, TP = self.P.verify, self.P.track
        self._stamp = stamp
        cx, cy, r = self.tgt
        since = stamp - self.last_detect
        stable_grace = (self.stable_track_hits >= TP.stable_track_frames
                        and 0 < since <= TP.stable_no_flow_s)
        reliable_local = (TP.local_reacq_hits > 0 and self.references and T is not None
                          and self._local_motion_reliable() and not self.coast_motion_lost)
        H, W = fr.shape
        in_view = min(cx, cy, W - 1 - cx, H - 1 - cy) >= 0.5 * r
        recent_identity = (not self.template_hold and self._identity_score >= TP.identity_score
                           and self.n_frames - self._identity_frame <= max(4, TP.identity_every + 1))
        stable_local = (TP.local_reacq_hits > 0 and T is not None and self.references
                        and not self.coast_motion_lost and in_view
                        and (self.stable_track_hits >= TP.stable_track_frames or recent_identity)
                        and self.dbg['motion_error'] <= max(2.5, min(4.0, 0.2 * r)))
        local_grace = (stable_local and 0 < since <= min(TP.stable_template_grace_s, TP.coast_s)
                       and self.recovery_candidate is None)
        local_coast = local_grace and self.phase == COAST
        if T is None:
            delta = _delta(V.delta_search, r)
        elif self.phase == COAST:
            delta = min(_delta(V.delta_search, r), _delta(V.delta_track, r) * (1 + since / TP.coast_s))
        else:
            delta = None
        quick = (TP.fast_measure_every > 0 and (self.phase == TRACK or local_coast)
                 and self.n_frames % TP.fast_measure_every != 0
                 and self._measure_width_ratio is not None)
        widths = (max(1.0, self._measure_width_ratio * r),) if quick else None
        if quick:
            m = measure_channels(fr, self.cam, cx, cy, r, V, 'track', delta, local=True,
                                 circle_first=reliable_local and self.tgt_meas is not None and self.tgt_meas.ellipse is None,
                                 quick=True, widths=widths, channels=(self._measure_channel or 'gray',))
        else:
            m = measure_channels(fr, self.cam, cx, cy, r, V, 'track', delta, local=True,
                                 circle_first=reliable_local and self.tgt_meas is not None and self.tgt_meas.ellipse is None)
        if quick and not m.ok and not local_coast:
            m = measure_channels(fr, self.cam, cx, cy, r, V, 'track', delta, local=True,
                                 circle_first=reliable_local and self.tgt_meas is not None and self.tgt_meas.ellipse is None)
        elif m.ok and m.r > 1:
            self._measure_width_ratio = m.line_w / m.r
            self._measure_channel = m.channel
        self._recover_shift(fr, m, cx, cy, r)
        if not m.ok and local_grace:
            self.dbg['track_meas'] = m
            self.phase = COAST
            self.out = PREDICTED
            self.dbg['prediction_source'] = 'flow'
            return
        alternate = self.measured_radius
        if (not m.ok and alternate is not None and since <= TP.coast_s
                and 0.005 * r < abs(alternate - r) <= 0.1 * r):
            retry = measure_channels(fr, self.cam, cx, cy, alternate, V, 'track', delta, local=True)
            self._recover_shift(fr, retry, cx, cy, alternate)
            if retry.ok and self._reference_score(fr, retry) >= TP.identity_score:
                m = retry
                self.dbg['scale_retry'] = True
        if (not m.ok and not stable_grace and T is not None and not self.coast_motion_lost
                and 0 < r <= TP.shift_retry_r and time.perf_counter() < self._deadline):
            # 小环时预测差 2~4 px 整圆复核就会失败（扇区凑不满），而环本身没丢。
            # 在预测点周围补 4 个快复核，把光流残差吃掉；命中后会做身份校验。
            d = max(1.5, min(4.0, 0.3 * r))
            for dx, dy in ((d, 0.0), (-d, 0.0), (0.0, d), (0.0, -d)):
                if time.perf_counter() >= self._deadline:
                    break
                retry = measure_channels(fr, self.cam, cx + dx, cy + dy, r, V, 'track', delta,
                                         local=True, quick=True)
                if retry.ok:
                    m = retry
                    self.dbg['shift_retry'] = round(d, 1)
                    break
        self.dbg['track_meas'] = m
        if not m.ok and reliable_local and 0 < since <= min(TP.coast_s, TP.reacq_expand_s):
            self._begin_reacquire(self.last_detect, flow_only=True)
            return
        if (not m.ok and T is not None and not self.coast_motion_lost and self.recovery_candidate is None
                and TP.local_reacq_hits == 0
                and self.stable_track_hits >= TP.stable_track_frames
                and 0 < since <= min(TP.stable_template_grace_s, TP.coast_s)):
            self.phase = COAST
            self.out = PREDICTED
            return
        if not m.ok or self.coast_motion_lost:
            recovered = m if m.ok and self._recovery_identity(fr, m) else None
            if recovered is None:
                recovered = self._template_measure(fr, stamp)
            if recovered is not None:
                strong = (recovered.identity_score >= 0.9
                          and recovered.sectors >= V.lock_sectors
                          and recovered.inlier_ratio >= V.lock_min_inlier_ratio
                          and math.hypot(recovered.cx - cx, recovered.cy - cy) <= max(24, 3 * r))
                if (strong and recovered.source != 'template') or self._confirm_recovery(recovered, stamp):
                    self._accept_recovery(recovered, stamp)
                elif stable_grace and (T is None or self.coast_motion_lost):
                    self.phase = COAST
                    self.coast_motion_lost = True
                elif self.miss_streak < TP.track_miss_grace and since <= TP.coast_s \
                        and not self.coast_motion_lost:
                    # 偶发单帧失败（快速下降、短暂模糊）先预测，不立刻判丢。
                    self.miss_streak += 1
                    self.phase = COAST
                    self.out = PREDICTED
                    self.dbg['prediction_source'] = 'flow' if T is not None else 'hold'
                else:
                    self.phase = REACQUIRE
                    self.coast_motion_lost = False
                    self.stable_track_hits = 0
                    if self.reacq is None:
                        self.reacq = dict(start=stamp, x=cx, y=cy, r=r)
                    self.out = LOST
                return
            self.recovery_candidate = None
            if self._appearance_result is not None:
                estimate = self._appearance_result
                self.tgt = [estimate.cx, estimate.cy, estimate.r]
                self.phase = COAST
                self.out = PREDICTED
                self.coast_motion_lost |= T is None
                self.dbg['prediction_source'] = 'appearance'
                return
        if m.ok and not self.coast_motion_lost:
            self.phase = TRACK
            self.miss_streak = 0
            self.dbg['measurement_shift'] = math.hypot(m.cx - cx, m.cy - cy)
            strong = m.sectors >= V.lock_sectors and m.inlier_ratio >= V.lock_min_inlier_ratio
            radius = m.r if strong or self.reference_radius is None else self.reference_radius
            self.tgt = [m.cx, m.cy, radius]
            self.tgt_meas = m
            self.measured_radius = m.r
            self.last_detect = stamp
            self.out = DETECTED
            if m.sectors >= V.lock_sectors and m.inlier_ratio >= V.lock_min_inlier_ratio:
                # 几何测量校正尺度，外观参考仍受身份验证约束。
                if self.references:
                    self.reference_radius = m.r
                self.identity_hits += 1
                # 身份复核只限制参考更新，不打断已通过的连续测量。
                if self.template_hold and self._recovery_identity(fr, m):
                    self.template_hold -= 1
            self._remember(fr, m)
            if self.template_hold or self.dbg.get('target_identity_score', -1) < TP.identity_score:
                self.stable_track_hits = 0
            elif m.sectors >= V.lock_sectors and m.inlier_ratio >= V.lock_min_inlier_ratio:
                self.stable_track_hits = min(self.stable_track_hits + 1, TP.stable_track_frames + 2)
            else:
                self.stable_track_hits = max(0, self.stable_track_hits - 1)
        elif (T is None or self.coast_motion_lost) and stable_grace:
            self.phase = COAST
            self.coast_motion_lost = True
        elif T is None or self.coast_motion_lost or since > TP.coast_s:
            self._begin_reacquire(stamp)
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
                expected_r = max(q['r'], self.reference_radius or q['r'])
                # 下降中环会持续变大：重捕窗口按“可能长大 2~3 倍”放宽
                if el <= TP.reacq_expand_s:
                    r_lo, r_hi = 0.75 * q['r'], 1.5 * expected_r
                else:
                    r_lo, r_hi = 0.4 * q['r'], 3.0 * expected_r
                if el <= TP.reacq_expand_s:
                    h = max(TP.reacq_box_min / 2, TP.reacq_box_r * q['r'] / 2) * (1 + el)
                    region = (q['x'] - h, q['y'] - h, q['x'] + h, q['y'] + h)
                else:
                    region = (0, 0, W, H)
                return region, r_lo, r_hi, (q['x'], q['y']), expected_r
        S = self.P.search
        if self.height:
            r0 = self.cam.fx * (self.P.ring.diameter_m / 2) / self.height
            r_lo, r_hi = r0 * S.height_band[0], r0 * S.height_band[1]
        else:
            r_lo, r_hi = S.r_min, S.r_max
            if self.click is not None:
                r_lo = min(r_lo, S.click_r_min)
                # 没有高度时，半径上限随点击以来的地面放大倍数增大：高空看不清、降低后才出现的环也在范围内。
                # 下限不跟着放大，否则点击时小于 r_min 的环会一直追不上下限。
                r_hi = min(S.r_max * max(1.0, self.click['scale']), 0.5 * math.hypot(W, H))
                if self.reference_radius is not None:
                    # 点击时实测到过 >r_max 的大环（低空）：用这个尺度放宽上限，不盲目扫大半径。
                    r_hi = max(r_hi, min(3.0 * self.reference_radius, 0.5 * math.hypot(W, H)))
        if self.r_floor:
            # 最大可见半径约为画面对角线的一半，超过它的环已经看不全
            r_lo = max(r_lo, self.r_floor)
            r_hi = max(r_hi, min(1.6 * self.r_floor, 0.5 * math.hypot(W, H)))
            # 丢失后环可能继续变大：上限放到丢失时半径的 3 倍（= 6*r_floor），
            # 不依赖可能过期的 reference_radius。
            r_hi = max(r_hi, min(6.0 * self.r_floor, 0.5 * math.hypot(W, H)))
            if self.reference_radius is not None:
                # 丢失后环还在继续变大（下降），上限跟着参考半径放 3 倍
                r_hi = max(r_hi, min(3.0 * self.reference_radius, 0.5 * math.hypot(W, H)))
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

    def _probe_click(self, fr, x, y, r_lo, r_hi, stamp, coarse=False, r0=None, deadline=None):
        """点击点直接复核：操作员点的位置就是环，不等 ED/匹配滤波生成候选。

        半径未知，用 1.35（邻域重试用 1.6）等比阶梯扫一遍；先快筛，通过后再完整复核。
        点没命中说明点击只是"大概位置"，在后续几帧用 ±0.4r_hi 的小网格重试。
        r0：从阶梯的哪一级开始（上一帧预算不够时把剩下的阶梯排到下一帧）。
        """
        S, V, TP = self.P.search, self.P.verify, self.P.track
        H, W = fr.shape
        if not (0 <= x < W and 0 <= y < H):
            return
        best = None
        limit = self._deadline if deadline is None else deadline
        r = max(S.click_r_min, r_lo, 2.0) if r0 is None else max(r0, 2.0)
        grow = 1.6 if coarse else 1.35
        truncated = False
        while r <= r_hi:
            if time.perf_counter() >= limit:
                # 这一帧预算用完了，把剩下的半径阶梯排到下一帧；整条丢掉就会
                # 因为慢机器上第一次点击没扫到而永远锁不上。
                truncated = True
                break
            probe = measure_channels(fr, self.cam, x, y, r, V, 'track', local=True, quick=True)
            if probe.ok:
                m = measure_channels(fr, self.cam, x, y, r, V, 'lock', local=True)
                if m.ok and (best is None or m.inlier_ratio > best.inlier_ratio):
                    best = m
                    if m.inlier_ratio >= 0.9:
                        break           # 已经很干净，不再扫更大的半径，省一次性开销
            r *= grow
        if best is None:
            if truncated:
                self._click_probe.insert(0, (x, y, coarse, r))
            else:
                self._spread_probe(x, y, r_hi)
            return
        m = best
        if not self._recovery_identity(fr, m):
            self.dbg['click_probe'] = 'identity'
            return
        if any(self._same(c, m.cx, m.cy, m.r) for c in self.cands):
            return
        m.ed_ok = ed_support(fr, m, S, V)
        window = max(TP.confirm_window, TP.identity_init_hits + 2)
        c = _Cand(m, stamp, window)
        # 点击候选是操作员显式指定的位置，不必再等满 6 帧身份确认；
        # 3 帧足够过滤掉偶发抖动，锁定延迟从 6~7 帧降到 3 帧左右。
        c.from_click = True
        c.needed = max(TP.confirm_hits, TP.identity_init_hits)
        if not self.references:
            c.reference = self._identity_patch(fr, m)
            if c.reference is None:
                return
        c.frame = self.n_frames
        self.cands.append(c)
        if self.reference_radius is None and m.r > S.r_max:
            # 只有点击时实测到超过常规上限的大环才记下来，用来放宽搜索上限。
            self.reference_radius = m.r
        self.dbg['click_probe'] = (round(m.cx, 1), round(m.cy, 1), round(m.r, 1),
                                   round(m.inlier_ratio, 2), bool(m.ed_ok))

    def _spread_probe(self, x, y, r_hi):
        """点击只是"大概位置"：原地点没命中时，后续几帧在 ±0.4r_hi 的小网格上重试。

        每帧只做一个点（粗阶梯），命中即建候选，避免把一次点击变成一帧几十毫秒的负担。
        """
        if self._probe_spread:
            return
        self._probe_spread = True
        d = max(6.0, min(24.0, 0.4 * float(r_hi)))
        self._click_probe.extend([(x + d, y, True, None), (x - d, y, True, None),
                                  (x, y + d, True, None), (x, y - d, True, None),
                                  (x + 0.7 * d, y + 0.7 * d, True, None),
                                  (x - 0.7 * d, y - 0.7 * d, True, None),
                                  (x + 0.7 * d, y - 0.7 * d, True, None),
                                  (x - 0.7 * d, y + 0.7 * d, True, None)])

    def _step_search(self, fr, T, stamp):
        P, V, TP = self.P, self.P.verify, self.P.track
        S = P.search
        pending_first = S.verify_pending_first and min(S.budget_ms, TP.frame_budget_ms) > 0
        now = time.perf_counter()
        deadline = min(self._deadline, now + S.budget_ms / 1000)
        if self.click is not None and (self._click_probe or any(c.from_click for c in self.cands)):
            # 点击首锁保证至少 10ms：1080p 上光流+预处理经常已经吃满 28ms 预算，
            # 否则点击后每帧都在预算守卫处返回，候选建了也永远等不到确认帧。
            deadline = max(deadline, now + 0.010)
        H, W = fr.shape
        region, r_lo, r_hi, ref, ref_r = self._region(W, H)
        region = (max(0, region[0]), max(0, region[1]), min(W, region[2]), min(H, region[3]))
        self.dbg['region'] = region
        self.dbg['r_range'] = (r_lo, r_hi)
        flow_only = self.phase == REACQUIRE and self.reacq.get('flow_only', False)
        if flow_only and (T is None or not self._local_motion_reliable()):
            self.reacq['flow_only'] = flow_only = False
        if self.phase == REACQUIRE and T is not None and self.tgt is not None and not self.cands:
            cx, cy, r = self.tgt
            age = stamp - self.last_detect
            prior = self.recovery_candidate
            local = (flow_only and 0 < age <= min(TP.coast_s, TP.reacq_expand_s) and 0 <= cx < W and 0 <= cy < H
                     and (prior is None or self._same(prior, cx, cy, r)))
            if local:
                # 光流仍可靠时只复查预测位置，给短时模糊留出恢复时间。
                if 'track_meas' not in self.dbg:
                    m = measure_channels(fr, self.cam, cx, cy, r, V, 'local', local=True)
                    self.dbg['recover_meas'] = m
                    if m.ok and self._recovery_identity(fr, m) and m.identity_score >= TP.identity_score:
                        if self._confirm_recovery(m, stamp, local=True):
                            self._accept_recovery(m, stamp)
                            return
                    else:
                        self.recovery_candidate = None
                self.out = PREDICTED
                self.dbg['prediction_source'] = 'flow'
                return
            if ('track_meas' not in self.dbg and time.perf_counter() < deadline
                    and not (prior is not None and prior.meas.source == 'template')):
                m = measure_channels(fr, self.cam, cx, cy, r, V, 'lock', local=True)
                self.dbg['recover_meas'] = m
                shifted = (TP.local_reacq_hits > 0 and self.references and m.reason.startswith('shift ')
                           and math.hypot(m.cx - cx, m.cy - cy) <= 0.75 * r)
                if ((m.ok or shifted) and self._recovery_identity(fr, m)
                        and (not shifted or m.identity_score >= TP.identity_score)):
                    if shifted:
                        m.ok, m.reason = True, ''
                        self.dbg['recovery_shift'] = math.hypot(m.cx - cx, m.cy - cy)
                    if self._confirm_recovery(m, stamp, local=bool(shifted)):
                        self._accept_recovery(m, stamp)
                    return
        # 先确认已有几何候选，无候选时交替外观匹配与区域搜索。
        try_template = not self.cands and (self.recovery_candidate is not None or self.n_frames % 3 != 0
                        or self._template_seeds is not None
                        or self.last_detect is None
                        or stamp - self.last_detect <= min(TP.coast_s, TP.stable_no_flow_s))
        if pending_first and self.pending and self.recovery_candidate is None and not self._template_seeds:
            try_template = False
        if self.phase == REACQUIRE and try_template:
            m = self._template_measure(fr, stamp)
            if self.ambiguous:
                self.recovery_candidate = None
                return
            if m is not None:
                if self._confirm_recovery(m, stamp):
                    self._accept_recovery(m, stamp)
                return
            self.recovery_candidate = None
        if (self.phase == REACQUIRE and self._search_meta is not None
                and self._search_meta.get('kind') == 'template'):
            # 模板找回在后台跑：主循环不阻塞，先按光流预测输出，
            # 结果下一帧由 _poll_search 收回来再做几何复核。
            self.out = PREDICTED
            self.dbg['prediction_source'] = 'flow'
            return
        if (self.phase == REACQUIRE and TP.recovery_local_s > 0 and self._template_frame == self.n_frames
                and time.perf_counter() >= deadline):
            return
        if region[2] <= region[0] or region[3] <= region[1]:
            self.pending = []
            return
        # 点击直测优先于常规搜索和整帧预算：慢机器上光流/预处理就能吃掉 28ms，
        # 原来它在预算守卫之后，点击阶梯经常整帧轮不到（实测点了 10 秒才锁上）。
        # 已有点击候选时先不复核阶梯，避免同一位置每帧重复建候选。
        if (self._click_probe and S.click_probe and self.click is not None
                and not any(c.from_click for c in self.cands)):
            px, py, coarse, r0 = self._click_probe.pop(0)
            probe_deadline = max(deadline, time.perf_counter() + 0.006)
            self._probe_click(fr, px, py, r_lo, r_hi, stamp, coarse=coarse, r0=r0,
                              deadline=probe_deadline)
        if not pending_first and time.perf_counter() >= deadline:
            return
        any_pass = False
        ed_idle = self._search_future is None or self._search_future.done()
        clicked = self.click is not None
        geometry_only = TP.small_ring_geometry_hits > 0 and clicked and not self.references and T is not None

        def strong_small_ring(m):
            return (geometry_only and m.ok and m.r <= min(S.r_min, S.click_focus_r_max)
                    and math.dist((m.cx, m.cy), ref) <= (S.click_focus_half or S.click_tile)
                    and m.sectors == V.sectors and m.max_gap == 0 and m.inlier_ratio >= .95 and m.rms <= .35)

        # 1. 已有候选：搬运后在当前帧复核；还没有 EDCircles 支持的，每帧再试一次候选级放大检测
        confirm_end = (time.perf_counter() + deadline) * 0.5
        n_confirmed = 0
        for c in sorted(self.cands, key=lambda c: (self.phase == SEARCH and c.ed_hits == 0, -sum(c.hits))):
            if c.from_click and c.created == stamp:
                continue      # 本帧刚由点击直测建立并已记过一次，同一帧不能算两帧确认
            if time.perf_counter() >= confirm_end and not (pending_first and n_confirmed == 0):
                break
            n_confirmed += 1
            d = _delta(V.delta_search if T is None else V.delta_track, c.r)
            m = measure_channels(fr, self.cam, c.cx, c.cy, c.r, V, 'lock', d,
                                 local=self.click is not None)
            if m.ok and not self._recovery_identity(fr, m):
                m.ok, m.reason = False, 'appearance'
            score = -1.0
            if m.ok and not self.references:
                # 累积相对地面的残差，排除移动的近圆亮块。
                drift = c.drift + (m.cx - c.cx, m.cy - c.cy)
                score = self._reference_score(fr, m, [c.reference]) if c.reference is not None else -1
                if score < TP.identity_score or np.linalg.norm(drift) > max(3.0, 0.5 * m.r):
                    m.ok, m.reason = False, 'initial identity'
                else:
                    c.drift = drift
            if self.references:
                if c.frame != self.n_frames - 1:
                    c.hits.clear()
                if m.ok:
                    c.needed = max(c.needed, TP.reacq_confirm_hits if m.identity_score >= TP.identity_score else TP.reacq_weak_hits)
            if strong_small_ring(m) and score >= max(.90, TP.identity_score):
                c.geometry_hits = c.geometry_hits + 1 if c.frame == self.n_frames - 1 else 1
            else:
                c.geometry_hits = 0
            c.frame = self.n_frames
            c.hits.append(m.ok)
            if m.ok:
                c.cx, c.cy, c.r, c.meas, c.last_pass = m.cx, m.cy, m.r, m, stamp
                any_pass = True
                if c.ed_hits == 0 and (ed_idle or self.n_frames % 4 == 0) and ed_support(fr, m, S, V):
                    c.ed_hits = 1

        found = self.pending
        waiting = {id(f) for f in found}
        self.pending = []
        n_ver = 0
        full_quota = [max(1, TP.max_verify // 8)]   # 快筛失败后仍允许完整复核的候选数
        def rank_candidates(candidates):
            ranked = [f for f in merge_candidates(candidates)
                      if region[0] <= f.x < region[2] and region[1] <= f.y < region[3]
                      and r_lo / 1.15 <= f.r <= r_hi * 1.15]
            if ref is not None:
                ranked.sort(key=lambda f: ((int(math.hypot(f.x - ref[0], f.y - ref[1]) /
                                                max(32, S.click_tile / 2)) if clicked else 0),
                                            -f.score, id(f) not in waiting,
                                            math.hypot(f.x - ref[0], f.y - ref[1])))
            self.dbg['found'].extend((f.x, f.y, f.r) for f in ranked)
            return ranked

        def verify_candidates(candidates, allow_one=False):
            nonlocal n_ver, any_pass
            for index, f in enumerate(candidates):
                now = time.perf_counter()
                if n_ver >= TP.max_verify:
                    return candidates[index:]
                # 完整复核的耗时随半径增长（实测 3~20 ms），开始前必须给够剩余预算；
                # 否则单次复核就能把整帧拖长到 40 ms 以上，画面就卡了。
                if now + max(0.5, 0.8 * self._verify_cost_ms) / 1000 > deadline \
                        and not (allow_one and n_confirmed + n_ver == 0):
                    return candidates[index:]
                same = [c for c in self.cands if self._same(c, f.x, f.y, f.r)]
                if same:
                    if f.score >= ED_SCORE:
                        for c in same:
                            c.ed_hits += 1
                    continue
                n_ver += 1
                if clicked:
                    # 单遍快复核先筛掉大部分纹理候选；完整复核只保证点击附近最近的几个。
                    probe = measure_channels(fr, self.cam, f.x, f.y, f.r, V, 'track', local=True,
                                             quick=True)
                    if not probe.ok:
                        if full_quota[0] <= 0:
                            self.dbg['rejected']['prefilter'] = self.dbg['rejected'].get('prefilter', 0) + 1
                            continue
                        full_quota[0] -= 1
                t_ver = time.perf_counter()
                m = measure_channels(fr, self.cam, f.x, f.y, f.r, V, 'lock', local=clicked)
                self._verify_cost_ms = 0.7 * self._verify_cost_ms + 0.3 * (time.perf_counter() - t_ver) * 1000
                m.source = 'ed' if f.score >= ED_SCORE else 'mf'
                if not m.ok or not (r_lo / 1.15 <= m.r <= r_hi * 1.15):
                    reason = m.reason or 'radius range'
                    self.dbg['rejected'][reason] = self.dbg['rejected'].get(reason, 0) + 1
                    continue
                if not self._recovery_identity(fr, m):
                    reason = f'identity {m.identity_score:.2f}'
                    self.dbg['rejected'][reason] = self.dbg['rejected'].get(reason, 0) + 1
                    continue
                if any(self._same(c, m.cx, m.cy, m.r) for c in self.cands):
                    continue
                idle = self._search_future is None or self._search_future.done()
                # 后台 ED 一直占着任务槽时，每 4 帧也要做一次候选级局部 ED 确认，避免首锁被饿死。
                m.ed_ok = m.source == 'ed' or ((idle or self.n_frames % 4 == 0) and ed_support(fr, m, S, V))
                window = max(TP.confirm_window, TP.reacq_weak_hits if self.references else TP.identity_init_hits + 2)
                candidate = _Cand(m, stamp, window)
                candidate.geometry_hits = int(strong_small_ring(m))
                candidate.needed = ((TP.reacq_confirm_hits if m.identity_score >= TP.identity_score else TP.reacq_weak_hits)
                                    if self.references else max(TP.confirm_hits, TP.identity_init_hits))
                if not self.references:
                    candidate.reference = self._identity_patch(fr, m)
                    if candidate.reference is None:
                        continue
                candidate.frame = self.n_frames
                self.cands.append(candidate)
                any_pass = True
            return []

        queued = []
        if pending_first and found:
            queued = verify_candidates(rank_candidates(found), allow_one=True)
            found = []

        ed_chans = fr.channels(V.channels)
        t0 = time.perf_counter()
        ed_cell = max(S.click_tile, min(S.ed_cell, int(math.ceil(2 * r_hi)))) if clicked else S.ed_cell
        # 重捕阶段大部分帧只做"模板找回 + 光流预测"，把唯一的后台任务槽留给模板；
        # 但每 3 帧放一次 ED/FFT：模板没命中时，框里清楚的环还能靠几何候选找回。
        reacquire = self.phase == REACQUIRE and S.async_template
        gen_paused = reacquire and self.n_frames % 3 != 0
        ed_jobs = (plan_ed(region, (W, H), r_hi, ed_cell)
                   if S.use_edcircles and edc.backend(S.ed_backend) and not gen_paused else [])
        run_mf = (S.use_matched_filter and not gen_paused
                  and (not S.async_search or self._search_future is None)
                  and (not pending_first or (n_ver < TP.max_verify and t0 < deadline)))
        mf_jobs = self.searcher.plan(region, (W, H), r_lo, r_hi,
                                    S.click_tile + 64 if clicked else None) if run_mf else []
        if clicked:
            ed_jobs.sort(key=lambda j: math.hypot((j.cell[0] + j.cell[2]) / 2 - ref[0],
                                                 (j.cell[1] + j.cell[3]) / 2 - ref[1]))
            # 先覆盖点击点的小半径任务，再轮询外围，避免金字塔分块中心影响优先级。
            mf_jobs.sort(key=lambda j: (math.hypot(max(0, j.valid[0] * 2 ** j.level - ref[0],
                                                      ref[0] - j.valid[2] * 2 ** j.level),
                                                   max(0, j.valid[1] * 2 ** j.level - ref[1],
                                                       ref[1] - j.valid[3] * 2 ** j.level)),
                                        j.radii[0] * 2 ** j.level))
        active_mf, active_cursor = mf_jobs, self.cursor
        # 小环在点击附近高频复查，其余帧继续完整范围轮询。
        # 点击框已经扩大/点击点漂出画面时，点击附近的小框没有意义，全部帧都用于完整范围。
        sweep = self.cursor + self.focus_cursor if S.click_focus_half else self.n_frames
        click_inside = clicked and 0 <= ref[0] < W and 0 <= ref[1] < H
        focused = (clicked and click_inside and self.click['level'] == 0
                   and sweep % 2 == 0 and r_lo < S.click_focus_r_max)
        if focused and run_mf:
            side = S.click_focus_half or S.click_tile
            focus = (max(region[0], ref[0] - side), max(region[1], ref[1] - side),
                     min(region[2], ref[0] + side), min(region[3], ref[1] + side))
            active_mf = self.searcher.plan(focus, (W, H), r_lo, min(r_hi, S.click_focus_r_max),
                                           S.click_tile + 64, S.click_focus_radii_per_job)
            if S.click_focus_half:
                expected = ref_r or S.r_min
                active_mf.sort(key=lambda j: (
                    math.hypot((j.valid[0] + j.valid[2]) * 2 ** (j.level - 1) - ref[0],
                               (j.valid[1] + j.valid[3]) * 2 ** (j.level - 1) - ref[1]),
                    abs((j.radii[0] + j.radii[-1]) * 2 ** (j.level - 1) - expected)))
            active_cursor = self.focus_cursor
        n_ed = n_mf = 0
        ed_request = None
        # 两种候选生成交替优先，预留剩余预算做原图复核。
        generation_end = t0 + max(0.0, deadline - t0) * 0.5
        mf_first = not clicked or self.n_frames % 2 == 0
        mf_request = None
        if S.async_search and active_mf and time.perf_counter() < generation_end:
            channels = ed_chans if clicked else ['gray']
            ch = channels[(active_cursor // len(active_mf)) % len(channels)]
            mf_request = ch, active_mf[active_cursor % len(active_mf)]
        if (not S.async_search and clicked and mf_first and active_mf
                and time.perf_counter() < generation_end and self._mf_budget_ok(deadline)):
            ch = ed_chans[(active_cursor // len(active_mf)) % len(ed_chans)]
            job = active_mf[active_cursor % len(active_mf)]
            found += self._mf_job(build_pyramid(fr.u8(ch), job.level), job)
            if focused:
                self.focus_cursor += 1
            else:
                self.cursor += 1
            n_mf += 1
        while n_ed < len(ed_jobs) and (S.async_ed or time.perf_counter() < (generation_end if clicked else deadline)):
            sweep = self.ed_cursor // len(ed_jobs)
            ch = ed_chans[(sweep if clicked else self.n_frames + sweep) % len(ed_chans)]
            preferred_r = r_lo if self.height else max(r_lo, S.r_min * (self.click['scale'] if clicked else 1))
            job = ed_jobs[self.ed_cursor % len(ed_jobs)]
            zoom = 1.0
            cw = job.crop[2] - job.crop[0]
            chh = job.crop[3] - job.crop[1]
            if max(cw, chh) > 720:
                # 点击框扩大后裁剪可达上千像素：放大只会让 ED 输入到百万像素级、结果超时被丢弃，
                # 按尺寸降采样，保证单个后台任务能在 0.35 s 的时效内跑完。
                zoom = 720.0 / max(cw, chh)
            elif clicked and (sweep // len(ed_chans)) % 2 == 0:
                zoom = min(2.0, S.ed_local_max_zoom, max(1.0, S.ed_local_target_r / preferred_r))
            if S.async_ed:
                if self._search_future is None:
                    ed_request = (fr, job, ch, r_lo, r_hi, zoom, stamp)
                break
            else:
                found += run_ed(fr.u8(ch), job, r_lo, r_hi, S, zoom)
                self.ed_cursor += 1
                n_ed += 1
            if mf_jobs or (time.perf_counter() - t0) * 1000 >= S.budget_ms:
                break
        if (not S.async_search and clicked and not mf_first and active_mf
                and time.perf_counter() < generation_end and self._mf_budget_ok(deadline)):
            ch = ed_chans[(active_cursor // len(active_mf)) % len(ed_chans)]
            job = active_mf[active_cursor % len(active_mf)]
            found += self._mf_job(build_pyramid(fr.u8(ch), job.level), job)
            if focused:
                self.focus_cursor += 1
            else:
                self.cursor += 1
            n_mf += 1
        # 重捕阶段不跑同步匹配滤波：单个任务 30~50 ms，一跑这条重捕路径就卡一下。
        # 候选由后台 ED、模板匹配和光流预测位置复核提供，这一帧先按预测输出。
        if (not S.async_search and not clicked and mf_jobs and self.phase != REACQUIRE
                and time.perf_counter() < deadline):
            pyr = build_pyramid(fr.u8('gray'), max(j.level for j in mf_jobs))
            # 没有候选、也没有待复核结果时，这一帧本来就没别的事可做：
            # 允许超预算跑一个任务（只一个），否则慢机器上匹配滤波永远排不上，
            # 只能靠 ED 生成候选，小环很久锁不上。有活干时仍然严格按预算。
            overrun = 1 if not self.cands and not self.pending else 0
            while n_mf < len(mf_jobs) and (self._mf_budget_ok(deadline) or overrun > 0):
                self.cursor %= len(mf_jobs)
                found += self._mf_job(pyr, mf_jobs[self.cursor])
                self.cursor += 1
                n_mf += 1
                if not self._mf_budget_ok(deadline):
                    overrun -= 1
                # 生成最多用一半预算：原先固定 20ms 在 1080p 上会被 5 个 MF 任务吃光，
                # 复核一个候选都跑不到（实测 found~100 / rejected 为空）。
                if time.perf_counter() >= min(deadline, t0 + max(0.0, deadline - t0) * 0.5):
                    break
        self.dbg['jobs'] = dict(ed=(n_ed, len(ed_jobs)), mf=(n_mf, len(mf_jobs)))
        queued += verify_candidates(rank_candidates(found))
        self.pending = queued[:TP.max_verify * 2]

        # 3. 清理：连续 3 帧不过、或太久没过的丢掉；两个候选测到同一个环的合并
        keep = []
        drops = {}
        reprobe = None
        for c in sorted(self.cands, key=lambda c: (-sum(c.hits), -c.last_pass)):
            # 点击候选经常是"看得见两帧、糊一帧"的小环：允许连丢 6 帧再丢，
            # 让它把 6 次通过慢慢攒够；其他候选仍然连丢 3 帧就丢。
            grace = 6 if c.from_click else 3
            if len(list(c.hits)) >= grace and not any(list(c.hits)[-grace:]):
                drops['consecutive'] = drops.get('consecutive', 0) + 1
                if c.from_click and reprobe is None:
                    reprobe = (c.cx, c.cy)
                continue
            if stamp - c.last_pass > TP.cand_timeout_s:
                drops['timeout'] = drops.get('timeout', 0) + 1
                continue
            if any(self._same(k, c.cx, c.cy, c.r) for k in keep):
                drops['duplicate'] = drops.get('duplicate', 0) + 1
                continue
            keep.append(c)
        if drops:
            self.dbg['cand_drop'] = drops
        self.cands = keep[:TP.max_cands]
        if (reprobe is not None and self.click is not None and not self.cands
                and self._reprobe_rounds < 4):
            # 点击候选连丢几帧被丢掉了，但环多半只是短暂模糊（小环高速下降时很常见）。
            # 围绕它最后的位置再用粗阶梯点几次，否则这次点击就再也没有候选来源了。
            self._reprobe_rounds += 1
            self._click_probe.append((reprobe[0], reprobe[1], True, None))
            self.dbg['reprobe'] = (round(reprobe[0], 1), round(reprobe[1], 1), self._reprobe_rounds)

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
        need_ed = (self.phase == SEARCH and not self.references
                   and TP.lock_need_edcircles and S.use_edcircles and bool(ed_jobs))
        q = [c for c in self.cands if c.last_pass == stamp and c.hits[-1] and sum(c.hits) >= c.needed
             and (not self.references or (len(c.hits) >= c.needed and all(list(c.hits)[-c.needed:])))
             and (not need_ed or c.ed_hits > 0
                  or (TP.lock_click_bypass_ed and c.from_click)
                  or (geometry_only and c.geometry_hits >= TP.small_ring_geometry_hits))]
        win = self._choose(q, ref, ref_r)
        if win is not None:
            if self.references:
                self._accept_recovery(win.meas, stamp)
                return
            if self.phase == SEARCH:
                self.target_id += 1
            self.phase = TRACK
            self.tgt = [win.cx, win.cy, win.r]
            self.tgt_meas = win.meas
            self.measured_radius = win.r
            self.last_detect = stamp
            self.cands = []
            self.reacq = None
            self.click = None
            self.out = DETECTED
            self.pending = []
            self._remember(fr, win.meas)
        elif len(q) >= 2:
            self.ambiguous = True
        # 后台只有一个任务槽：ED 和匹配滤波交替提交，避免 MF 连续占槽把 ED 饿死
        # （首锁要求 EDCircles 支持，ED 一直跑不到就永远锁不上）。
        if self.phase in (SEARCH, REACQUIRE):
            if ed_request is not None and (mf_request is None or self._last_search_kind == 'mf'):
                if self._submit_ed(*ed_request):
                    self.ed_cursor += 1
                    self.dbg['jobs']['ed'] = (1, len(ed_jobs))
            elif mf_request is not None:
                ch, job = mf_request
                image = fr.u8('gray') if ch == 'gray' else fr._bgr
                if self._submit_search('mf', stamp, np.eye(3), self._run_mf, image.copy(), ch, job):
                    if focused:
                        self.focus_cursor += 1
                    else:
                        self.cursor += 1
                    self.dbg['jobs']['mf'] = (1, len(mf_jobs))

    def _choose(self, q, ref, ref_r):
        TP = self.P.track
        if not q:
            return None
        if len(q) == 1:
            return q[0]
        if self.references:
            ranked = sorted(q, key=lambda c: c.meas.identity_score, reverse=True)
            if ranked[0].meas.identity_score - ranked[1].meas.identity_score < TP.identity_margin:
                return None
            return ranked[0]
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
            pyr = build_pyramid(fr.u8('gray'), max(j.level for j in jobs))
            found += [c for j in jobs for c in s.run(pyr, j)]
    out = []
    for c in merge_candidates(found, limit):
        m = measure_channels(fr, cam, c.x, c.y, c.r, V, 'lock')
        m.source = 'ed' if c.score >= ED_SCORE else 'mf'
        if m.ok:
            m.ed_ok = S.use_edcircles and bool(edc.backend(S.ed_backend)) and ed_support(fr, m, S, V)
        out.append((c, m))
    return out, fr.noise('gray')
