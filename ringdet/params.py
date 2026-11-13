"""全部可调参数。默认值对应 ~/桌面/白色石灰圆环识别方案_落地版.md。

调参时不改代码：写一个只含要改字段的 yaml，用 --params 传入；
或命令行 --set verify.k_sigma=2.0 覆盖单项。
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class RingParams:
    diameter_m: float = 2.0      # 名义直径
    line_width_m: float = 0.10   # 只决定搜索核中间带宽度；复核按图像实测线宽


@dataclass
class SearchParams:
    r_min: float = 8.0           # 没有高度时的半径范围（原图 px）
    r_max: float = 60.0
    height_band: Tuple[float, float] = (0.8, 1.25)   # 有高度时：fx*R/h 乘以这个范围
    r_step: float = 1.06         # 半径等比步长
    level_r_max: float = 20.0    # 层内半径超过它就去下一层金字塔
    tile: int = 512              # 分块边长（层内 px），块间重叠 = 最大核半径
    radii_per_job: int = 6       # 每个任务 = 一块 x 至多这么多个半径
    budget_ms: float = 18.0      # 搜索候选生成与复核共用的每帧预算（Nano 上 40ms 会让画面发卡）
    verify_pending_first: bool = False  # 优先复核积压候选，预算耗尽时也允许一次复核
    min_score: float = 2.0       # S = min(M-I, M-O) / max(sigma, noise_floor)
    side_ratio: float = 3.0      # |I-O| > side_ratio*S 的位置不当候选
    noise_floor: float = 2.0
    m_band_min: float = 1.5      # 中间带最小宽度（层内 px）
    side_band_min: float = 2.0   # 内外背景带最小宽度
    cell: int = 128              # 候选分格（原图 px）
    per_cell: int = 3
    per_tile: int = 12
    # EDCircles（ED_Lib C++，./build.sh 编译）：主力候选来源，结果走同一套复核
    use_edcircles: bool = True
    ed_backend: str = 'auto'     # auto / edlib / opencv（cv2.ximgproc 移植版）
    ed_gradient: int = 8         # ED 梯度阈值（桌面测试图上 8 比默认 20 多找到暗环，误检不增加）
    ed_anchor: int = 0
    ed_sigma: float = 1.0
    ed_min_path: int = 10
    ed_validate: bool = False    # True = 先用 EDPF 筛边缘段（ED_Lib 默认），会删掉高空细环的弱边缘
    ed_min_axis_ratio: float = 0.6   # 椭圆短轴/长轴低于它不要（下视倾斜 30° 时约 0.87）
    ed_cell: int = 640           # 区域 ED 的分格边长（原图 px）
    # 候选级 EDCircles：复核通过的候选，在它周围的小块上放大后再跑一次 EDCircles（高空细环全图跑连不成圆）
    ed_local: bool = True
    ed_local_max_r: float = 80.0       # 只对半径不超过它的候选做本地放大 ED；大环全图 ED 已能连成圆，本地复核又慢又没用
    ed_local_target_r: float = 24.0    # 放大到半径约这么大
    ed_local_max_zoom: float = 4.0
    ed_local_margin: float = 2.2       # 小块半边长 = margin * r
    ed_local_center_tol: Tuple[float, float] = (1.5, 0.15)   # 圆心吻合：<= max(a, b*r)
    ed_local_radius_tol: float = 0.15
    # 匹配滤波补高空细环、断续环；点击后与 EDCircles 交替优先运行。
    use_matched_filter: bool = True
    # 点击后优先扫描附近的小环；搜索框随地面运动搬运，持续无候选时逐级扩大。
    click_box: float = 600.0     # 点击搜索框初始边长（原图 px）
    click_r_min: float = 4.0     # 点击先验下的小环搜索下限，不改变无点击搜索范围
    click_tile: int = 192        # 点击搜索任务的有效区边长，按离点击点的距离轮询
    escalate_s: float = 3.0      # 框内这么久没有候选通过复核就扩大一级
    click_max_levels: int = 1    # 最多扩大几级（每级边长 x2）；0 = 只在初始框里找
    click_only: bool = True      # True：没点击就不搜索（输出 LOST）；False：没点击时全图搜索
    click_probe: bool = True     # 点击当帧先在点击点按 1.25 半径阶梯直接复核一次（操作员点的就是环）
    fft_cache_mb: float = 96.0    # 核频谱缓存上限
    async_ed: bool = True        # 区域 ED 在单个后台线程执行，结果在当前帧重新验证
    async_template: bool = True  # 重捕的模板匹配放后台线程；主循环先输出光流预测，不阻塞单次 30~50ms 的匹配
    async_search: bool = False  # FFT 和大范围模板匹配使用同一个后台搜索线程
    click_focus_r_max: float = 12.0  # 点击附近高频复查的小环半径上限
    click_focus_half: int = 0  # 0 = 使用 click_tile 作为局部搜索半边长
    click_focus_radii_per_job: int = 0  # 0 = 使用 radii_per_job


@dataclass
class VerifyParams:
    backend: str = 'auto'        # auto / python；原生计算不可用时保留 Python 路径
    rays_small: int = 64
    rays_large: int = 128
    rays_large_r: float = 30.0   # 半径 >= 它用 rays_large 条射线
    step: float = 0.5            # 射线采样间隔（px）
    delta_search: Tuple[float, float] = (4.0, 0.3)   # 找峰半宽 = max(a, b*r)
    delta_track: Tuple[float, float] = (3.0, 0.15)
    width_ratio: Tuple[float, float] = (0.02, 0.2)   # 线宽候选 = r*ratio，下限 1 px（2 m 环上约 2~20 cm）
    gap: float = 1.0             # 亮线与背景带之间留的空隙（px）
    k_sigma: float = 2.5         # 有效射线：min(峰-内背景, 峰-外背景) >= max(k_sigma*sigma, min_contrast)
    min_contrast: float = 5.0
    sectors: int = 16
    sector_min_rays: int = 2     # 扇区有效：内点射线 >= max(sector_min_rays, sector_min_frac*每扇区射线数)
    sector_min_frac: float = 0.5
    side_ratio: float = 3.0      # 射线 |内背景-外背景| > side_ratio*min(峰-内, 峰-外) 时作废（亮物边缘振铃）
    ransac_iters: int = 100
    inlier_tol: Tuple[float, float] = (1.0, 0.04)    # 内点：到拟合圆距离 <= max(a, b*r)
    # 复核用的通道，按顺序试，前一个不通过才试下一个。white = min(B,G,R)，有颜色的地面上石灰对比更大
    channels: Tuple[str, ...] = ('gray', 'white')
    use_ellipse: bool = True     # 圆模型扇区不全时试椭圆模型（斜视时圆环是椭圆）
    ellipse_min_axis_ratio: float = 0.6   # 短轴/长轴下限（倾斜约 53°）
    # 首锁门槛
    lock_sectors: int = 14       # tools/eval_gates.py：12->14 后合成图误检 6->0，召回 42->40/45
    lock_max_gap: int = 2
    lock_min_inlier_ratio: float = 0.85
    lock_max_rms: Tuple[float, float] = (0.6, 0.04)  # 首锁：内点残差 RMS <= max(a, b*r)；手画石灰环不够圆，0.4 会漏掉 r>=14 的真环
    lock_shift: float = 0.5      # 中心相对候选移动 <= lock_shift*r
    lock_radius_tol: float = 0.25
    # 跟踪门槛
    track_sectors: int = 8
    track_max_gap: int = 6
    track_shift: Tuple[float, float] = (3.0, 0.3)    # 相对预测 <= max(a, b*r)
    track_radius_tol: float = 0.2
    # 共用门槛
    min_inlier_ratio: float = 0.6
    max_rms: Tuple[float, float] = (0.8, 0.05)       # 跟踪：内点残差 RMS <= max(a, b*r)；必须不严于首锁的 lock_max_rms


@dataclass
class MotionParams:
    width: int = 960             # 光流在这个宽度上算
    max_corners: int = 300
    quality: float = 0.01
    min_distance: int = 20
    fb_max: float = 1.0          # 前后向回到原位的误差上限（光流图 px）
    ransac_px: float = 2.0
    min_inliers: int = 50
    grid: int = 4                # 内点覆盖检查：grid x grid 格子里至少 min_cells 格有内点
    min_cells: int = 4
    backend: str = 'auto'        # auto / cpu / cuda；auto 在 CUDA 不可用时回退 CPU
    refresh_frames: int = 8      # 角点定期补充，数量或局部覆盖不足时提前补充
    refresh_ratio: float = 0.6   # 剩余角点少于 max_corners*它 时提前补充


@dataclass
class TrackParams:
    confirm_window: int = 5      # 最近 confirm_window 个处理帧中
    confirm_hits: int = 3        # 至少 confirm_hits 帧通过，且当前帧通过
    identity_init_hits: int = 6
    small_ring_geometry_hits: int = 0  # 小圆未获 ED 支持时所需的连续强复核帧数
    local_reacq_hits: int = 0      # 可靠局部光流内原目标重捕的连续复核帧数
    assoc: Tuple[float, float] = (3.0, 0.25)   # 同一候选：中心差 <= max(a, b*r)
    assoc_radius: float = 0.15   # 且半径差 <= 15%
    max_cands: int = 30
    max_verify: int = 6          # 每帧最多复核多少个新候选（一次完整复核可达 10~20ms，不能放开）
    cand_timeout_s: float = 1.0  # 候选这么久没通过就丢弃
    coast_s: float = 0.3
    track_miss_grace: int = 2      # 跟踪态允许连续失败几帧再判丢失（期间输出预测）
    shift_retry_r: float = 60.0    # 复核失败时在预测点周围补试的最大半径（px）；0 = 关闭
    stable_track_frames: int = 6
    stable_no_flow_s: float = 0.12
    stable_template_grace_s: float = 0.0  # 光流可靠时短暂等待几何复核恢复
    appearance_coast_s: float = 0.5
    appearance_score: float = 0.85
    appearance_psr: float = 2.0
    reacq_box_r: float = 6.0     # 重捕框边长 = 6r
    reacq_box_min: float = 240.0  # 小环重捕框的最小边长（原图 px）
    reacq_expand_s: float = 1.0  # 之后扩到全图、半径 ±50%
    reacq_giveup_s: float = 3.0  # 之后回到 SEARCH
    lock_need_edcircles: bool = True   # 首锁要求 EDCircles 至少找到过一次（EDCircles 不可用时自动不要求）
    lock_click_bypass_ed: bool = True  # 点击直测的候选由操作员指定，不再要求 ED 支持，但仍需多帧严格复核
    prefer_ratio: float = 0.5    # 两个候选都合格时，离参考点（点击点，没有则画面中心）距离比 < 0.5 的胜出
    prefer_dist_r: float = 2.0   # 重捕时优先最后位置 2r 以内的候选
    # 下降中途丢失后回到全图搜索时，半径下限 = 最后跟踪半径 x 它（飞机只降不升，环不会变小太多）；
    # 防止近地出画后锁到地上一圈碎石之类的小圆。有高度时与高度给出的范围取交集；reset() 或点击会清掉。
    lost_r_floor: float = 0.5
    template_score: float = 0.65  # 局部外观匹配通过后仍需几何复核
    identity_score: float = 0.8   # 重捕身份分；较弱结果还须严格几何与更长确认
    identity_every: int = 1       # 稳定跟踪时每 N 帧做一次身份比对；1 = 每帧
    fast_measure_every: int = 0   # 稳定跟踪时每 N 帧做一次完整复核，其余帧走单遍快路径；0 = 关闭
    identity_margin: float = 0.06  # 两个重捕候选的身份分太接近时暂不接回
    reacq_confirm_hits: int = 3
    reacq_weak_hits: int = 5
    template_update_hits: int = 8  # 重捕后通过稳定测量才更新身份参考
    learn_rate: float = 0.15     # 可靠测量的工作模板、形状和运动统计更新率
    frame_budget_ms: float = 28.0  # 包含预处理和光流的整帧软预算
    template_backend: str = 'auto'  # auto / cpu / cuda；CUDA 用于较大的重捕图块
    template_cuda_min_area: int = 0  # auto 模式使用 CUDA 的最小搜索面积（px²）
    recovery_local_s: float = 0.0    # 光流可靠时先在预测点附近重捕的时长
    stagger_recovery_scales: bool = False  # 分帧搜索重捕尺度
    compact_gpu_matching: bool = True
    online_learning: bool = True   # 背景感知相关滤波，只产生重捕候选
    online_min_radius: float = 8.0  # 小于此像素半径保留原像素模板路径
    online_update_frames: int = 3  # 可靠帧之间的最小滤波器更新间隔
    online_psr: float = 8.0        # 响应峰相对背景的显著性下限


@dataclass
class Params:
    ring: RingParams = field(default_factory=RingParams)
    search: SearchParams = field(default_factory=SearchParams)
    verify: VerifyParams = field(default_factory=VerifyParams)
    motion: MotionParams = field(default_factory=MotionParams)
    track: TrackParams = field(default_factory=TrackParams)

    @classmethod
    def load(cls, path=None, overrides=()):
        p = cls()
        import yaml
        if path:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            if path.endswith('.json'):
                data = json.loads(text)
            else:
                data = yaml.safe_load(text) or {}
            p.update(data)
        for item in overrides:
            key, _, value = item.partition('=')
            sec, _, name = key.strip().partition('.')
            p.update({sec: {name: yaml.safe_load(value)}})
        return p

    def update(self, data):
        for sec, kv in data.items():
            if not hasattr(self, sec):
                raise KeyError(f'未知参数段: {sec}')
            obj = getattr(self, sec)
            for k, v in kv.items():
                if not hasattr(obj, k):
                    raise KeyError(f'未知参数: {sec}.{k}')
                cur = getattr(obj, k)
                if isinstance(cur, tuple):
                    v = (v,) if isinstance(v, str) else v
                    v = tuple(str(x) if isinstance(x, str) else float(x) for x in v)
                elif isinstance(cur, bool):
                    v = bool(v)
                elif isinstance(cur, int):
                    v = int(v)
                elif isinstance(cur, float):
                    v = float(v)
                setattr(obj, k, v)

    def to_dict(self):
        return dataclasses.asdict(self)
