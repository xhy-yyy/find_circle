# find_circle：白色石灰圆环识别与跟踪

无人机下视相机找地面的白色石灰圆环（直径约 2 m），输出圆心像素坐标、半径和状态。不用神经网络，不需要训练数据。

**上手：** 所有操作都用 `./run.sh`（`./run.sh help` 列出命令），步骤见 [docs/使用教程.md](docs/使用教程.md)。

## 使用流程

1. 操作员在画面上粗点一下环的大致位置。
2. 点击后优先扫描附近的小半径圆环，搜索下限为 4 px。小块匹配滤波与局部增强、放大后的 EDCircles 交替优先运行，灰度和白度通道轮询；复核在原图完成。搜索框随地面光流移动、随下降放大，3 s 没找到就扩大一级（默认最多扩一级）。
3. 首次建档默认要求 8 帧中至少 6 帧通过，当前帧通过，并且有 EDCircles 支持。候选图块须保持外观一致，累计运动须与地面预测一致，避免把白衣路人等移动亮块存为身份参考。确认后首锁并输出 `DETECTED`。
4. 首锁后优先用目标附近的光流预测位置，每帧重新测量。几何暂时失败时，直接匹配最近可靠关键帧；外观峰值、峰值唯一性及参考匹配均合格，且不与有效运动链冲突，才短时输出外观定位的 `PREDICTED`，最多 0.5 s，期间冻结学习。仅有有效光流时最多预测 0.3 s。稳定跟踪后光流和测量同时失效时最多观察 0.12 s，无可靠位置则输出 `LOST`。待确认候选可作为下一帧搜索中心，仍需身份验证及多帧确认。重捕搜索范围随位移和运动误差扩大，原图几何校正圆心；弱几何不改写可靠尺度，只在后续测量失败时作为一次局部复核的备选半径，且须通过身份检查。进入重捕后，光流有效时先在预测位置按首锁几何门限复查，候选仍须多帧确认，再运行外观和区域搜索。重捕 1 s 后扩大到全图，3 s 后回到搜索状态；全图重捕也保留经运动传播的目标尺度上限。
5. 连续测量沿用跟踪门限；重捕接回另做身份验证，比较圆环及周围地面纹理。明确且位移较小的结果可直接修正，其他结果要连续 3 帧确认；身份分较低时须通过首锁几何门限并连续 5 帧确认。两个目标难以区分时输出 `ambiguous`，不强行接回。重新点击会清空原身份，允许人工纠正附近的错误目标。

在线外观学习只使用可靠测量。默认启用背景感知相关滤波，学习圆环与附近背景的区别，环带权重高、少量邻域用于辅助；工作模板渐进更新，最近可靠关键帧直接保存。确认后的首张身份参考固定保留，另存两张近期参考和一张按尺度、视角变化选择的历史参考。旧参考不再匹配时，只有地面运动有效、严格几何测量合格、圆心修正与地面预测一致，且最近关键帧未超过 0.3 s，才继续刷新用于定位候选的关键帧；工作模板还要求与上一关键帧匹配分至少 0.9，才低速更新，连续 8 次合格后才补入近期身份参考。关键帧生成的候选仍须通过身份检查，预测或丢失会清零学习计数。重捕期间冻结模型，找回后经过至少 8 次可靠复核才恢复更新。运动、线宽统计和可靠尺度由各自合格的几何测量维护。学习门槛不打断已经通过的连续跟踪测量，无需神经网络、模型权重和现场离线训练。

没点击时不搜索，一直输出 `LOST`。这是为了少出误识别，可以用 `--set search.click_only=false` 改成全图搜索。

## 算法

| 步骤 | 做法 |
| --- | --- |
| 找候选 | EDCircles（[ED_Lib](https://github.com/CihanTopal/ED_Lib)，C++，MIT）与多尺度环带匹配滤波；点击后优先覆盖点击点的小半径任务，周边任务轮询；局部 CLAHE 增强与放大只用于找候选 |
| 复核 | 64～128 条射线找"两侧都比它暗"的亮峰，RANSAC 拟圆，圆不够时拟椭圆（斜视）；看 16 个扇区的覆盖、最大缺口、内点率、残差。先在灰度上复核，不通过再在白度通道上复核 |
| EDCircles 支持 | 全图检出；或在复核通过的候选周围取一小块、放大到半径约 24 px 后再跑 EDCircles，圆心半径吻合就算 |
| 首锁 | 扇区 ≥ 14/16、缺口 ≤ 2、内点 ≥ 85%；默认 8 帧中 6 帧通过；外观一致、运动符合地面；有 EDCircles 支持 |
| 帧间预测 | 复用通过前后向检查的角点，按覆盖情况补点；局部裁块提角点，CPU 前后向 LK 共用金字塔，CUDA 可用时使用 GPU 角点和光流；保留 RANSAC 及运动失效回退 |
| 局部找回 | 在线滤波候选经几何与身份检查后使用；保留可靠关键帧、近期参考和首锁参考的 NCC 回退。关键帧与当前图像直接匹配，大区域先缩放匹配，再在原图拟合。外观预测还须核对环带核心、参考和运动的一致性，不能作为几何实测或学习样本 |

白度通道是 min(B, G, R)。石灰三个颜色分量都高，取最小值后几乎不变暗；黄沙、红土、草地至少有一个分量低，取最小值后明显变暗，环更突出。黄沙测试图上，环的对比从灰度的 6.4 提高到 11.0。

高空小环：像素半径取决于分辨率和镜头，1280 px 宽、水平视场约 90° 时，100 m 高的 2 m 圆环半径仅约 6.4 px。点击后从 4 px 搜索，并提前在附近小块上增强、放大找候选；最后仍用原图亮线、扇区覆盖、拟合残差和多帧确认排除纹理误检。复核噪声按目标附近估计，保留噪声下限。

单独用 EDCircles 时，草丛、长椭圆土块、暗斑、枯枝圈也会被当成圆；单独用匹配滤波时，每张图会多出 5～46 个纹理小圈。两者都要过复核，首锁还要求 EDCircles 支持，这两类误检都去掉了。

在线滤波借鉴 [BACF](https://arxiv.org/abs/1703.04590) 的背景样本学习和空间约束，采用局部灰度对比、白度对比及梯度幅值三种手工特征；是针对圆环的实现，并非原论文的 HOG 配置。可靠图块归一化到 48×48，在扩展背景上学习中心响应、压低背景响应，滤波器的空间支持集中在环带附近。学习率随身份分、几何质量和响应峰显著性（PSR）降低；低置信度冻结更新。滤波器在 CPU 上执行，不依赖 CUDA。

默认只对半径至少 8 px 的目标启用在线滤波；更小的圆使用原像素检测、含周边纹理的模板和关键帧路径。在线滤波候选仍经几何复核；关键帧 NCC 的高可信定位以 `PREDICTED` 输出，预览标记 `KEYFRAME`，不更新最后实测时间、身份参考或在线模型。预览中的 `Online CF` 显示滤波器状态，`Updates` 是成功更新次数，`Age` 是距上次更新的处理帧数，`Hold` 是重捕后更新身份参考前仍需通过的稳定复核次数。可用 `--set track.online_learning=false` 关闭在线滤波作对照，关键帧路径仍保留；`--set track.appearance_coast_s=0` 关闭外观预测。

## 安装

```bash
sudo apt install cmake g++ libopencv-dev python3-opencv python3-numpy python3-yaml
cd ~/find_circle
./build.sh                 # 编译 ED_Lib -> ringdet/_lib/libedcircles.so
python3 -m pytest -q tests/
```

`python3 -m ringdet.edcircles` 打印当前用的 EDCircles 后端。没编译 `.so` 时会退回 opencv-contrib 的 `cv2.ximgproc`（同源移植，结果略有不同）；两者都没有时只用匹配滤波，首锁不再要求 EDCircles。

Jetson 上同样运行 `./build.sh`，JetPack 自带的 OpenCV 就够，不需要 contrib 和 CUDA。

## 测试图片和视频

```bash
# 单张图或文件夹。默认只列可首锁的结果；--all 也列只靠匹配滤波通过的，并画出未通过的原因
python3 tools/run_image.py 图片或文件夹 [--out out/xxx] [--click x,y --box 600] [--all]
# 排查某处为什么没检测到：直接在该位置复核并打印每项指标
python3 tools/run_image.py 图.png --probe x,y,r

# 视频：完整跟踪流程。--click 给第 0 帧的点击位置（原图 px），--show 可以用鼠标点
python3 tools/run_video.py 视频.mp4 --click x,y [--out out/xxx.mp4] [--jsonl out/xxx.jsonl] [--show]

# 只看 EDCircles 本身的效果（不复核）
python3 tools/run_edcircles.py 图片或文件夹 --grad 8 [--gt data/desktop_gt.json]
```

输出的标注图里：绿圈是通过的结果，底部是放大图（红点是复核用的亮峰点）。

## 机载运行

```bash
# 查看相机支持的采集格式
v4l2-ctl -d /dev/video0 --list-formats-ext

CALIB=cam.yaml UDP_OUT=地面站IP:9000 LISTEN=9001 ./run.sh camera
```

默认保持摄像头原有曝光和对焦设置。曾设置手动曝光、需要恢复自动时，运行 `EXPOSURE=auto ./run.sh camera`。户外需要限制运动拖影时，可用 `EXPOSURE=20 ./run.sh camera` 显式设置 2 ms 手动曝光，并根据画面亮度调试。数字曝光值的单位是 100 µs；控件名称以 `v4l2-ctl -l` 为准。

Jetson Orin Nano 更新代码后先运行 `./build.sh`，在板端编译 CPU 金字塔复用、射线评分、圆心精修和椭圆假设验证函数。Python OpenCV 与 C++ OpenCV 应使用同一套安装，不能复制电脑上的 `.so` 到板端。CUDA 需要 OpenCV 实际编译相应模块，仅安装 CUDA 驱动不会自动启用这些接口。

```bash
SHOW=1 ./run.sh camera --threads 4 --set motion.backend=auto --stats --quiet
./run.sh camera --set motion.backend=cuda       # 要求 CUDA；不可用时明确报错
./run.sh camera --set motion.backend=cpu        # 只用 CPU
REC_DIR="" ./run.sh camera                      # 关闭录像和文件日志
```

香橙派使用相同代码，在板端重新编译后运行 `SHOW=1 ./run.sh camera --threads 4 --set motion.backend=cpu --set track.template_backend=cpu --stats --quiet`。RK3588 的 GPU 不使用 CUDA，当前走 CPU 加速路径。两块板都保留摄像头原有曝光设置；对比性能时保持分辨率、录像、预览配置一致，可分别比较 `--threads 2` 和 `--threads 4`。

`auto` 默认尝试 CUDA，设备或接口不可用时回退 CPU；启动时打印运动估计和重捕匹配的实际后端。较大的重捕图块支持 CUDA 归一化互相关，缺少模块时回退 CPU 的共享 FFT 路径；`--set track.template_backend=cpu` 可单独关闭 GPU 匹配。区域 ED 搜索默认在单个后台线程执行，候选经有效运动链补偿后仍须在当前原图复核；运动链失效、超过 0.25 s 或重新点击时丢弃。处理帧采用 28 ms 软预算，完整重捕验证和单次几何复核仍可能超时，这不是已验证的板端 30 FPS 保证。

几何复核默认在 CPU 原生库中计算多档射线评分与双精度圆心精修；椭圆随机样本按生成器完整状态缓存，样本及后续随机状态与原路径一致。搜索范围、随机假设数、几何／身份门限、确认帧数和学习规则沿用原配置。`--set verify.backend=python` 可切回原有评分、精修和采样路径作对照；库未更新或相应函数不可用时自动回退。

GPU 匹配缓存最多 32 个模板，缓存图像数据不超过 16 MiB，并复用响应缓冲区。启动预热时按小、中、大响应区域比较“GPU 峰值筛选”和“下载响应图后 CPU 筛选”的完整耗时，仅启用结果一致且实测更快的路径。GPU 筛选只返回前两个峰、均值和标准差；同分峰、低方差、PSR 临界值或接口不支持时沿用原响应图筛选。启动日志显示各区域的分配；`--set track.compact_gpu_matching=false` 可关闭 GPU 峰值筛选，保留 GPU 匹配和缓存。EDCircles、在线滤波、身份确认与状态机继续使用 CPU。

相机录像使用有界后台队列，不阻塞识别；安装 `ffmpeg` 且相机后端支持原始 MJPEG 时直接保存压缩码流，否则后台编码。`--record-mode encode` 可固定使用编码路径，结束时显示写入与丢弃帧数。预览默认 15 FPS，识别仍逐帧处理；可用 `--display-fps 30` 提高预览刷新率。启动会预热常用滤波核和 CUDA 接口。

`--stats` 每秒报告实际处理的新帧 FPS、漏处理帧数、算法 P95 和帧龄；它们与录像文件头的 FPS 不同。输入为 30 FPS 时，新帧输出上限为 30 FPS；要实测超过 30 FPS，先确认摄像头支持并实际输出更高帧率。检查性能时应开启与实用场景相同的录像、预览配置。

`--quiet` 仅隐藏逐帧终端 JSON，UDP 和文件日志仍正常输出。

`--listen` 端口收 JSON：

| 消息 | 含义 |
| --- | --- |
| `{"click": [x, y]}` | 点击，原图像素 |
| `{"click_norm": [u, v]}` | 点击，0～1 归一化（G30 画面，已去掉显示黑边） |
| `{"height": 85.2}` | 可靠离地高度（m），用来缩小半径搜索范围；`null` 表示没有 |
| `{"reset": true}` | 重置 |

每帧输出一条 JSON：

| 字段 | 含义 |
| --- | --- |
| `state` | `DETECTED` 本帧测到 / `PREDICTED` 短时预测 / `LOST` 没有 |
| `center_px` | 圆心，原图像素（左上原点，x 右 y 下）；`LOST` 时为 `null` |
| `bearing` | 去畸变后的归一化坐标 (x/z, y/z)，结合姿态和高度可换算地面偏移 |
| `radius_px`、`ellipse` | 半径；斜视时 `ellipse` 给 (cx, cy, 半长轴, 半短轴, 角度°) |
| `sectors`、`rms_px` | 有效扇区数（满 16）、拟合残差，供外部判断质量 |
| `age_ms`、`frame_age_ms` | 距上次 `DETECTED` 的时间、这帧图像的年龄 |
| `target_id`、`ambiguous` | 目标编号（重新首锁会加 1）；两个候选分不清时为 `true` |
| `click_px`、`click_bearing` | 点击点在当前帧的位置和方向（光流搬运）；首锁后、未点击时为 `null` |
| `motion_ok` | 本帧地面光流是否有效 |

相机断流时每 200 ms 发一次 `LOST` 心跳（`phase` 为 `NO_FRAME`，字段和正常消息相同、值为空）。完整接口约定见 [docs/白色石灰圆环识别_技术方案.docx](docs/白色石灰圆环识别_技术方案.docx) 第四节。

相机标定用 ROS `camera_calibration` 的 yaml 格式（`--calib`）。没标定时按 90° 水平视场估算，`bearing` 只是近似值。

## 调参

全部参数在 [ringdet/params.py](ringdet/params.py)，注释里写了含义。不用改代码：

```bash
--set verify.lock_sectors=13 --set search.click_max_levels=2     # 单项覆盖
--params my.yaml                                                   # 只写要改的字段
```

```yaml
search:
  click_box: 800
  ed_gradient: 8
verify:
  lock_sectors: 14
```

常改的几项：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `search.click_box` | 600 | 点击框边长（px）。点击误差大就加大 |
| `search.click_r_min` / `search.click_tile` | 4 / 192 | 无高度输入时的点击搜索半径下限、点击搜索分块大小（px） |
| `search.click_focus_r_max` | 12 | 点击附近高频复查的小环半径上限（px），完整范围仍交替轮询 |
| `search.click_max_levels` | 1 | 点击框最多扩大几级；0 = 只在初始框里找 |
| `search.async_ed` | true | 后台生成区域 ED 候选；false 使用同步搜索 |
| `search.fft_cache_mb` | 64 | 核频谱缓存上限（MiB），避免高度变化导致无限增长 |
| `motion.backend` | auto | auto / cpu / cuda，auto 在 CUDA 不可用时回退 CPU |
| `motion.refresh_frames` | 8 | 定期刷新角点；数量或局部覆盖不足时提前刷新 |
| `track.frame_budget_ms` | 28 | 含光流和预处理的整帧软预算（ms） |
| `verify.backend` | auto | CPU 原生射线评分、圆心精修及等价样本缓存；python 使用原路径 |
| `verify.lock_sectors` | 14 | 首锁扇区下限（满 16）。漏检多可降到 13，误检会增加 |
| `track.confirm_hits` / `track.identity_init_hits` | 3 / 6 | 首锁取两者较大值，确认窗口至少为首次建档帧数加 2 |
| `track.coast_s` | 0.3 | 测不到时最多预测多久 |
| `track.stable_track_frames` / `track.stable_no_flow_s` | 6 / 0.12 | 稳定跟踪所需的可靠帧数 / 光流失效后的观察时间；观察期间输出 `LOST` |
| `track.appearance_coast_s` | 0.5 | 没有几何实测时，关键帧外观定位最多持续多久；0 关闭该预测 |
| `track.appearance_score` / `track.appearance_psr` | 0.85 / 2 | 关键帧 NCC 分与其响应显著性门限；另需峰值唯一性和参考复核。此 PSR 与在线滤波响应的 PSR 不同 |
| `track.reacq_box_min` | 240 | 小圆环的重捕搜索框最小边长（原图 px） |
| `track.template_score` | 0.65 | 局部外观匹配阈值；通过后仍须圆环几何复核 |
| `track.template_backend` | auto | auto / cpu / cuda，auto 对较大重捕图块优先使用可用的 CUDA 模板匹配 |
| `track.compact_gpu_matching` | true | 允许预热时按响应区域实测分配峰值筛选；false 保留响应图回传到 CPU 的路径 |
| `track.identity_score` | 0.80 | 重捕身份分；较弱结果须达到 `template_score - 0.15`，通过首锁几何门限并延长确认 |
| `track.identity_margin` | 0.06 | 两个重捕候选的身份分差不足时暂不接回 |
| `track.reacq_confirm_hits` / `track.reacq_weak_hits` | 3 / 5 | 重捕的连续确认帧数，分别用于身份明确和较弱的结果 |
| `track.template_update_hits` | 8 | 重捕后恢复身份参考更新所需的可靠复核次数 |
| `track.learn_rate` | 0.15 | 工作模板、形状和运动统计的渐进更新率；越大适应越快 |
| `track.online_learning` | true | 背景感知在线滤波；false 保留像素模板路径 |
| `track.online_min_radius` | 8 | 在线滤波启用的最小像素半径；更小的圆环仍走原像素路径 |
| `track.online_update_frames` | 3 | 在线滤波器两次成功更新之间的最小帧数 |
| `track.online_psr` | 8 | 在线候选与模型更新要求的响应峰显著性下限 |
| `track.lock_need_edcircles` | true | 首锁是否要求 EDCircles 支持。虚线环、断续环要关掉 |
| `verify.channels` | gray, white | 复核用的通道。只用灰度：`--set 'verify.channels=[gray]'` |
| `search.ed_local` | true | 候选级放大 EDCircles |

## 测试结果

这些图都是 AI 生成的，不是实拍，结果只能作参考，实机必须重新验证。

| 数据 | 结果 |
| --- | --- |
点击评估（`tools/eval_lock.py`）：每张图在环附近粗点一下（偏 86 px），看 12 帧内锁没锁上；另外在离环足够远的地方做"空点击"，框里没有环，锁上任何东西都算误锁。

| 数据 | 结果 |
| --- | --- |
| 桌面 18 张 100 m 俯视图（17 张有环，含黄沙、浅色水泥、浅灰碎石） | 锁上 16/17，误锁 0；空点击 25 次误锁 0；漏的是虚线环 |
| 同上，先按 JPEG 质量 85 重新压缩（接近 MJPEG） | 锁上 15/17，误锁 0；黄沙那张在 100 m 尺度锁不上，放大 1.3 倍（相当于约 77 m）后锁上 |
| 合成图 40 张（含白色实心圆、白线、半段弧、碎石；10 张无环） | 锁上 28/30，误锁 0；空点击 120 次误锁 0 |
| 实拍近照换 9 种地面、缩到 100 m 尺度（216 次有环 + 72 张无环） | 单帧可首锁 184/216，误检 0；r = 8 px 时黄沙 0/8、浅色水泥 1/8 |
| 合成下降视频 1080p（第 0 帧粗点，偏 72 px） | 第 4 帧（0.13 s）首锁；圆心误差 P50 0.08 px、P95 0.15 px；20 帧完全遮挡后，恢复可见 2 帧内重新测到 |
| 桌面 62 s 模拟降落视频（第 0 帧点击） | 0.2 s 首锁，一直跟到 51 s 环出画；之后输出 `LOST`，没有误锁 |

`search.budget_ms` 默认 40 ms，候选生成和复核共用预算，未复核的候选搬运到下一帧继续。单次 EDCircles、FFT 或几何复核不能中途打断，冷启动等情况仍可能超出预算。视频窗口按原视频帧率控制回放节奏，处理变慢时不加速追帧。Orin Nano 的实际耗时需在设备上测量。

测试图片和视频不在仓库里（AI 生成，放在本机 `~/桌面/无人机下视圆环图片`），真值在 `data/desktop_gt.json`。

复现：

```bash
python3 tools/eval_lock.py [--jpeg 85]                   # 点击后能否锁上真环、空点击会不会误锁
python3 tools/eval_gates.py --sets desktop,synth          # 候选来源 x 首锁门槛对比
python3 tools/ground_swap.py 近照.png --gt cx,cy,r        # 换地面测试
python3 tools/sim_descent.py --size 1920x1080 --r1 150   # 合成下降
```

## 已知限制

- **虚线环、断开的环**：EDCircles 连不成圆，默认不首锁。确实是虚线环就设 `track.lock_need_edcircles=false`，这时误检风险变大，要用实拍确认。
- **浅色地面上的小环**（浅色水泥、黄沙、强光下的干土）：石灰和地面对比很低，r ≤ 10 px 时换地面测试的召回不到一半；经过 JPEG 压缩后更差。首锁会推迟到飞机降低、环变大以后。
- **白度通道和 MJPEG**：测试图是 PNG，没有色度下采样。真实 MJPEG 的色度分辨率只有一半，1 px 细线上白度通道的增益会变小，要用实拍确认。
- **近地出画**：fx ≈ 960 时，离地约 1.8 m 环开始超出画面，之后输出 `LOST`，由外部模块接管。
- **两个一样的环**：点击点附近有两个都合格、距离又分不出时，报 `ambiguous`，不首锁。

## 目录

```
ringdet/            Python 包
  tracker.py        状态机：搜索 -> 确认 -> 跟踪 -> 预测 -> 重捕
  frame.py          一帧的灰度、白度通道
  search.py         候选：EDCircles 分格、匹配滤波分块、候选级放大 EDCircles
  verify.py         复核与测量：射线找峰、圆/椭圆拟合、门槛
  motion.py         地面光流
  edcircles.py      ED_Lib ctypes 绑定（退回 cv2.ximgproc）
  params.py         全部参数
  synth.py          合成测试数据
edcircles/          ED_Lib 的 C 接口
third_party/ED_Lib  ED_Lib 源码（MIT，改动见 ORIGIN.md）
tools/              测试、评估、机载运行脚本
tests/              冒烟测试
data/               测试图真值
```
