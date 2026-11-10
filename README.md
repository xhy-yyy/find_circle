# find_circle：白色石灰圆环识别与跟踪

无人机下视相机找地面的白色石灰圆环（直径约 2 m），输出圆心像素坐标、半径和状态。不用神经网络，不需要训练数据。

**上手：** 所有操作都用 `./run.sh`（`./run.sh help` 列出命令），步骤见 [docs/使用教程.md](docs/使用教程.md)。

## 使用流程

1. 100 m 高空往往看不清环。操作员在画面上粗点一下环的大致位置。
2. 程序只在点击点周围的框里找环。飞机下降时，框跟着地面光流移动、随下降放大，3 s 没找到就扩大一级（默认最多扩一级，不扩到全图）。
3. 找到后要连续确认：最近 5 帧里至少 3 帧通过，当前帧通过，并且有 EDCircles 支持（全图检出，或在候选周围放大后检出），才首锁并输出 `DETECTED`。
4. 首锁后每帧在预测位置重新测量。短时测不到（遮挡、模糊）输出 `PREDICTED`，最多 0.3 s；超时就输出 `LOST`，先在最后位置附近重捕。3 s 还没找回，就在全图找半径不小于最后半径一半的环，近地出画后不会锁到地上的小圈。

没点击时不搜索，一直输出 `LOST`。这是为了少出误识别，可以用 `--set search.click_only=false` 改成全图搜索。

## 算法

| 步骤 | 做法 |
| --- | --- |
| 找候选 | EDCircles（[ED_Lib](https://github.com/CihanTopal/ED_Lib)，C++，MIT）为主，灰度、白度两个通道轮流跑；多尺度环带匹配滤波补 EDCircles 漏掉的小环、断续环 |
| 复核 | 64～128 条射线找"两侧都比它暗"的亮峰，RANSAC 拟圆，圆不够时拟椭圆（斜视）；看 16 个扇区的覆盖、最大缺口、内点率、残差。先在灰度上复核，不通过再在白度通道上复核 |
| EDCircles 支持 | 全图检出；或在复核通过的候选周围取一小块、放大到半径约 24 px 后再跑 EDCircles，圆心半径吻合就算 |
| 首锁 | 扇区 ≥ 14/16、缺口 ≤ 2、内点 ≥ 85%；5 帧中 3 帧通过；有 EDCircles 支持 |
| 帧间预测 | 金字塔 LK 光流 + RANSAC 相似变换，搬运点击框、候选和目标 |

白度通道是 min(B, G, R)。石灰三个颜色分量都高，取最小值后几乎不变暗；黄沙、红土、草地至少有一个分量低，取最小值后明显变暗，环更突出。黄沙测试图上，环的对比从灰度的 6.4 提高到 11.0。

候选级放大：100 m 时环半径只有 8 px 左右，边缘段太短，全图跑 EDCircles 连不成圆；放大候选周围一小块后边缘变长，就能连起来。只放大复核已经通过的候选，耗时很少。

单独用 EDCircles 时，草丛、长椭圆土块、暗斑、枯枝圈也会被当成圆；单独用匹配滤波时，每张图会多出 5～46 个纹理小圈。两者都要过复核，首锁还要求 EDCircles 支持，这两类误检都去掉了。

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
# 先设好相机：MJPEG、关自动对焦、限曝光（拖影是高空细环的头号敌人）
v4l2-ctl -d /dev/video0 --list-formats-ext
v4l2-ctl -d /dev/video0 -c focus_automatic_continuous=0 -c auto_exposure=1 -c exposure_time_absolute=20

python3 tools/run_camera.py --dev /dev/video0 --size 1920x1080 --calib cam.yaml \
    --udp 地面站IP:9000 --listen 9001 --record raw.avi
```

控件名称各相机不同，以 `v4l2-ctl -l` 为准。`exposure_time_absolute` 的单位是 100 µs，20 即 2 ms。

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
| `search.click_max_levels` | 1 | 点击框最多扩大几级；0 = 只在初始框里找 |
| `verify.lock_sectors` | 14 | 首锁扇区下限（满 16）。漏检多可降到 13，误检会增加 |
| `track.confirm_hits` | 3 | 5 帧里几帧通过才首锁 |
| `track.coast_s` | 0.3 | 测不到时最多预测多久 |
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

耗时（x86 笔记本，1080p）：跟踪中每帧 P50 约 11 ms；首锁前搜索中 P50 约 97 ms、P95 约 112 ms。`search.budget_ms` 默认 40 ms，但每帧至少各跑一块 EDCircles 和匹配滤波，实际会超。Orin Nano 上要实测。

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
