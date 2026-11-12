#!/usr/bin/env python3
"""机载实时运行：USB 相机 -> 跟踪器 -> 每帧结果（JSON）。

用法：
  python3 tools/run_camera.py [--dev /dev/video0] [--size 1920x1080] [--fps 30]
      [--calib cam.yaml] [--udp 192.168.1.10:9000] [--listen 9001]
      [--record raw.avi] [--log out.jsonl] [--show] [--quiet]

输出：每帧一条 JSON（字段见 README"输出"），--udp 发到指定地址，否则打印到 stdout；--log 同时写入文件。
输入：--listen 端口收 JSON：
  {"click": [x, y]}           原图像素
  {"click_norm": [u, v]}      归一化 0~1（G30 画面点击，已去掉显示黑边）
  {"height": 85.2}            可靠离地高度 m，null 表示没有
  {"reset": true}
--show 时鼠标左键 = 点击，r = 重置，q = 退出。
相机格式、曝光、对焦先用 v4l2-ctl 设好（见 README）。
"""
import argparse
import dataclasses
import json
import math
import os
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from ringdet import Params, Camera, RingTracker, Result  # noqa: E402
from ringdet.draw import draw_result  # noqa: E402


class AsyncVideoWriter(threading.Thread):
    """录像队列有界，编码和磁盘等待不阻塞视觉。"""

    def __init__(self, path, fps, size, copy_mjpeg=False):
        super().__init__(daemon=True, name='ring-record')
        self.frames = queue.Queue(maxsize=4)
        self.dropped = self.written = 0
        self.error = None
        self.closed = False
        self.proc = self.writer = None
        ffmpeg = shutil.which('ffmpeg') if copy_mjpeg else None
        self.copy_mjpeg = ffmpeg is not None
        if self.copy_mjpeg:
            self.proc = subprocess.Popen([ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
                '-f', 'mjpeg', '-framerate', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'copy',
                '-r', str(fps), '-f', 'avi', path], stdin=subprocess.PIPE, bufsize=0)
        else:
            self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
            if not self.writer.isOpened():
                raise RuntimeError(f'无法创建录像：{path}')
            self.writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 95)
        self.start()

    def submit(self, frame):
        if self.closed or self.error:
            return
        while True:
            try:
                self.frames.put_nowait(frame)
                return
            except queue.Full:
                try:
                    self.frames.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass

    def run(self):
        try:
            while True:
                frame = self.frames.get()
                if frame is None:
                    break
                if self.proc is not None:
                    if not isinstance(frame, bytes):
                        ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        if not ok:
                            raise RuntimeError('JPEG 编码失败')
                        frame = jpeg.tobytes()
                    data = memoryview(frame)
                    while data:
                        written = self.proc.stdin.write(data)
                        if not written:
                            raise RuntimeError('录像管道已关闭')
                        data = data[written:]
                else:
                    self.writer.write(frame)
                self.written += 1
        except (OSError, RuntimeError, cv2.error) as e:
            self.error = str(e)
            print(f'[录像停止] {e}', file=sys.stderr)
        finally:
            if self.writer is not None:
                self.writer.release()
            if self.proc is not None:
                try:
                    self.proc.stdin.close()
                except OSError:
                    pass
                try:
                    rc = self.proc.wait(timeout=5)
                    if rc:
                        self.error = self.error or f'ffmpeg 退出码 {rc}'
                        print(f'[录像停止] {self.error}', file=sys.stderr)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait()

    def close(self):
        if self.closed:
            return
        self.submit(None)
        self.closed = True
        self.join(timeout=5)
        if self.is_alive() and self.proc is not None:
            self.proc.terminate()
            self.join(timeout=2)
        if self.is_alive():
            print('[录像] 写入线程未在截止时间内结束', file=sys.stderr)
        print(f'录像写入 {self.written} 帧，队列丢弃 {self.dropped} 帧', file=sys.stderr)


class LatestFrame(threading.Thread):
    """只解码主循环能接收的帧，录像独立接收相机原始帧。"""

    def __init__(self, cap, recorder=None, always_retrieve=False):
        super().__init__(daemon=True)
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None
        self.stamp = 0.0
        self.seq = 0
        self.taken_seq = 0
        self.alive = True
        self.recorder = recorder
        self.always_retrieve = always_retrieve   # 硬件解码通路：免去软件 JPEG 解码，每帧都取最新
        self._record_warned = False

    def run(self):
        captured = 0
        while self.alive:
            ok = self.cap.grab()
            t = time.monotonic()
            if not ok:
                time.sleep(0.005)
                continue
            captured += 1
            with self.lock:
                wanted = self.taken_seq == self.seq
            if not wanted and self.recorder is None and not self.always_retrieve:
                continue
            ok, f = self.cap.retrieve()
            if not ok or f is None:
                continue
            if f.ndim == 3 and f.shape[2] == 4:
                f = cv2.cvtColor(f, cv2.COLOR_BGRA2BGR)
            if f.ndim == 2 and f.shape[0] == 1:
                packet = f
                if self.recorder is not None:
                    self.recorder.submit(packet.tobytes())
                if not wanted:
                    continue
                try:
                    f = cv2.imdecode(packet, cv2.IMREAD_COLOR)
                except cv2.error:
                    f = None
                if f is None:
                    self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
                    print('[相机] 原始 MJPEG 解码失败，回退到采集后端解码', file=sys.stderr)
                    continue
            elif self.recorder is not None:
                if self.recorder.copy_mjpeg and not self._record_warned:
                    print('[录像] 采集后端返回 BGR，使用后台 JPEG 编码', file=sys.stderr)
                    self._record_warned = True
                self.recorder.submit(f)
            if wanted or self.always_retrieve:
                with self.lock:
                    self.frame, self.stamp, self.seq = f, t, captured

    def get(self):
        with self.lock:
            self.taken_seq = self.seq
            return self.frame, self.stamp, self.seq


def open_camera(dev, W, H, fps, capture='auto', pixel_format='MJPG'):
    """打开相机。auto 且是 Jetson + MJPG 时走 NVDEC 硬解，JPEG 解码不占 CPU；
    返回 (cap, 是否 GStreamer 通路)。"""
    plugin = Path('/usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvvideo4linux2.so')
    if capture == 'auto' and pixel_format == 'MJPG' and plugin.is_file():
        device = str(dev).replace('\\', '\\\\').replace('"', '\\"')
        pipeline = (f'v4l2src device="{device}" io-mode=2 ! '
                    f'image/jpeg,width={W},height={H},framerate={fps}/1 ! '
                    'jpegparse ! nvv4l2decoder mjpeg=true ! nvvidconv ! '
                    'video/x-raw,format=BGRx ! '
                    'appsink max-buffers=1 drop=true sync=false')
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print(f'相机采集：{W}x{H} MJPG {fps} fps，NVIDIA NVDEC 硬解，只保留最新帧', file=sys.stderr)
            return cap, True
        cap.release()
        print('硬件采集通路不可用，回退 V4L2', file=sys.stderr)
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*pixel_format))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, fps)
    # 只留 1 个缓冲时，取帧/还帧之间驱动放不下下一帧，会隔一帧丢一帧（实测只剩 15 fps）。
    # 2 个缓冲即可恢复 29.9 fps，代价只是最多多 1 帧延迟。
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    return cap, False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dev', default='/dev/video0')
    ap.add_argument('--size', default='1920x1080')
    ap.add_argument('--fps', type=float, default=30)
    ap.add_argument('--calib')
    ap.add_argument('--fx', type=float)
    ap.add_argument('--udp')
    ap.add_argument('--listen', type=int)
    ap.add_argument('--record')
    ap.add_argument('--record-mode', choices=['auto', 'encode'], default='auto',
                    help='auto 优先直存相机 MJPEG，encode 使用后台软件编码')
    ap.add_argument('--capture', choices=['auto', 'v4l2'], default='auto',
                    help='auto 优先 NVIDIA MJPEG 硬解；v4l2 为兼容路径')
    ap.add_argument('--camera-format', choices=['MJPG', 'YUYV'], default='MJPG', help='USB 摄像头传输格式')
    ap.add_argument('--threads', type=int, default=4, help='OpenCV CPU 线程数')
    ap.add_argument('--display-fps', type=float, default=15, help='预览刷新率，识别仍逐帧运行')
    ap.add_argument('--log', help='每帧结果同时写入这个 JSONL 文件')
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--stats', action='store_true', help='每秒打印采集帧率、实际处理帧率、漏处理帧数及算法 P95')
    ap.add_argument('--stale-ms', type=float, default=200, help='超过这么久没有新帧就发 LOST 心跳')
    ap.add_argument('--params')
    ap.add_argument('--set', action='append', default=[])
    a = ap.parse_args()
    if a.threads < 1 or a.display_fps <= 0:
        ap.error('threads 和 display-fps 必须大于 0')
    cv2.setNumThreads(a.threads)

    W, H = (int(v) for v in a.size.split('x'))
    cap, hardware = open_camera(a.dev, W, H, a.fps, a.capture, a.camera_format)
    if not cap.isOpened():
        sys.exit(f'打不开相机 {a.dev}')
    if hardware:
        rw, rh, fourcc, negotiated_fps = W, H, 'MJPG', float(a.fps)
    else:
        rw, rh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, 'little').decode(errors='replace')
        negotiated_fps = cap.get(cv2.CAP_PROP_FPS)
    source_fps = negotiated_fps if negotiated_fps > 0 else a.fps
    print(f'相机：{rw}x{rh} {fourcc} {negotiated_fps:g} fps（请求 {a.fps:g}）', file=sys.stderr)
    if (rw, rh) != (W, H):
        print(f'[警告] 实际分辨率 {rw}x{rh} 与要求 {W}x{H} 不同', file=sys.stderr)
    P = Params.load(a.params, a.set)
    cam = Camera.from_yaml(a.calib, (rw, rh)) if a.calib else Camera.default(rw, rh, a.fx)
    if not cam.calibrated:
        print('[警告] 未标定：按 90° 水平视场估算焦距，bearing 只是近似', file=sys.stderr)
    trk = RingTracker(P, cam)

    out_sock = out_addr = None
    if a.udp:
        host, port = a.udp.rsplit(':', 1)
        out_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        out_addr = (host, int(port))
    in_sock = None
    if a.listen:
        in_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        in_sock.bind(('0.0.0.0', a.listen))
        in_sock.setblocking(False)

    trk.warmup()
    print(f'运动估计：{trk.motion.backend}，CPU 线程数 {cv2.getNumThreads()}', file=sys.stderr)
    if trk.motion.backend_error:
        print(f'[CUDA] {trk.motion.backend_error}', file=sys.stderr)
    print(f'重捕匹配：{trk.template_backend}，峰值筛选：{trk.template_peak_backend}', file=sys.stderr)
    if trk.template_peak_error:
        print(f'[GPU 峰值筛选回退 CPU] {trk.template_peak_error}', file=sys.stderr)
    if trk._gpu_peak_routes:
        print('峰值筛选分配：' + '，'.join(f'{size}={"cuda" if trk._gpu_peak_routes.get(i, False) else "cpu"}'
              for i, size in enumerate(('小区域', '中区域', '大区域'))), file=sys.stderr)
    print('在线滤波：' + (f'CPU 背景感知相关滤波，半径至少 {P.track.online_min_radius:g} px'
                          if P.track.online_learning else '关闭，使用像素模板'), file=sys.stderr)
    if trk.template_backend_error:
        print(f'[CUDA 匹配] {trk.template_backend_error}', file=sys.stderr)

    log_f = open(a.log, 'a', encoding='utf-8', buffering=1) if a.log else None

    def emit(d):
        s = json.dumps(d, ensure_ascii=False)
        if log_f:
            log_f.write(s + '\n')
        if out_sock:
            out_sock.sendto(s.encode(), out_addr)
        elif not a.quiet:
            print(s, flush=True)

    def handle(msg):
        if 'click' in msg:
            trk.set_click(*msg['click'])
        if 'click_norm' in msg:
            u, v = msg['click_norm']
            trk.set_click(u * rw - 0.5, v * rh - 0.5)
        if 'height' in msg:
            trk.set_height(msg['height'])
        if msg.get('reset'):
            trk.reset()

    writer = None
    if a.record:
        copy_mjpeg = (not hardware and a.record_mode == 'auto' and fourcc == 'MJPG'
                      and shutil.which('ffmpeg') is not None
                      and cap.set(cv2.CAP_PROP_CONVERT_RGB, 0))
        writer = AsyncVideoWriter(a.record, source_fps, (rw, rh), copy_mjpeg)
        print('录像：' + ('直存 MJPEG' if writer.copy_mjpeg else '后台 MJPEG 编码'), file=sys.stderr)
    disp_scale = min(1.0, 1280 / rw)
    clicks = []
    if a.show:
        cv2.namedWindow('ringdet')
        cv2.setMouseCallback('ringdet', lambda ev, x, y, *_: clicks.append((x / disp_scale, y / disp_scale))
                             if ev == cv2.EVENT_LBUTTONDOWN else None)

    grab = LatestFrame(cap, writer, always_retrieve=hardware)
    grab.start()
    last_seq = 0
    last_new = time.monotonic()
    last_display = 0.0
    last_backend = trk.motion.backend
    stats_start, stats_count, stats_skipped, stats_ms = time.monotonic(), 0, 0, []
    measured_fps = 0.0
    capture_fps = 0.0
    last_search_error = last_match_error = ''
    try:
        while True:
            if in_sock:
                while True:
                    try:
                        data, _ = in_sock.recvfrom(4096)
                        handle(json.loads(data.decode()))
                    except BlockingIOError:
                        break
                    except (ValueError, KeyError, TypeError) as e:
                        print(f'[输入忽略] {e}', file=sys.stderr)
            for c in clicks:
                trk.set_click(*c)
            clicks.clear()
            frame, stamp, seq = grab.get()
            now = time.monotonic()
            if frame is None or seq == last_seq:
                if now - last_new > a.stale_ms / 1000:
                    # 心跳和正常消息字段相同，只是全部为空
                    d = {f.name: None for f in dataclasses.fields(Result) if f.name != 'debug'}
                    d.update(state='LOST', phase='NO_FRAME', frame_id=last_seq, stamp=now,
                             target_id=trk.target_id, sectors=0, ambiguous=False, motion_ok=False,
                             frame_age_ms=(now - stamp) * 1000 if frame is not None else None)
                    emit(d)
                    last_new = now
                time.sleep(0.002)
                continue
            stats_skipped += max(0, seq - last_seq - 1)
            last_seq, last_new = seq, now
            res = trk.process(frame, stamp, seq)
            if trk.motion.backend != last_backend:
                print(f'运动估计回退到 {trk.motion.backend}：{trk.motion.backend_error}', file=sys.stderr)
                last_backend = trk.motion.backend
            if res.debug.get('ed_error') and res.debug['ed_error'] != last_search_error:
                last_search_error = res.debug['ed_error']
                print(f'[区域搜索] {last_search_error}', file=sys.stderr)
            if trk.template_backend_error and trk.template_backend_error != last_match_error:
                last_match_error = trk.template_backend_error
                print(f'[CUDA 匹配回退] {last_match_error}', file=sys.stderr)
            d = res.to_dict()
            d['frame_age_ms'] = (time.monotonic() - stamp) * 1000
            d['ms'] = res.debug.get('ms')
            d['camera_fps'] = capture_fps
            emit(d)
            stats_count += 1
            stats_ms.append(res.debug['ms'])
            elapsed = time.monotonic() - stats_start
            if elapsed >= 1:
                measured_fps = stats_count / elapsed
                capture_fps = (stats_count + stats_skipped) / elapsed
                if a.stats:
                    ordered = sorted(stats_ms)
                    p95 = ordered[max(0, math.ceil(.95 * len(ordered)) - 1)]
                    print(f'[性能] 采集 {capture_fps:.1f} FPS，实际处理 {measured_fps:.1f} FPS，漏处理 {stats_skipped} 帧，'
                          f'算法 P95 {p95:.1f} ms，帧龄 {d["frame_age_ms"]:.1f} ms', file=sys.stderr)
                stats_start, stats_count, stats_skipped, stats_ms = time.monotonic(), 0, 0, []
            res.debug['fps'] = measured_fps
            res.debug['camera_fps'] = capture_fps
            res.debug['frame_age_ms'] = d['frame_age_ms']
            if a.show:
                display_time = time.monotonic()
                if display_time - last_display >= 1 / a.display_fps:
                    cv2.imshow('ringdet', draw_result(frame, res, scale=disp_scale))
                    last_display = display_time
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    break
                if k == ord('r'):
                    trk.reset()
    except KeyboardInterrupt:
        pass
    finally:
        grab.alive = False
        grab.join(timeout=1)
        cap.release()
        if writer is not None:
            writer.close()
        trk.close()
        if log_f:
            log_f.close()


if __name__ == '__main__':
    main()
