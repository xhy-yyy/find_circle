#!/usr/bin/env python3
"""机载实时运行：USB 相机 -> 跟踪器 -> 每帧结果（JSON）。

用法：
  python3 tools/run_camera.py [--dev /dev/video0] [--size 1920x1080] [--fps 30]
      [--calib cam.yaml] [--udp 192.168.1.10:9000] [--listen 9001]
      [--record raw.avi] [--show] [--quiet]

输出：每帧一条 JSON（字段见 README"输出"），--udp 发到指定地址，否则打印到 stdout。
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
import os
import socket
import sys
import threading
import time

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from ringdet import Params, Camera, RingTracker, Result  # noqa: E402
from ringdet.draw import draw_result  # noqa: E402


class LatestFrame(threading.Thread):
    """采集线程只留最新一帧，处理慢时丢旧帧而不是排队。"""

    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None
        self.stamp = 0.0
        self.seq = 0
        self.alive = True

    def run(self):
        while self.alive:
            ok, f = self.cap.read()
            t = time.monotonic()
            if not ok:
                time.sleep(0.005)
                continue
            with self.lock:
                self.frame, self.stamp, self.seq = f, t, self.seq + 1

    def get(self):
        with self.lock:
            return self.frame, self.stamp, self.seq


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
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--stale-ms', type=float, default=200, help='超过这么久没有新帧就发 LOST 心跳')
    ap.add_argument('--params')
    ap.add_argument('--set', action='append', default=[])
    a = ap.parse_args()

    W, H = (int(v) for v in a.size.split('x'))
    cap = cv2.VideoCapture(a.dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, a.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        sys.exit(f'打不开相机 {a.dev}')
    rw, rh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, 'little').decode(errors='replace')
    print(f'相机：{rw}x{rh} {fourcc} {cap.get(cv2.CAP_PROP_FPS):.0f} fps', file=sys.stderr)
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

    def emit(d):
        s = json.dumps(d, ensure_ascii=False)
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
        writer = cv2.VideoWriter(a.record, cv2.VideoWriter_fourcc(*'MJPG'), a.fps, (rw, rh))
        writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 95)
    disp_scale = min(1.0, 1280 / rw)
    clicks = []
    if a.show:
        cv2.namedWindow('ringdet')
        cv2.setMouseCallback('ringdet', lambda ev, x, y, *_: clicks.append((x / disp_scale, y / disp_scale))
                             if ev == cv2.EVENT_LBUTTONDOWN else None)

    grab = LatestFrame(cap)
    grab.start()
    last_seq = 0
    last_new = time.monotonic()
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
            last_seq, last_new = seq, now
            if writer is not None:
                writer.write(frame)
            res = trk.process(frame, stamp, seq)
            d = res.to_dict()
            d['frame_age_ms'] = (time.monotonic() - stamp) * 1000
            emit(d)
            if a.show:
                vis = draw_result(frame, res)
                cv2.imshow('ringdet', cv2.resize(vis, None, fx=disp_scale, fy=disp_scale) if disp_scale < 1 else vis)
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
            writer.release()


if __name__ == '__main__':
    main()
