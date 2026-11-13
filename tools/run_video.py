#!/usr/bin/env python3
"""视频回放跑完整跟踪器（多帧确认、跟踪、预测、重捕）。

用法：
  python3 tools/run_video.py video.mp4 [--out out/xxx_det.mp4 | --no-record] [--jsonl out/xxx.jsonl]
      [--click x,y[,帧号]]  第几帧点一下（默认第 0 帧）
      [--height 100 --fx 960 | --calib cam.yaml]
      [--start 秒] [--max-frames N] [--show]
--show 时：鼠标左键 = 点击，空格 = 暂停，r = 重置，q = 退出。
"""
import argparse
import collections
import json
import os
import sys
import time

import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from ringdet import Params, Camera, RingTracker  # noqa: E402
from ringdet.draw import draw_result  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('video')
    output = ap.add_mutually_exclusive_group()
    output.add_argument('--out', help='保存标注视频')
    output.add_argument('--no-record', action='store_true', help='跳过标注视频编码，可保留逐帧日志')
    ap.add_argument('--jsonl')
    ap.add_argument('--click')
    ap.add_argument('--height', type=float)
    ap.add_argument('--fx', type=float)
    ap.add_argument('--calib')
    ap.add_argument('--start', type=float, default=0.0)
    ap.add_argument('--max-frames', type=int, default=0)
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--params')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--debug', action='store_true', help='jsonl 里写入完整调试字段（rejected/identity/r_range 等）')
    ap.add_argument('--set', action='append', default=[])
    a = ap.parse_args()
    if a.threads < 1:
        ap.error('threads 必须大于 0')
    cv2.setNumThreads(a.threads)

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        sys.exit(f'打不开：{a.video}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if a.start > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, a.start * 1000)
    P = Params.load(a.params, a.set)
    cam = Camera.from_yaml(a.calib, (W, H)) if a.calib else Camera.default(W, H, a.fx)
    trk = RingTracker(P, cam)
    trk.warmup()
    if a.height:
        trk.set_height(a.height)
    click = None
    if a.click:
        v = [float(x) for x in a.click.split(',')]
        click = (v[0], v[1], int(v[2]) if len(v) > 2 else 0)

    name = os.path.splitext(os.path.basename(a.video))[0]
    out = None if a.no_record else a.out or os.path.join('out', f'{name}_det.mp4')
    writer = None
    if out:
        os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
        if not writer.isOpened():
            sys.exit(f'无法创建标注视频：{out}')
    if a.jsonl:
        os.makedirs(os.path.dirname(a.jsonl) or '.', exist_ok=True)
    jf = open(a.jsonl, 'w', encoding='utf-8') if a.jsonl else None

    disp_scale = min(1.0, 1280 / W)
    pending = []
    if a.show:
        cv2.namedWindow('ringdet')
        cv2.setMouseCallback('ringdet', lambda ev, x, y, *_: pending.append((x / disp_scale, y / disp_scale))
                             if ev == cv2.EVENT_LBUTTONDOWN else None)
    counts = collections.Counter()
    first = None
    ms = []
    i = 0
    paused = False
    frame = None
    next_display = time.perf_counter()
    stats_start, stats_count, measured_fps = next_display, 0, 0.0
    while True:
        if not paused or frame is None:
            ok, frame = cap.read()
            if not ok:
                break
            if click and i == click[2]:
                trk.set_click(click[0], click[1])
            for c in pending:
                trk.set_click(*c)
            pending.clear()
            stamp = a.start + i / fps
            res = trk.process(frame, stamp, i)
            stats_count += 1
            now = time.perf_counter()
            if now - stats_start >= 1:
                measured_fps = stats_count / (now - stats_start)
                stats_start, stats_count = now, 0
            res.debug['fps'] = measured_fps
            res.debug['source_fps'] = fps
            counts[res.state] += 1
            ms.append(res.debug['ms'])
            if res.state == 'DETECTED' and first is None:
                first = (i, stamp)
                print(f'首锁：第 {i} 帧（{stamp:.2f} s） c=({res.center_px[0]:.1f},{res.center_px[1]:.1f}) r={res.radius_px:.1f}')
            if writer is not None:
                writer.write(draw_result(frame, res))
            if a.show:
                vis = draw_result(frame, res, scale=disp_scale)
            if jf:
                jf.write(json.dumps(res.to_dict(debug=a.debug), ensure_ascii=False,
                                    default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)) + '\n')
            i += 1
            if a.max_frames and i >= a.max_frames:
                break
        if a.show:
            cv2.imshow('ringdet', vis)
            if paused:
                delay = 0
            else:
                now = time.perf_counter()
                next_display = max(next_display + 1 / fps, now)
                delay = max(1, int(round((next_display - now) * 1000)))
            k = cv2.waitKey(delay) & 0xFF
            if k == ord('q'):
                break
            if k == ord(' '):
                paused = not paused
                next_display = time.perf_counter()
                stats_start, stats_count, measured_fps = next_display, 0, 0.0
            if k == ord('r'):
                trk.reset()
    if writer is not None:
        writer.release()
    cap.release()
    trk.close()
    if jf:
        jf.close()
    n = sum(counts.values())
    print(f'{n} 帧：' + '  '.join(f'{k} {v}' for k, v in counts.most_common()))
    if ms:
        print(f'每帧耗时 P50 {np.median(ms):.0f} ms，P95 {np.percentile(ms, 95):.0f} ms')
    if out:
        print(f'标注视频：{out}')
    if a.jsonl:
        print(f'逐帧结果：{a.jsonl}')


if __name__ == '__main__':
    main()
