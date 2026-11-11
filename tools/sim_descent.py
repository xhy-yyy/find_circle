#!/usr/bin/env python3
"""合成下降视频跑完整跟踪器：首锁帧、每帧状态、圆心误差、遮挡和出画时的行为、耗时。

用法：python3 tools/sim_descent.py [--frames 120] [--r0 10] [--r1 80] [--save out.mp4]
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from ringdet import Params, Camera, RingTracker, synth  # noqa: E402
from ringdet.draw import draw_result  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=120)
    ap.add_argument('--r0', type=float, default=10.0)
    ap.add_argument('--r1', type=float, default=80.0)
    ap.add_argument('--contrast', type=float, default=35.0)
    ap.add_argument('--size', default='960x540')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--fps', type=float, default=30.0)
    ap.add_argument('--no-click', action='store_true', help='不点击（默认第 0 帧在真值附近 (+60,-40) px 处粗点一下）')
    ap.add_argument('--save', help='保存带标注的视频')
    a = ap.parse_args()
    size = tuple(int(v) for v in a.size.split('x'))
    rng = np.random.default_rng(a.seed)
    n = a.frames
    # 4 帧大面积遮挡（短于 COAST 上限，应保持 PREDICTED）；20 帧完全看不见（超过上限，应 LOST 后重捕）
    occ = set(range(int(n * 0.35), int(n * 0.35) + 4))
    hide = set(range(int(n * 0.55), int(n * 0.55) + 20))
    frames, gts = synth.descent(rng, n=n, size=size, r0=a.r0, r1=a.r1, contrast=a.contrast,
                                occlude=occ, hide=hide, exit_frames=int(n * 0.1))
    P = Params()
    trk = RingTracker(P, Camera.default(*size))
    trk.warmup()
    writer = None
    if a.save:
        writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*'mp4v'), a.fps, size)
    first = None
    errs, states, ms = [], [], []
    for i, (f, gt) in enumerate(zip(frames, gts)):
        if not a.no_click and i == 0:
            trk.set_click(gt[0] + 60, gt[1] - 40)
        t = time.perf_counter()
        res = trk.process(f, i / a.fps, i)
        ms.append((time.perf_counter() - t) * 1000)
        states.append(res.state[0])
        tag = ''
        if res.center_px is not None:
            e = math.hypot(res.center_px[0] - gt[0], res.center_px[1] - gt[1])
            if res.state == 'DETECTED':
                errs.append(e)
                if first is None:
                    first = i
            if e > 0.5 * gt[2]:
                tag = f'  <-- 偏离真值 {e:.1f} px'
        if tag or i in occ or i == first or (not gt[3] and res.state != 'LOST') \
                or (i in hide and res.state != 'LOST') or (i - 1 in hide and i not in hide):
            ev = '遮挡240°' if i in occ else ('看不见' if i in hide else '')
            print(f'帧 {i:3d} gt r={gt[2]:5.1f} 在画内={gt[3]!s:5} | {res.state:9} {res.phase:9} '
                  f'{ev:6} {ms[-1]:5.0f} ms{tag}')
        if writer is not None:
            writer.write(draw_result(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR), res))
    if writer is not None:
        writer.release()
    trk.close()
    print('\n状态序列（D=DETECTED P=PREDICTED L=LOST）:')
    for k in range(0, n, 60):
        print(f'  {k:3d}: ' + ''.join(states[k:k + 60]))
    inside = [i for i, g in enumerate(gts) if g[3] and i not in hide]
    back = next((i for i in range(max(hide) + 1, n) if states[i] == 'D'), None)
    print(f'\n完全遮挡 {len(hide)} 帧（{min(hide)}~{max(hide)}）后，第 {back} 帧重新 DETECTED'
          + ('' if back is None else f'（恢复可见后 {back - max(hide) - 1} 帧）'))
    det_in = sum(1 for i in inside if states[i] == 'D' and (first is not None and i >= first))
    print(f'\n首锁：第 {first} 帧（{"无" if first is None else f"{first / a.fps:.2f} s"}）')
    print(f'首锁后环完整在画内的帧中 DETECTED 占比：{det_in}/{sum(1 for i in inside if first is not None and i >= first)}')
    if errs:
        print(f'DETECTED 圆心误差：P50 {np.median(errs):.2f} px，P95 {np.percentile(errs, 95):.2f} px，最大 {max(errs):.2f} px')
    ms = np.array(ms)
    pre = ms[:first] if first else ms
    post = ms[first:] if first is not None else np.array([])
    print(f'耗时：首锁前 P50 {np.median(pre):.0f} ms / P95 {np.percentile(pre, 95):.0f} ms；'
          + (f'跟踪中 P50 {np.median(post):.0f} ms / P95 {np.percentile(post, 95):.0f} ms' if len(post) else ''))


if __name__ == '__main__':
    main()
