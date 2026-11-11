#!/usr/bin/env python3
"""按实际用法评估首锁：点击 -> 连续多帧 -> 是否锁上真环 / 是否误锁。

每张测试图做一次"真环点击"（在环附近点，偏离 (dx, dy)），无环图和有环图都再做几次"空点击"
（点在离真环足够远的地方，框里没有环）。同一张静态图每帧加不同的噪声，当作多帧。

  命中 = 真环点击后 n 帧内 DETECTED 且圆心离真值 < 0.5r
  误锁 = 任何点击后锁到了真环以外的位置（空点击锁上任何东西都算）

--jpeg Q：先按 JPEG 质量 Q（4:2:0 色度下采样，接近 USB 相机的 MJPEG）重新压缩一遍再测。

用法：python3 tools/eval_lock.py [--frames 12] [--set ...] [--sets desktop,synth] [--synth-n 40] [--jpeg 85] [-v]
"""
import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from ringdet import Params, Camera, RingTracker, synth  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DESK = os.path.expanduser('~/桌面/无人机下视圆环图片')
OFFSETS = [(70, -50)]                       # 真环点击：粗点偏离
EMPTY_FRAC = [(fx, fy) for fy in (0.2, 0.5, 0.8) for fx in (0.15, 0.38, 0.62, 0.85)]   # 空点击位置（相对宽高）


def load_params(sets):
    return Params.load(None, sets)


def run_click(img, click, P, n, seed):
    """返回 (锁上的圆心 or None, 帧号)。"""
    H, W = img.shape[:2]
    trk = RingTracker(P, Camera.default(W, H))
    trk.warmup()
    trk.set_click(*click)
    rng = np.random.default_rng(seed)
    base = img.astype(np.int16)
    for i in range(n):
        f = np.clip(base + rng.integers(-2, 3, base.shape), 0, 255).astype(np.uint8)
        res = trk.process(f, i / 30.0, i)
        if res.state == 'DETECTED':
            trk.close()
            return res.center_px, i
    trk.close()
    return None, n


def job(item):
    cv2.setNumThreads(1)
    kind, key, img_spec, gt, sets, n, jpeg = item
    P = load_params(sets)
    if kind == 'desk':
        img = cv2.imread(os.path.join(DESK, img_spec), cv2.IMREAD_COLOR)
    else:
        rng = np.random.default_rng(img_spec)
        g, g_gt = synth.scene(rng, size=(960, 540), ring=(img_spec % 4 != 3))
        img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        gt = None if g_gt is None else [g_gt['cx'], g_gt['cy'], g_gt['r']]
    if jpeg:
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg)])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    H, W = img.shape[:2]
    out = []
    if gt is not None:
        for dx, dy in OFFSETS:
            c, i = run_click(img, (gt[0] + dx, gt[1] + dy), P, n, 1)
            hit = c is not None and math.hypot(c[0] - gt[0], c[1] - gt[1]) < 0.5 * gt[2]
            out.append(dict(key=key, kind='ring', hit=hit, false=c is not None and not hit, frame=i, c=c))
    box = P.search.click_box * (2 ** P.search.click_max_levels)
    for fx, fy in EMPTY_FRAC:
        click = (fx * W, fy * H)
        if gt is not None and math.hypot(click[0] - gt[0], click[1] - gt[1]) < box * 0.75 + gt[2]:
            continue                       # 框可能盖到真环，不算空点击
        c, i = run_click(img, click, P, n, 2)
        out.append(dict(key=key, kind='empty', hit=False, false=c is not None, frame=i, c=c))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=12)
    ap.add_argument('--sets', default='desktop,synth')
    ap.add_argument('--synth-n', type=int, default=40)
    ap.add_argument('--set', action='append', default=[])
    ap.add_argument('--jpeg', type=int, default=0)
    ap.add_argument('-v', action='store_true')
    a = ap.parse_args()
    items = []
    if 'desktop' in a.sets:
        with open(os.path.join(ROOT, 'data', 'desktop_gt.json'), encoding='utf-8') as f:
            gts = {k: v for k, v in json.load(f).items() if not k.startswith('_')}
        for k, gt in gts.items():
            if os.path.exists(os.path.join(DESK, k)):
                items.append(('desk', k, k, gt, a.set, a.frames, a.jpeg))
    if 'synth' in a.sets:
        for i in range(a.synth_n):
            items.append(('synth', f'synth{i:02d}', 500 + i, None, a.set, a.frames, a.jpeg))
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as ex:
        rows = [r for rs in ex.map(job, items) for r in rs]
    for group, pref in (('桌面图', ''), ('合成图', 'synth')):
        g = [r for r in rows if (r['key'].startswith('synth')) == (pref == 'synth')]
        if not g:
            continue
        ring = [r for r in g if r['kind'] == 'ring']
        empty = [r for r in g if r['kind'] == 'empty']
        fr = [r['frame'] for r in ring if r['hit']]
        print(f"{group}: 真环点击 命中 {sum(r['hit'] for r in ring)}/{len(ring)}，误锁 {sum(r['false'] for r in ring)}；"
              f"空点击 误锁 {sum(r['false'] for r in empty)}/{len(empty)}"
              + (f"；首锁帧 中位 {int(np.median(fr))}" if fr else ''))
        if a.v or group == '桌面图':
            for r in g:
                if r['kind'] == 'ring' and not r['hit'] or r['false']:
                    tag = '误锁' if r['false'] else '漏'
                    c = '' if r['c'] is None else f" @({r['c'][0]:.0f},{r['c'][1]:.0f})"
                    print(f"   {tag:3} {r['kind']:5} {r['key']}{c}")


if __name__ == '__main__':
    main()
