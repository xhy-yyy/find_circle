#!/usr/bin/env python3
"""单帧检测评估：候选来源（EDCircles / 匹配滤波 / 两者）x 首锁门槛，一次测量、事后套门槛。

数据：
  desktop  data/desktop_gt.json 里的桌面测试图（AI 生成，按真值统计）
  synth    合成图（ringdet.synth.scene，每 4 张 1 张无环；含白色实心圆、白线、半段弧、亮斑、碎石）
  swap     实拍石灰环近照的覆盖率图贴到 9 种地面上（tools/ground_swap.py 的做法），需要 --swap-image

用法：python3 tools/eval_gates.py [--sets desktop,synth] [--synth-n 60] [--swap-image 近照.png]
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
from ringdet import Params, Camera, synth  # noqa: E402
from ringdet.search import RingSearcher, build_pyramid, merge_candidates, plan_ed, run_ed  # noqa: E402
from ringdet.tracker import frame_noise  # noqa: E402
from ringdet.verify import measure  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DESK = [os.path.expanduser('~/桌面/无人机下视圆环图片/') + d for d in ('其他场景_小圆轻模糊/', '多场景组图/100米俯视/')]

# 首锁门槛组：(扇区下限, 最大缺口, 内点率下限, rms 上限 (a, b) => max(a, b*r))
GATES = {
    'loose(旧)': (12, 3, 0.60, (0.7, 0.025)),
    'mid': (13, 3, 0.80, (0.5, 0.03)),
    'strict': (14, 2, 0.85, (0.4, 0.03)),
    'strict_rms06': (14, 2, 0.85, (0.6, 0.04)),
}


def relaxed(P):
    """测量用的宽松参数：门槛全放开，事后再套。"""
    import copy
    Q = copy.deepcopy(P)
    Q.verify.lock_sectors = 0
    Q.verify.lock_max_gap = 99
    Q.verify.min_inlier_ratio = 0.0
    Q.verify.max_rms = (1e9, 0.0)
    return Q


def gate_ok(m, c_r, c_x, c_y, g, V):
    sec, gap, inl, rms = g
    return (not math.isnan(m.r) and m.sectors >= sec and m.max_gap <= gap and m.inlier_ratio >= inl
            and m.rms <= max(rms[0], rms[1] * m.r) and abs(m.r - c_r) <= V.lock_radius_tol * c_r
            and math.hypot(m.cx - c_x, m.cy - c_y) <= V.lock_shift * c_r)


def measure_image(gray, P, r_range):
    """返回 [(来源, m, 候选)]，来源 'ed' / 'mf'。"""
    H, W = gray.shape
    cam = Camera.default(W, H)
    S = P.search
    g = gray.astype(np.float32)
    region = (0, 0, W, H)
    ed = []
    for j in plan_ed(region, (W, H), r_range[1], S.ed_cell):
        ed += run_ed(gray, j, *r_range, S)
    s = RingSearcher(P)
    jobs = s.plan(region, (W, H), *r_range)
    pyr = build_pyramid(g, max(j.level for j in jobs))
    mf = merge_candidates([c for j in jobs for c in s.run(pyr, j)], 200)
    noise = max(frame_noise(g), S.noise_floor)
    Q = relaxed(P)
    out = []
    for src, cands in (('ed', merge_candidates(ed)), ('mf', mf)):
        for c in cands:
            m = measure(g, cam, c.x, c.y, c.r, noise, Q.verify, mode='lock')
            out.append((src, m, c))
    return out


def summarize(rows, gt_list, r_range, V):
    """rows[i] = 第 i 张图的 measure_image 结果。返回 {(来源组合, 门槛): (命中, 有环数, 误检, 圆心误差列表)}"""
    res = {}
    for combo in ('ed', 'mf', 'ed+mf'):
        for gname, g in GATES.items():
            hits = rings = fp = 0
            errs = []
            for meas, gt in zip(rows, gt_list):
                passed = []
                for src, m, c in meas:
                    if src not in combo.split('+'):
                        continue
                    if gate_ok(m, c.r, c.x, c.y, g, V) and r_range[0] / 1.15 <= m.r <= r_range[1] * 1.15:
                        if not any(math.hypot(m.cx - q.cx, m.cy - q.cy) < 0.5 * m.r for q in passed):
                            passed.append(m)
                hit = None
                if gt is not None:
                    rings += 1
                    for m in passed:
                        e = math.hypot(m.cx - gt[0], m.cy - gt[1])
                        if e < 0.5 * gt[2] and abs(m.r - gt[2]) < 0.3 * gt[2]:
                            hit = m
                            errs.append(e)
                            break
                hits += hit is not None
                fp += len(passed) - (hit is not None)
            res[(combo, gname)] = (hits, rings, fp, errs)
    return res


def _desk_job(item):
    cv2.setNumThreads(1)
    path, gt, r_range = item
    P = Params()
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return measure_image(gray, P, r_range)


def _synth_job(i):
    cv2.setNumThreads(1)
    rng = np.random.default_rng(200 + 1000 * i)
    img, gt = synth.scene(rng, size=(960, 540), ring=(i % 4 != 3))
    P = Params()
    return measure_image(img, P, (8, 60)), (None if gt is None else (gt['cx'], gt['cy'], gt['r']))


def print_table(title, res, n_img):
    print(f'\n== {title}（{n_img} 张）==')
    print(f"{'来源':<8}" + ''.join(f'{g:>26}' for g in GATES))
    for combo in ('ed', 'mf', 'ed+mf'):
        line = f'{combo:<8}'
        for g in GATES:
            h, n, fp, errs = res[(combo, g)]
            e = f'{np.median(errs):.2f}' if errs else '-'
            line += f'{f"{h}/{n} 误检{fp} 误差{e}":>26}'
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', default='desktop,synth')
    ap.add_argument('--synth-n', type=int, default=60)
    a = ap.parse_args()
    V = Params().verify
    workers = max(1, (os.cpu_count() or 2) - 2)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        if 'desktop' in a.sets:
            with open(os.path.join(ROOT, 'data', 'desktop_gt.json'), encoding='utf-8') as f:
                gts = {k: v for k, v in json.load(f).items() if not k.startswith('_')}
            items = []
            for name, gt in gts.items():
                path = next((d + name for d in DESK if os.path.exists(d + name)), None)
                if path:
                    items.append((path, gt))
            for rr in ((5, 60), (6, 270)):
                rows = list(ex.map(_desk_job, [(p, g, rr) for p, g in items]))
                print_table(f'桌面测试图 半径 {rr[0]}~{rr[1]} px', summarize(rows, [g for _, g in items], rr, V), len(items))
        if 'synth' in a.sets:
            out = list(ex.map(_synth_job, range(a.synth_n)))
            print_table('合成图 960x540 半径 8~60 px', summarize([o[0] for o in out], [o[1] for o in out], (8, 60), V), a.synth_n)


if __name__ == '__main__':
    main()
