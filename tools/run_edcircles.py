#!/usr/bin/env python3
"""直接用 EDCircles 检测圆，看效果。默认用 ED_Lib C++（./build.sh 编译），--backend opencv 用 cv2.ximgproc 移植版。

用法：
  python3 tools/run_edcircles.py 图片或文件夹 [...] [--grad 20,12,8] [--anchor 0] [--sigma 1.0]
      [--minpath 10] [--validate] [--backend auto|edlib|opencv]
      [--rmin 5 --rmax 60] [--gt data/desktop_gt.json] [--out out/edcircles]
--grad 逗号分隔多个梯度阈值，每个各跑一遍（ED_Lib/OpenCV 默认 20）。
--validate 先用 EDPF 筛边缘段（ED_Lib EDCircles(Mat) 的默认行为），高空细环的弱边缘会被删掉。
--gt：{"文件名": [cx, cy, r]}，给了就统计 命中 / 圆心误差 / 其他检测数。
命中 = 有检测圆心离真值 < 0.5r 且半径差 < 30%。
输出：out/<名字>_g<阈值>.jpg（绿=命中，红=其他检测，黄十字=真值），sheet_g<阈值>.jpg 是各图真值附近的放大拼图。
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from ringdet import edcircles as edc  # noqa: E402

EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')


def list_images(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += sorted(f for f in glob.glob(os.path.join(p, '*')) if f.lower().endswith(EXTS))
        else:
            out.append(p)
    return out


def edcircles(gray, grad, anchor, sigma, minpath, validate, backend):
    """返回 [(cx, cy, a, b, angle_deg)]：圆 a == b == r；椭圆给两个半轴。"""
    d = edc.detect(gray, grad=grad, anchor=anchor, sigma=sigma, min_path=minpath, validate=validate, prefer=backend)
    return [(float(x), float(y), float(a), float(b), math.degrees(t)) for x, y, a, b, t, _ in d]


def mean_r(d):
    return (d[2] + d[3]) / 2


def match(dets, gt):
    """返回 (命中检测下标 或 None, 圆心误差)。多个都匹配时取最近的。"""
    best, err = None, None
    for i, d in enumerate(dets):
        e = math.hypot(d[0] - gt[0], d[1] - gt[1])
        if e < 0.5 * gt[2] and abs(mean_r(d) - gt[2]) < 0.3 * gt[2] and (err is None or e < err):
            best, err = i, e
    return best, err


def draw_det(img, d, color, th=1):
    cv2.ellipse(img, ((d[0], d[1]), (2 * d[2], 2 * d[3]), d[4]), color, th, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--grad', default='20,12,8')
    ap.add_argument('--anchor', type=int, default=0)
    ap.add_argument('--sigma', type=float, default=1.0)
    ap.add_argument('--minpath', type=int, default=10)
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--backend', default='auto', choices=['auto', 'edlib', 'opencv'])
    ap.add_argument('--rmin', type=float, default=5)
    ap.add_argument('--rmax', type=float, default=60)
    ap.add_argument('--gt')
    ap.add_argument('--out', default='out/edcircles')
    a = ap.parse_args()
    if not edc.backend(a.backend):
        sys.exit(f'EDCircles 不可用：{edc.describe()}')
    print(f'EDCircles 后端：{edc.describe() if a.backend == "auto" else a.backend}')
    files = list_images(a.paths)
    gts = {}
    if a.gt:
        with open(a.gt, encoding='utf-8') as f:
            gts = {k: v for k, v in json.load(f).items() if not k.startswith('_')}
    os.makedirs(a.out, exist_ok=True)
    grads = [int(g) for g in a.grad.split(',')]
    summary = {g: [] for g in grads}
    for g in grads:
        tiles = []
        print(f'\n===== EDCircles  GradientThreshold={g}  Anchor={a.anchor}  Sigma={a.sigma}  '
              f'MinPathLength={a.minpath}  EDPF验证={a.validate} =====')
        for path in files:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                print(f'[跳过] 读不了：{path}')
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            t = time.perf_counter()
            dets = edcircles(gray, g, a.anchor, a.sigma, a.minpath, a.validate, a.backend)
            ms = (time.perf_counter() - t) * 1000
            in_rng = [d for d in dets if a.rmin <= mean_r(d) <= a.rmax]
            name = os.path.basename(path)
            gt = gts.get(name)
            hit, err = match(in_rng, gt) if gt else (None, None)
            others = len(in_rng) - (hit is not None)
            line = f'{name[:24]:<26} 检测 {len(dets):3d}（半径 {a.rmin:g}~{a.rmax:g}: {len(in_rng):3d}）  {ms:5.0f} ms'
            if gt:
                if hit is not None:
                    d = in_rng[hit]
                    line += (f'  命中 r={mean_r(d):.1f}（真值 {gt[2]:.1f}） 圆心误差 {err:.2f} px'
                             f'{"  椭圆" if d[2] != d[3] else ""}  其他 {others}')
                else:
                    line += f'  未命中  其他 {others}'
                summary[g].append((hit is not None, others, err, ms))
            print(line)
            vis = img.copy()
            for i, d in enumerate(in_rng):
                draw_det(vis, d, (0, 220, 0) if i == hit else (0, 0, 255), 2 if i == hit else 1)
            if gt:
                x, y = int(round(gt[0])), int(round(gt[1]))
                cv2.drawMarker(vis, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 8, 1)
                R = int(max(4 * gt[2], 40))
                pad = cv2.copyMakeBorder(vis, R, R, R, R, cv2.BORDER_CONSTANT)
                crop = cv2.resize(pad[y:y + 2 * R, x:x + 2 * R], (240, 240), interpolation=cv2.INTER_NEAREST)
                cv2.putText(crop, f'{len(tiles)} {"HIT" if hit is not None else "MISS"}', (4, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0) if hit is not None else (0, 0, 255), 2)
                tiles.append(crop)
            cv2.imwrite(os.path.join(a.out, f'{os.path.splitext(name)[0]}_g{g}.jpg'), vis,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
        if tiles:
            while len(tiles) % 4:
                tiles.append(np.zeros_like(tiles[0]))
            sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
            cv2.imwrite(os.path.join(a.out, f'sheet_g{g}.jpg'), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if gts:
        print('\n汇总（半径范围内）：')
        for g, rows in summary.items():
            if not rows:
                continue
            hits = sum(r[0] for r in rows)
            errs = [r[2] for r in rows if r[0]]
            print(f'  阈值 {g:3d}: 命中 {hits}/{len(rows)}  其他检测 平均 {np.mean([r[1] for r in rows]):.1f}/张'
                  f'（最多 {max(r[1] for r in rows)}）  圆心误差 中位 '
                  f'{np.median(errs) if errs else float("nan"):.2f} px  耗时 {np.mean([r[3] for r in rows]):.0f} ms/张')
    print(f'\n标注图：{a.out}/')


if __name__ == '__main__':
    main()
