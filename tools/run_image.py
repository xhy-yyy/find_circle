#!/usr/bin/env python3
"""单张图片 / 文件夹测试：全图（或点击框内）搜索 + 单帧复核。

默认只列"可首锁"的结果：EDCircles 找到且通过复核（与跟踪器的首锁条件一致）。
只有匹配滤波通过的（大多是纹理小圈）跟踪器不会拿来首锁，加 --all 才列出来。
单张图没有多帧确认，实机还要再过 5 帧中 3 帧。

用法：
  python3 tools/run_image.py 图片或文件夹 [...] [--out out/]
      [--rmin 6 --rmax 200]                  半径范围（原图 px），默认 6 ~ 短边/4
      [--height 100 --fx 960 | --calib cam.yaml]   有高度时自动算半径范围
      [--click x,y]                          只搜点击点周围 --box 边长的框
      [--probe x,y,r]                        在指定位置直接复核，打印每项指标（排查漏检用）
      [--all]                                也列出只有匹配滤波通过的，并画出未通过的候选和原因
      [--set verify.k_sigma=2.0 ...] [--params p.yaml]
输出：out/<名字>_det.jpg（绿=通过，编号按可信度；灰=未通过；黄框=搜索区域；底部是通过项的放大图）
      out/results.json
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
from ringdet import Params, Camera, detect_frame  # noqa: E402
from ringdet.frame import Frame  # noqa: E402
from ringdet.tracker import measure_channels  # noqa: E402
from ringdet import draw  # noqa: E402

EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')


def list_images(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += sorted(f for f in glob.glob(os.path.join(p, '*')) if f.lower().endswith(EXTS))
        else:
            out.append(p)
    return out


def quality_key(m):
    # 有 EDCircles 支持的排前面（跟踪器只拿它们首锁）
    return (m.ed_ok, m.sectors, m.inlier_ratio, m.contrast)


def zoom(img, m, size=240):
    R = int(math.ceil(max(m.r * 1.6, 12)))
    x, y = int(round(m.cx)), int(round(m.cy))
    pad = cv2.copyMakeBorder(img, R, R, R, R, cv2.BORDER_CONSTANT)
    crop = pad[y:y + 2 * R + 1, x:x + 2 * R + 1]
    k = size / crop.shape[0]
    big = cv2.resize(crop, (size, size), interpolation=cv2.INTER_NEAREST)
    if big.ndim == 2:
        big = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
    ox, oy = x - R, y - R
    for px, py in (m.points if m.points is not None else []):
        cv2.circle(big, (int((px - ox + 0.5) * k), int((py - oy + 0.5) * k)), 1, draw.RED, -1)
    cv2.circle(big, (int((m.cx - ox + 0.5) * k), int((m.cy - oy + 0.5) * k)), int(m.r * k), draw.GREEN, 1, cv2.LINE_AA)
    return big


def describe(m):
    src = {'ed': 'EDCircles', 'mf': '匹配滤波'}.get(m.source, '')
    if m.source == 'mf' and m.ed_ok:
        src += '+局部EDCircles'
    if m.channel == 'white':
        src += '，白度通道'
    return (f'[{src}] c=({m.cx:.1f},{m.cy:.1f}) r={m.r:.1f} 扇区={m.sectors}/16 最大缺口={m.max_gap} '
            f'内点={m.inlier_ratio:.0%} rms={m.rms:.2f} 对比={m.contrast:.1f} 线宽≈{m.line_w:.1f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--out', default='out')
    ap.add_argument('--rmin', type=float)
    ap.add_argument('--rmax', type=float)
    ap.add_argument('--height', type=float, help='离地高度 m')
    ap.add_argument('--fx', type=float, help='焦距 px（未标定时用；默认按 90° 水平视场 = 宽/2）')
    ap.add_argument('--calib', help='ROS camera_info 格式的标定 yaml')
    ap.add_argument('--click', help='x,y')
    ap.add_argument('--box', type=float, help='点击搜索框边长 px，默认取参数 search.click_box')
    ap.add_argument('--probe', action='append', default=[], help='x,y,r，可多次')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--params')
    ap.add_argument('--set', action='append', default=[])
    a = ap.parse_args()

    P = Params.load(a.params, a.set)
    os.makedirs(a.out, exist_ok=True)
    files = list_images(a.paths)
    if not files:
        sys.exit('没有找到图片')
    results = []
    for path in files:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f'[跳过] 读不了：{path}')
            continue
        H, W = img.shape[:2]
        cam = Camera.from_yaml(a.calib, (W, H)) if a.calib else Camera.default(W, H, a.fx)
        if a.height:
            r0 = cam.fx * (P.ring.diameter_m / 2) / a.height
            r_range = (r0 * P.search.height_band[0], r0 * P.search.height_band[1])
        else:
            r_range = (a.rmin or 6.0, a.rmax or max(30.0, min(W, H) / 4))
        region = None
        if a.click:
            cx, cy = (float(v) for v in a.click.split(','))
            h = (a.box or P.search.click_box) / 2
            region = (max(0, cx - h), max(0, cy - h), min(W, cx + h), min(H, cy + h))
        t = time.perf_counter()
        res, noise = detect_frame(img, cam, P, region=region, r_range=r_range)
        dt = (time.perf_counter() - t) * 1000
        passed = sorted([m for _, m in res if m.ok and (a.all or m.ed_ok)], key=quality_key, reverse=True)
        if a.click:
            passed.sort(key=lambda m: math.hypot(m.cx - cx, m.cy - cy))
        name = os.path.splitext(os.path.basename(path))[0]
        print(f'\n{path}  {W}x{H}  半径范围 {r_range[0]:.1f}~{r_range[1]:.1f} px  噪声σ={noise:.1f}  '
              f'候选 {len(res)}  通过 {len(passed)}  {dt:.0f} ms')
        for i, m in enumerate(passed):
            print(f'  #{i + 1} {describe(m)}')

        vis = img.copy()
        if region is not None:
            cv2.rectangle(vis, (int(region[0]), int(region[1])), (int(region[2]) - 1, int(region[3]) - 1), draw.YELLOW, 1)
        if a.all:
            for c, m in res:
                if not m.ok:
                    draw.circle(vis, c.x, c.y, c.r, draw.GRAY)
                    draw.text(vis, m.reason, (int(c.x + c.r + 2), int(c.y)), draw.GRAY, 0.35)
        for i, m in enumerate(passed):
            draw.draw_measurement(vis, m, draw.GREEN)
            draw.text(vis, f'#{i + 1}', (int(m.cx + m.r + 4), int(m.cy - 4)), draw.GREEN, 0.6)
        probes = []
        fr = Frame(img, P.search.noise_floor)
        for pr in a.probe:
            x, y, r = (float(v) for v in pr.split(','))
            m = measure_channels(fr, cam, x, y, r, P.verify, 'lock')
            print(f'  [probe {x:.0f},{y:.0f},{r:.0f}] {"通过" if m.ok else "未通过：" + m.reason}  '
                  + (describe(m) if not math.isnan(m.r) else f'有效射线 {m.n_valid}/{m.n_inside}'))
            probes.append(m)
            draw.cross(vis, x, y, 6, draw.CYAN)
            if not math.isnan(m.r):
                draw.draw_measurement(vis, m, draw.CYAN)
        draw.text(vis, f'{name}  pass={len(passed)}  cands={len(res)}  noise={noise:.1f}  {dt:.0f}ms', (10, 25))
        tiles = [zoom(img, m) for m in (passed + [m for m in probes if not math.isnan(m.r)])[:max(1, W // 240)]]
        if tiles:
            strip = np.zeros((240, W, 3), np.uint8)
            for i, tl in enumerate(tiles):
                strip[:, i * 240:(i + 1) * 240] = tl
                draw.text(strip, f'#{i + 1}' if i < len(passed) else 'probe', (i * 240 + 5, 20), draw.GREEN)
            vis = np.vstack([vis, strip])
        out_path = os.path.join(a.out, f'{name}_det.jpg')
        cv2.imwrite(out_path, vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
        results.append(dict(file=path, size=[W, H], r_range=list(r_range), noise=noise, ms=dt,
                            candidates=len(res),
                            passed=[dict(cx=m.cx, cy=m.cy, r=m.r, ellipse=m.ellipse, sectors=m.sectors,
                                         max_gap=m.max_gap, inlier_ratio=m.inlier_ratio, rms=m.rms,
                                         contrast=m.contrast, line_w=m.line_w) for m in passed]))
    with open(os.path.join(a.out, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f'\n标注图和 results.json 在 {a.out}/')


if __name__ == '__main__':
    main()
