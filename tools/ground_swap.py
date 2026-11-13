#!/usr/bin/env python3
"""用实拍的石灰环近照，换到不同地面、缩到高空尺度，测各检测通道的召回和误检。

做法：
  1. 从近照估计每个像素的石灰覆盖率 alpha（按白度 min(R,G,B) 相对周围地面的提升），
     石灰环本身的粗细不匀、断续、撒落的粉末都保留。
  2. 按目标半径把 alpha 缩小（面积平均），贴到程序生成的各种地面上：
     out = alpha * 石灰颜色 + (1 - alpha) * 地面
  3. 光学模糊 + 噪声 + JPEG 4:2:0（接近 USB 相机的 MJPEG），每张图还放碎石和白色干扰物。
  4. 每个通道分别跑单帧检测，统计真环通过率和误检。

用法：python3 tools/ground_swap.py 近照.png [--gt cx,cy,r] [--radii 8,10,14] [--trials 8] [--out out/ground_swap]
--gt 不给时先在近照上自动检测一次。
"""
import argparse
import math
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from ringdet import Params, Camera, detect_frame, synth  # noqa: E402

CHANNELS = ('双通道',)       # 复核时先灰度、再白度 min(B,G,R)，见 ringdet/frame.py


# 与近照同一曝光下（石灰约 200）的地面颜色，BGR
GROUNDS = {
    '红土': (75, 105, 160),
    '深色土': (60, 70, 85),
    '绿草': (45, 120, 75),
    '枯草': (90, 150, 170),
    '黄沙': (120, 170, 195),
    '红色塑胶跑道': (60, 70, 170),
    '深色沥青': (70, 70, 72),
    '灰水泥': (140, 140, 140),
    '浅色水泥': (170, 170, 170),
}


def extract_matte(bgr):
    """返回 alpha (float32, 0~1) 和石灰颜色 (BGR)。"""
    w = np.min(bgr, axis=2).astype(np.uint8)
    k = int(max(31, round(min(bgr.shape[:2]) / 12)) | 1)
    bg = cv2.medianBlur(w, k).astype(np.float32)          # 线细于窗口：中值 ≈ 地面
    wf = w.astype(np.float32)
    lift = wf - bg
    sel = lift > 40
    if sel.sum() < 50:
        sys.exit('近照里没找到明显比地面白的像素，换一张图')
    L = float(np.percentile(wf[sel], 98))
    alpha = np.clip(lift / np.maximum(L - bg, 1.0), 0, 1)
    alpha[alpha < 0.12] = 0
    lime = np.median(bgr[alpha > 0.9].reshape(-1, 3), axis=0) if (alpha > 0.9).sum() > 20 else np.array([L] * 3)
    return alpha.astype(np.float32), lime.astype(np.float32)


def ground_image(rng, size, color, amp=12.0):
    W, H = size
    tex = synth.ground_texture(H, W, rng, base=0.0, amp=amp)
    add = np.zeros((H, W), np.float32)
    synth.add_speckles(add, rng, int(W * H / 900), amp=45.0)          # 碎石，亮暗都有
    img = np.empty((H, W, 3), np.float32)
    tint = rng.normal(0, 4, 3)
    for c in range(3):
        img[..., c] = color[c] + tint[c] + tex * (0.8 + 0.4 * rng.random()) + add
    return img


def add_white_distractors(img, rng, r):
    """白色干扰：实心白圆、白线、半段同尺寸白弧、大块浅色区域。直接画到三个通道上。"""
    H, W = img.shape[:2]
    lay = np.zeros((H, W), np.float32)
    synth.add_distractors(lay, rng, r, n_disc=4, n_line=3, n_arc=2, n_blob=3, contrast=(40, 90))
    for c in range(3):
        img[..., c] += lay
    return img


def make_case(seed, ground, r_target, size, alpha, lime, gt_r, ring=True):
    rng = np.random.default_rng(seed)
    W, H = size
    img = ground_image(rng, size, GROUNDS[ground])
    img = add_white_distractors(img, rng, r_target)
    gt = None
    if ring:
        k = gt_r / r_target
        a_s = cv2.resize(alpha, None, fx=1 / k, fy=1 / k, interpolation=cv2.INTER_AREA)
        h, w = a_s.shape
        x0 = int(rng.integers(5, W - w - 5))
        y0 = int(rng.integers(5, H - h - 5))
        sub = (rng.random(2) - 0.5)          # 亚像素位置随机
        M = np.float32([[1, 0, sub[0]], [0, 1, sub[1]]])
        a_s = cv2.warpAffine(a_s, M, (w, h), flags=cv2.INTER_LINEAR)
        roi = img[y0:y0 + h, x0:x0 + w]
        img[y0:y0 + h, x0:x0 + w] = roi * (1 - a_s[..., None]) + lime[None, None, :] * a_s[..., None]
        gt = (alpha_center[0] / k + x0 + sub[0], alpha_center[1] / k + y0 + sub[1], r_target)
    img = cv2.GaussianBlur(img, (0, 0), 0.7)
    img += rng.normal(0, 2.0, img.shape).astype(np.float32)
    u8 = np.clip(img, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode('.jpg', u8, [cv2.IMWRITE_JPEG_QUALITY, 80])   # OpenCV 默认 4:2:0
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), gt


def run_case(job):
    cv2.setNumThreads(1)
    seed, ground, r_t, size, ring = job
    bgr, gt = make_case(seed, ground, r_t, size, ALPHA, LIME, GT_R, ring)
    cam = Camera.default(*size)
    out = {}
    for ch in CHANNELS:
        res, _ = detect_frame(bgr, cam, PARAMS, r_range=(0.7 * r_t, 1.4 * r_t))
        hit, fp, why = False, 0, ''
        for c, m in res:
            if gt and math.hypot(c.x - gt[0], c.y - gt[1]) < 0.5 * r_t and not why:
                why = m.reason or 'ok'
            # 只统计可首锁的结果（与跟踪器一致：EDCircles 找到且通过复核）
            if not m.ok or (NEED_ED and not m.ed_ok):
                continue
            if gt and math.hypot(m.cx - gt[0], m.cy - gt[1]) < 0.5 * r_t:
                hit = True
            else:
                fp += 1
        out[ch] = (hit, fp, why or '不在候选里')
    return ground, r_t, ring, out


def init(alpha, lime, gt_r, center, params):
    global ALPHA, LIME, GT_R, alpha_center, PARAMS, NEED_ED
    ALPHA, LIME, GT_R, alpha_center, PARAMS = alpha, lime, gt_r, center, params
    NEED_ED = params.track.lock_need_edcircles and params.search.use_edcircles


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('image')
    ap.add_argument('--gt')
    ap.add_argument('--radii', default='8,10,14')
    ap.add_argument('--trials', type=int, default=8)
    ap.add_argument('--size', default='640x360')
    ap.add_argument('--out', default='out/ground_swap')
    ap.add_argument('--grounds', help='逗号分隔，默认全部')
    ap.add_argument('--set', action='append', default=[])
    a = ap.parse_args()
    P = Params.load(None, a.set)
    bgr = cv2.imread(a.image, cv2.IMREAD_COLOR)
    if bgr is None:
        sys.exit(f'读不了 {a.image}')
    if a.gt:
        cx, cy, r = (float(v) for v in a.gt.split(','))
    else:
        H, W = bgr.shape[:2]
        res, _ = detect_frame(bgr, Camera.default(W, H), P, r_range=(20, min(W, H) / 2))
        ok = sorted([m for _, m in res if m.ok], key=lambda m: (-m.sectors, -m.r))
        if not ok:
            sys.exit('近照上没检测到环，用 --gt cx,cy,r 手动给')
        cx, cy, r = ok[0].cx, ok[0].cy, ok[0].r
    print(f'近照环：c=({cx:.1f},{cy:.1f}) r={r:.1f}')
    alpha, lime = extract_matte(bgr)
    # 只保留环附近（1.4r 以内）的石灰，远处的白色杂物不搬过去
    H, W = alpha.shape
    yy, xx = np.mgrid[0:H, 0:W]
    alpha[np.hypot(xx - cx, yy - cy) > 1.4 * r] = 0
    x0, y0 = int(max(0, cx - 1.45 * r)), int(max(0, cy - 1.45 * r))
    x1, y1 = int(min(W, cx + 1.45 * r)), int(min(H, cy + 1.45 * r))
    alpha = np.ascontiguousarray(alpha[y0:y1, x0:x1])
    center = (cx - x0, cy - y0)
    os.makedirs(a.out, exist_ok=True)
    cv2.imwrite(os.path.join(a.out, 'matte.png'), (alpha * 255).astype(np.uint8))
    print(f'石灰颜色 BGR={lime.round(0)}，覆盖率图 {a.out}/matte.png')

    size = tuple(int(v) for v in a.size.split('x'))
    radii = [float(v) for v in a.radii.split(',')]
    grounds = a.grounds.split(',') if a.grounds else list(GROUNDS)
    jobs = []
    for g in grounds:
        for rt in radii:
            for t in range(a.trials):
                jobs.append((zlib.crc32(f'{g}|{rt}|{t}'.encode()), g, rt, size, True))
        for t in range(a.trials):
            jobs.append((zlib.crc32(f'{g}|empty|{t}'.encode()), g, radii[0], size, False))
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2), initializer=init,
                             initargs=(alpha, lime, r, center, P)) as ex:
        results = list(ex.map(run_case, jobs, chunksize=2))

    print(f'\n每格：通过/试验次数（每张图 {size[0]}x{size[1]}，半径范围 0.7r~1.4r，含碎石与白色干扰物）')
    print('只统计可首锁的结果（通过复核且有 EDCircles 支持）；--set track.lock_need_edcircles=false 统计全部来源')
    head = f"{'地面':<10}" + ''.join(f"{f'r={rt:g}':^16}" for rt in radii) + f"{'误检(有环+无环图)':^20}"
    print(head)
    print(f"{'':<10}" + ''.join(f"{'  '.join(CHANNELS):^16}" for _ in radii) + f"{'  '.join(CHANNELS):^20}")
    tot = {ch: [0, 0, 0] for ch in CHANNELS}
    fails = {}
    for g in grounds:
        row = f'{g:<10}'
        for rt in radii:
            cells = []
            for ch in CHANNELS:
                rs = [o[ch] for gg, rr, ring, o in results if gg == g and rr == rt and ring]
                h = sum(x[0] for x in rs)
                tot[ch][0] += h
                tot[ch][1] += len(rs)
                cells.append(f'{h}/{len(rs)}')
                for x in rs:
                    if not x[0]:
                        fails.setdefault((g, rt, ch), []).append(x[2])
            row += f"{'  '.join(cells):^16}"
        fps = []
        for ch in CHANNELS:
            n = sum(o[ch][1] for gg, rr, ring, o in results if gg == g)
            tot[ch][2] += n
            fps.append(str(n))
        row += f"{'  '.join(fps):^20}"
        print(row)
    n_img = len(results)
    print('\n合计：' + '   '.join(f'{ch} 通过 {v[0]}/{v[1]}，误检 {v[2]}（{n_img} 张图）' for ch, v in tot.items()))
    print('\n漏检原因（地面, r, 通道）：')
    for k, v in sorted(fails.items()):
        from collections import Counter
        print(f'  {k[0]} r={k[1]:g} {k[2]}: ' + ', '.join(f'{why}×{n}' for why, n in Counter(v).most_common(3)))

    # 样例拼图：每种地面一张 r=最小半径，放大显示
    init(alpha, lime, r, center, P)
    tiles = []
    for g in grounds:
        img, gt = make_case(1, g, radii[0], size, alpha, lime, r)
        x, y = int(gt[0]), int(gt[1])
        R = int(radii[0] * 2.2)
        pad = cv2.copyMakeBorder(img, R, R, R, R, cv2.BORDER_REFLECT)
        crop = cv2.resize(pad[y:y + 2 * R, x:x + 2 * R], (200, 200), interpolation=cv2.INTER_NEAREST)
        cv2.putText(crop, g.encode('ascii', 'ignore').decode() or str(len(tiles)), (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        tiles.append(crop)
    while len(tiles) % 5:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(os.path.join(a.out, f'samples_r{radii[0]:g}.png'),
                np.vstack([np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)]))
    print(f'\n样例（顺序同上表，r={radii[0]:g}）：{a.out}/samples_r{radii[0]:g}.png')


if __name__ == '__main__':
    main()
