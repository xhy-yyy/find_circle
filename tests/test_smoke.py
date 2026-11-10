"""冒烟测试：python3 -m pytest -q tests/"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from ringdet import Params, Camera, RingTracker, detect_frame, synth, edcircles  # noqa: E402


def _passed(img, gt, r_range=(8, 60)):
    H, W = img.shape
    res, _ = detect_frame(img, Camera.default(W, H), Params(), r_range=r_range)
    ok = [m for _, m in res if m.ok]
    hit = [m for m in ok if gt and math.hypot(m.cx - gt['cx'], m.cy - gt['cy']) < 0.5 * gt['r']]
    return ok, hit


def test_edcircles_backend_available():
    assert edcircles.backend() != '', edcircles.describe()


@pytest.mark.parametrize('r', [12, 20, 40])
def test_single_frame_finds_ring(r):
    rng = np.random.default_rng(r)
    img, gt = synth.scene(rng, size=(640, 360), r=r, contrast=60, distractors=False)
    ok, hit = _passed(img, gt)
    assert hit, f'没找到 r={r} 的环'
    assert math.hypot(hit[0].cx - gt['cx'], hit[0].cy - gt['cy']) < 0.5


def test_no_ring_no_lock():
    rng = np.random.default_rng(5)
    trk = RingTracker(Params(), Camera.default(640, 360))
    trk.set_click(320, 180)
    for i in range(10):
        img, _ = synth.scene(rng, size=(640, 360), ring=False)
        res = trk.process(img, i / 30, i)
        assert res.state == 'LOST'


def test_no_click_no_search():
    rng = np.random.default_rng(0)
    frames, _ = synth.descent(rng, n=10, size=(640, 360), r0=15, r1=20, distractors=False)
    trk = RingTracker(Params(), Camera.default(640, 360))
    assert all(trk.process(f, i / 30, i).state == 'LOST' for i, f in enumerate(frames))


def test_tracker_locks_and_follows():
    rng = np.random.default_rng(0)
    frames, gts = synth.descent(rng, n=40, size=(640, 360), r0=12, r1=30, distractors=False)
    trk = RingTracker(Params(), Camera.default(640, 360))
    trk.set_click(gts[0][0] + 80, gts[0][1] - 50)     # 粗点：偏 ~94 px
    states, clicks = [], []
    for i, (f, gt) in enumerate(zip(frames, gts)):
        res = trk.process(f, i / 30, i)
        states.append(res.state)
        clicks.append(res.click_px)
        if res.state == 'DETECTED':
            assert math.hypot(res.center_px[0] - gt[0], res.center_px[1] - gt[1]) < 1.0
    first = states.index('DETECTED')
    assert first <= 10
    # 首锁前给出点击点位置，首锁后清空
    assert clicks[0] is not None and clicks[first] is None
    assert states[first:].count('DETECTED') >= 0.9 * (len(states) - first)


def test_ring_appears_only_after_descent():
    """100 m 看不清：点击后环一直看不见，降到半径约 70 px（超过默认上限 60）才出现，仍要锁上。"""
    rng = np.random.default_rng(3)
    n = 60
    frames, gts = synth.descent(rng, n=n, size=(640, 360), r0=8, r1=100, hide=set(range(51)))
    trk = RingTracker(Params(), Camera.default(640, 360))
    trk.set_click(gts[0][0] + 40, gts[0][1] - 30)
    states = []
    for i, (f, gt) in enumerate(zip(frames, gts)):
        res = trk.process(f, i / 30, i)
        states.append(res.state)
        if res.state == 'DETECTED':
            assert math.hypot(res.center_px[0] - gt[0], res.center_px[1] - gt[1]) < 2.0
        if i < 51:
            assert res.state == 'LOST'
    assert 'DETECTED' in states[51:], states[51:]
