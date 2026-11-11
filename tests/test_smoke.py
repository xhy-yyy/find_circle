"""冒烟测试：python3 -m pytest -q tests/"""
import math
import os
import sys

import numpy as np
import cv2
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


def test_target_motion_ignores_different_foreground_motion():
    from ringdet.motion import GroundMotion
    rng = np.random.default_rng(21)
    far = np.clip(synth.ground_texture(360, 640, rng, amp=18), 0, 255).astype(np.uint8)
    near = np.clip(synth.ground_texture(360, 640, rng, amp=55), 0, 255).astype(np.uint8)
    first = far.copy()
    first[:, :360] = near[:, :360]
    second = cv2.warpAffine(far, np.float32([[1, 0, 4], [0, 1, -2]]), (640, 360))
    foreground = cv2.warpAffine(near, np.float32([[1, 0, -12], [0, 1, 7]]), (640, 360))
    second[:, :360] = foreground[:, :360]
    motion = GroundMotion(Params().motion)
    motion.focus = (510, 180, 12)
    motion.update(first)
    transform = motion.update(second)
    assert transform is not None
    assert math.dist(transform.apply(510, 180), (514, 178)) < 1


def test_reacquire_same_ring_after_occlusion_with_decoy():
    hidden = set(range(15, 36))
    frames, gts = synth.descent(np.random.default_rng(17), n=55, size=(640, 360),
                               r0=12, r1=16, contrast=65, drift=(1.2, 0.2),
                               jitter=1.2, rot_deg=0.8, hide=hidden, distractors=False)
    tracker = RingTracker(Params(), Camera.default(640, 360))
    tracker.set_click(gts[0][0], gts[0][1])
    results = []
    for i, (frame, gt) in enumerate(zip(frames, gts)):
        if i in hidden:
            frame = frame.astype(np.float32)
            synth.draw_ring(frame, gt[0] + 65, gt[1], gt[2], 1.5, 65)
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        result = tracker.process(frame, i / 30, i)
        results.append(result)
        if i in hidden:
            assert result.state != 'DETECTED', f'遮挡时误锁：frame={i}, {result.center_px}'
        elif result.state == 'DETECTED':
            assert math.dist(result.center_px, gt[:2]) < 2
    assert any(r.state == 'LOST' for r in results[15:36])
    assert any(r.state == 'DETECTED' for r in results[36:40])
    assert {r.target_id for r in results if r.state == 'DETECTED'} == {1}


@pytest.mark.parametrize('dx,dy', [(6, -2), (18, -12)])
def test_template_recovers_jump_when_optical_flow_fails(monkeypatch, dx, dy):
    rng = np.random.default_rng(24)
    image = synth.ground_texture(360, 640, rng)
    synth.draw_ring(image, 320, 180, 14, 1.5, 70)
    image = synth.degrade(image, rng)
    tracker = RingTracker(Params(), Camera.default(640, 360))
    tracker.set_click(320, 180)
    for i in range(10):
        result = tracker.process(image, i / 30, i)
    assert result.state == 'DETECTED'
    target_id = result.target_id
    monkeypatch.setattr(tracker.motion, 'update', lambda _: None)
    shifted = cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, dy]]), (640, 360))
    result = tracker.process(shifted, 10 / 30, 10)
    assert result.state == 'DETECTED'
    assert result.target_id == target_id
    assert math.dist(result.center_px, (320 + dx, 180 + dy)) < 1.5


def test_foreign_circle_cannot_replace_trusted_identity(monkeypatch):
    rng = np.random.default_rng(35)
    background = synth.ground_texture(360, 640, rng)
    image = background.copy()
    synth.draw_ring(image, 320, 180, 18, 2, 75)
    image = synth.degrade(image, rng)
    tracker = RingTracker(Params(), Camera.default(640, 360))
    tracker.set_click(320, 180)
    for i in range(12):
        result = tracker.process(image, i / 30, i)
    assert result.state == 'DETECTED' and tracker.references
    original = tracker.references[0]['image'].copy()
    adaptive = tracker.template['image'].copy()
    reference_count = len(tracker.references)
    monkeypatch.setattr(tracker.motion, 'update', lambda _: None)
    tracker.process(np.zeros((360, 640), dtype=np.uint8), 11.5 / 30, 12)
    assert tracker.phase == 'REACQUIRE'
    foreign = synth.ground_texture(360, 640, np.random.default_rng(36))
    synth.draw_ring(foreign, 320, 180, 18, 2, 75)
    foreign = synth.degrade(foreign, rng)
    for i in range(12, 32):
        result = tracker.process(foreign, i / 30, i)
        assert result.state != 'DETECTED'
        np.testing.assert_array_equal(tracker.references[0]['image'], original)
        np.testing.assert_array_equal(tracker.template['image'], adaptive)
        assert len(tracker.references) == reference_count
    for i in range(32, 34):
        result = tracker.process(image, i / 30, i)
        assert result.state != 'DETECTED'
    result = tracker.process(image, 34 / 30, 34)
    assert result.state == 'DETECTED' and result.target_id == 1
    assert tracker.template_hold > 0
    np.testing.assert_array_equal(tracker.references[0]['image'], original)
    assert len(tracker.references) == reference_count


def test_identity_probation_keeps_valid_continuous_tracking(monkeypatch):
    rng = np.random.default_rng(41)
    image = synth.ground_texture(360, 640, rng)
    synth.draw_ring(image, 320, 180, 18, 2, 75)
    image = synth.degrade(image, rng)
    tracker = RingTracker(Params(), Camera.default(640, 360))
    tracker.set_click(320, 180)
    for i in range(12):
        result = tracker.process(image, i / 30, i)
    assert result.state == 'DETECTED'
    references = [ref['image'].copy() for ref in tracker.references]
    working = tracker.template['image'].copy()
    observation = tracker.learned['observation'].copy()
    tracker.template_hold = tracker.P.track.template_update_hits
    monkeypatch.setattr(tracker, '_recovery_identity', lambda *_: False)
    for i in range(12, 24):
        result = tracker.process(image, i / 30, i)
        assert result.state == 'DETECTED' and result.phase == 'TRACK'
        assert math.dist(result.center_px, (320, 180)) < 1.5
    assert tracker.template_hold == tracker.P.track.template_update_hits
    assert len(tracker.references) == len(references)
    np.testing.assert_array_equal(tracker.template['image'], working)
    np.testing.assert_array_equal(tracker.learned['observation'], observation)
    for ref, original in zip(tracker.references, references):
        np.testing.assert_array_equal(ref['image'], original)


def test_pending_recovery_cannot_bypass_confirmation(monkeypatch):
    rng = np.random.default_rng(42)
    image = synth.ground_texture(360, 640, rng)
    synth.draw_ring(image, 320, 180, 18, 2, 75)
    image = synth.degrade(image, rng)
    tracker = RingTracker(Params(), Camera.default(640, 360))
    tracker.set_click(320, 180)
    for i in range(12):
        result = tracker.process(image, i / 30, i)
    assert result.state == 'DETECTED'
    monkeypatch.setattr(tracker.motion, 'update', lambda _: None)
    shifted = cv2.warpAffine(image, np.float32([[1, 0, 70], [0, 1, 0]]), (640, 360))
    for i in range(12, 14):
        result = tracker.process(shifted, i / 30, i)
        assert result.state == 'LOST' and result.phase == 'REACQUIRE'
    result = tracker.process(shifted, 14 / 30, 14)
    assert result.state == 'DETECTED' and result.target_id == 1
    assert math.dist(result.center_px, (390, 180)) < 1.5


def test_equal_identity_candidates_remain_ambiguous(monkeypatch):
    rng = np.random.default_rng(37)
    background = synth.ground_texture(360, 640, rng)
    image = background.copy()
    synth.draw_ring(image, 320, 180, 18, 2, 75)
    image = synth.degrade(image, rng)
    params = Params()
    params.track.reacq_box_min = 300
    tracker = RingTracker(params, Camera.default(640, 360))
    tracker.set_click(320, 180)
    for i in range(12):
        result = tracker.process(image, i / 30, i)
    assert result.state == 'DETECTED' and tracker.references
    monkeypatch.setattr(tracker.motion, 'update', lambda _: None)
    tracker.process(np.zeros((360, 640), dtype=np.uint8), 11.5 / 30, 12)
    assert tracker.phase == 'REACQUIRE'
    reference = tracker.references[0]['image'].copy()
    ambiguous = np.clip(background, 0, 255).astype(np.uint8)
    n = reference.shape[0]
    for x in (265, 375):
        x0, y0 = x - n // 2, 180 - n // 2
        ambiguous[y0:y0+n, x0:x0+n] = reference
    result = tracker.process(ambiguous, 12 / 30, 12)
    assert result.state == 'LOST' and result.ambiguous
    np.testing.assert_array_equal(tracker.references[0]['image'], reference)


def test_click_can_replace_identity_even_near_current_target():
    rng = np.random.default_rng(38)
    image = synth.ground_texture(360, 640, rng)
    synth.draw_ring(image, 320, 180, 18, 2, 75)
    image = synth.degrade(image, rng)
    tracker = RingTracker(Params(), Camera.default(640, 360))
    tracker.set_click(320, 180)
    for i in range(12):
        result = tracker.process(image, i / 30, i)
    assert result.state == 'DETECTED' and tracker.references
    tracker.tgt = [350, 180, 18]
    tracker.set_click(320, 180)
    result = tracker.process(image, 12 / 30, 12)
    assert result.click_px is not None
    assert not tracker.references
    assert tracker.learned is None


@pytest.mark.parametrize('offset', [(65, -40), (220, 80)])
@pytest.mark.parametrize('height', [None, 100])
def test_click_locks_small_ring_without_descent(offset, height):
    rng = np.random.default_rng(51)
    image = synth.ground_texture(360, 640, rng, amp=10)
    synth.draw_ring(image, 300.3, 190.2, 5.5, 0.9, 85)
    tracker = RingTracker(Params(), Camera.default(640, 360, fx=550))
    tracker.set_height(height)
    tracker.set_click(300.3 + offset[0], 190.2 + offset[1])
    for i in range(24):
        frame = synth.degrade(image, rng, blur=0.55, noise=1, jpeg=85)
        result = tracker.process(frame, i / 30, i)
        if result.state == 'DETECTED':
            break
    assert result.state == 'DETECTED'
    assert math.dist(result.center_px, (300.3, 190.2)) < 1
    assert abs(result.radius_px - 5.5) < 1


@pytest.mark.parametrize('seed', [52, 53, 54])
def test_static_clutter_near_click_does_not_lock(seed):
    rng = np.random.default_rng(seed)
    image, _ = synth.scene(rng, size=(640, 360), r=6, ring=False)
    tracker = RingTracker(Params(), Camera.default(640, 360))
    tracker.set_click(320, 180)
    for i in range(12):
        frame = np.clip(image.astype(np.int16) + rng.integers(-1, 2, image.shape), 0, 255).astype(np.uint8)
        assert tracker.process(frame, i / 30, i).state == 'LOST'


def test_learned_motion_supports_tracking_without_optical_flow(monkeypatch):
    rng = np.random.default_rng(55)
    image = synth.ground_texture(360, 640, rng)
    synth.draw_ring(image, 240, 180, 14, 1.5, 80)
    image = synth.degrade(image, rng)
    tracker = RingTracker(Params(), Camera.default(640, 360))
    tracker.set_click(240, 180)
    for i in range(16):
        frame = cv2.warpAffine(image, np.float32([[1, 0, 5 * i], [0, 1, 0]]), (640, 360))
        result = tracker.process(frame, i / 30, i)
    assert result.state == 'DETECTED'
    assert tracker.learned['hits'] >= 3
    assert tracker.learned['velocity'][0] > 90
    monkeypatch.setattr(tracker.motion, 'update', lambda _: None)
    monkeypatch.setattr(tracker, '_template_measure', lambda *_: None)
    for i in range(16, 19):
        frame = cv2.warpAffine(image, np.float32([[1, 0, 5 * i], [0, 1, 0]]), (640, 360))
        result = tracker.process(frame, i / 30, i)
        assert result.state == 'DETECTED' and result.target_id == 1
        assert math.dist(result.center_px, (240 + 5 * i, 180)) < 1


def test_protected_memory_recovers_when_working_template_is_unusable(monkeypatch):
    rng = np.random.default_rng(56)
    image = synth.ground_texture(360, 640, rng)
    synth.draw_ring(image, 320, 180, 18, 2, 75)
    image = synth.degrade(image, rng)
    tracker = RingTracker(Params(), Camera.default(640, 360))
    tracker.set_click(320, 180)
    for i in range(12):
        result = tracker.process(image, i / 30, i)
    assert result.state == 'DETECTED' and tracker.references
    reference = tracker.references[0]['image'].copy()
    tracker.template['image'] = rng.integers(0, 256, tracker.template['image'].shape, dtype=np.uint8)
    tracker.P.search.use_edcircles = False
    tracker.P.search.use_matched_filter = False
    monkeypatch.setattr(tracker.motion, 'update', lambda _: None)
    shifted = cv2.warpAffine(image, np.float32([[1, 0, 70], [0, 1, 0]]), (640, 360))
    for i in range(12, 15):
        result = tracker.process(shifted, i / 30, i)
        assert (result.state == 'DETECTED') == (i == 14)
    assert result.target_id == 1
    assert math.dist(result.center_px, (390, 180)) < 1.5
    np.testing.assert_array_equal(tracker.references[0]['image'], reference)
