"""冒烟测试：python3 -m pytest -q tests/"""
import math
import os
import sys
import shutil
import threading
import time
from concurrent.futures import Future

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
    assert tracker.online is not None
    online = tracker.online['spectrum'].copy()
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
        np.testing.assert_array_equal(tracker.online['spectrum'], online)
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


def test_recovery_uses_trusted_scale_after_weak_radius_shrink(monkeypatch):
    rng = np.random.default_rng(67)
    image = synth.ground_texture(360, 640, rng)
    synth.draw_ring(image, 320, 180, 18, 2, 75)
    image = synth.degrade(image, rng)
    tracker = RingTracker(Params(), Camera.default(640, 360))
    tracker.set_click(320, 180)
    for i in range(12):
        result = tracker.process(image, i / 30, i)
    assert result.state == 'DETECTED'
    tracker.tgt[2] *= .72
    monkeypatch.setattr(tracker.motion, 'update', lambda _: None)
    for i in range(12, 15):
        result = tracker.process(image, i / 30, i)
    assert result.state == 'DETECTED' and result.target_id == 1
    assert math.dist(result.center_px, (320, 180)) < 1.5
    assert abs(result.radius_px - 18) < 1.5
    tracker.close()


def test_frozen_appearance_does_not_accumulate_scale_drift(monkeypatch):
    from ringdet.motion import Motion
    rng = np.random.default_rng(68)
    image = synth.ground_texture(360, 640, rng)
    synth.draw_ring(image, 320, 180, 18, 2, 75)
    image = synth.degrade(image, rng)
    tracker = RingTracker(Params(), Camera.default(640, 360))
    tracker.set_click(320, 180)
    try:
        for i in range(12):
            result = tracker.process(image, i / 30, i)
        assert result.state == 'DETECTED'
        references = [ref['image'].copy() for ref in tracker.references]
        score = tracker._reference_score
        monkeypatch.setattr(tracker, '_reference_score', lambda *_: .7)
        transform = Motion(np.array([[1.01, 0., -3.2], [0., 1.01, -1.8]]), 80)
        monkeypatch.setattr(tracker.motion, 'update', lambda _: transform)
        for i in range(12, 52):
            result = tracker.process(image, i / 30, i)
            assert result.state == 'DETECTED'
        assert abs(tracker.reference_radius - 18) < 1
        for ref, original in zip(tracker.references, references):
            np.testing.assert_array_equal(ref['image'], original)
        monkeypatch.setattr(tracker, '_reference_score', score)
        monkeypatch.setattr(tracker.motion, 'update', lambda _: None)
        shifted = cv2.warpAffine(image, np.float32([[1, 0, 70], [0, 1, 0]]), (640, 360))
        for i in range(52, 55):
            result = tracker.process(shifted, i / 30, i)
        assert result.state == 'DETECTED' and result.target_id == 1
        assert math.dist(result.center_px, (390, 180)) < 1.5
    finally:
        tracker.close()


def test_expensive_template_cannot_starve_search_or_timeout(monkeypatch):
    tracker = RingTracker(Params(), Camera.default(320, 180))
    tracker.phase = 'REACQUIRE'
    tracker.tgt = [160., 90., 12.]
    tracker.last_detect = 0.
    tracker.reacq = dict(start=0., x=160., y=90., r=12.)
    tracker.P.track.frame_budget_ms = 2
    tracker.P.search.use_edcircles = False
    tracker.P.search.use_matched_filter = False
    monkeypatch.setattr(tracker.motion, 'update', lambda _: None)
    def slow_match(*_):
        time.sleep(.005)
        return None
    monkeypatch.setattr(tracker, '_template_measure', slow_match)
    image = np.zeros((180, 320), np.uint8)
    results = [tracker.process(image, 1.1 + i / 30, i) for i in range(3)]
    assert any('jobs' in result.debug for result in results)
    tracker.P.track.frame_budget_ms = 0
    result = tracker.process(image, 3.1, 3)
    assert result.state == 'LOST' and result.phase == 'SEARCH'
    assert tracker.r_floor is not None
    tracker.close()


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
    assert tracker.online is None


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
    assert tracker.online is None


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


def test_online_recovery_works_without_template_search(monkeypatch):
    rng = np.random.default_rng(73)
    image = synth.ground_texture(360, 640, rng, amp=12)
    synth.draw_ring(image, 320, 180, 18, 2, 75)
    image = synth.degrade(image, rng)
    tracker = RingTracker(Params(), Camera.default(640, 360))
    tracker.set_click(320, 180)
    try:
        for i in range(15):
            result = tracker.process(image, i / 30, i)
        assert result.state == 'DETECTED' and tracker.online['updates'] >= 2
        protected = tracker.references[0]['image'].copy()
        monkeypatch.setattr(tracker.motion, 'update', lambda _: None)
        def unavailable(*_):
            pytest.fail('高置信度在线候选不应重复做模板搜索')
        monkeypatch.setattr(tracker, '_match_templates', unavailable)
        shifted = cv2.warpAffine(image, np.float32([[1, 0, 70], [0, 1, 0]]), (640, 360))
        for i in range(15, 18):
            result = tracker.process(shifted, i / 30, i)
            assert (result.state == 'DETECTED') == (i == 17)
        assert result.target_id == 1 and result.debug['recovery_source'] == 'online'
        assert math.dist(result.center_px, (390, 180)) < 1.5
        np.testing.assert_array_equal(tracker.references[0]['image'], protected)
    finally:
        tracker.close()


def test_online_candidates_adapt_to_scale_and_rotation():
    from ringdet.motion import Motion
    from ringdet.frame import Frame
    rng = np.random.default_rng(74)
    image = synth.ground_texture(360, 640, rng, amp=12)
    synth.draw_ring(image, 320, 180, 18, 2, 75)
    cv2.circle(image, (365, 180), 14, 200, -1)
    image = synth.degrade(image, rng)
    tracker = RingTracker(Params(), Camera.default(640, 360))
    tracker.set_click(320, 180)
    try:
        for i in range(18):
            result = tracker.process(image, i / 30, i)
        assert result.state == 'DETECTED' and tracker.online is not None
        A = cv2.getRotationMatrix2D((320, 180), 12, 1.25)
        A[:, 2] += (45, -12)
        moved = cv2.warpAffine(image, A, (640, 360))
        tracker._propagate(Motion(A, 80))
        tracker.dbg = {}
        peaks = tracker._online_peaks(Frame(moved, 2), *tracker.tgt, 150)
        assert peaks and tracker.dbg['online_psr'] >= tracker.P.track.online_psr
        assert math.dist(peaks[0][1:3], (365, 168)) < 4
    finally:
        tracker.close()


def test_motion_learning_is_independent_of_frozen_appearance(monkeypatch):
    from ringdet.motion import Motion
    rng = np.random.default_rng(75)
    image = synth.ground_texture(360, 640, rng, amp=12)
    synth.draw_ring(image, 320, 180, 18, 2, 75)
    image = synth.degrade(image, rng)
    tracker = RingTracker(Params(), Camera.default(640, 360))
    tracker.set_click(320, 180)
    try:
        for i in range(15):
            result = tracker.process(image, i / 30, i)
        assert result.state == 'DETECTED' and tracker.online is not None
        working = tracker.template['image'].copy()
        spectrum = tracker.online['spectrum'].copy()
        reference = tracker.references[0]['image'].copy()
        monkeypatch.setattr(tracker, '_reference_score', lambda *_: .7)
        monkeypatch.setattr(tracker.motion, 'update', lambda _: Motion(np.float64([[1, 0, 3], [0, 1, 0]]), 80))
        for k in range(1, 13):
            moved = cv2.warpAffine(image, np.float32([[1, 0, 3 * k], [0, 1, 0]]), (640, 360))
            result = tracker.process(moved, (14 + k) / 30, 14 + k)
            assert result.state == 'DETECTED'
        assert tracker.learned['stamp'] == pytest.approx(26 / 30)
        assert tracker.learned['velocity'][0] > 60
        np.testing.assert_array_equal(tracker.template['image'], working)
        np.testing.assert_array_equal(tracker.online['spectrum'], spectrum)
        np.testing.assert_array_equal(tracker.references[0]['image'], reference)
    finally:
        tracker.close()


def test_shared_fft_ncc_matches_opencv():
    rng = np.random.default_rng(61)
    roi = rng.integers(0, 256, (240, 260), dtype=np.uint8)
    templates = [roi[40:40+n, 55:55+n].copy() for n in (19, 27, 35)]
    params = Params()
    params.track.template_backend = 'cpu'
    for template, scores in zip(templates, RingTracker(params)._match_templates(roi, templates)):
        expected = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
        np.testing.assert_allclose(scores, expected, atol=2e-4)
        assert cv2.minMaxLoc(scores)[3] == (55, 40)


def test_first_confirmation_is_not_starved_by_regional_ed(monkeypatch):
    import ringdet.tracker as module
    rng = np.random.default_rng(65)
    image = synth.ground_texture(240, 360, rng, amp=8)
    synth.draw_ring(image, 180, 120, 6, 1.2, 95)
    image = synth.degrade(image, rng, blur=.5, noise=1)
    tracker = RingTracker(Params(), Camera.default(360, 240))
    tracker.warmup()
    tracker.set_click(180, 120)
    monkeypatch.setattr(module, 'ed_support', lambda *_: tracker.n_frames >= 7)
    def occupied(*_):
        tracker._ed_future = Future()
        tracker._ed_meta = dict(valid=True, stamp=0., A=np.eye(3))
        return True
    monkeypatch.setattr(tracker, '_submit_ed', occupied)
    try:
        for i in range(8):
            result = tracker.process(image, i / 30, i)
            if result.state == 'DETECTED':
                break
        assert result.state == 'DETECTED' and result.frame_id <= 7
    finally:
        tracker.close()


@pytest.mark.parametrize('invalidate', [None, 'motion', 'click', 'age'])
def test_async_ed_candidates_require_valid_current_coordinates(invalidate):
    from ringdet.motion import Motion
    from ringdet.search import Candidate
    tracker = RingTracker()
    tracker.dbg = {}
    future = Future()
    tracker._ed_future = future
    tracker._ed_meta = dict(valid=True, stamp=1.0,
                           A=np.array([[1., 0, 100], [0, 1., 200], [0, 0, 1.]]))
    tracker._propagate(Motion(np.array([[1.1, 0, 5], [0, 1.1, -2.]]), 80))
    if invalidate == 'motion':
        tracker._propagate(None)
    elif invalidate == 'click':
        tracker.set_click(120, 120)
        tracker._apply_click(1.03)
    future.set_result([Candidate(10, 12, 6, 100, -1)])
    found = tracker._poll_ed(1.4 if invalidate == 'age' else 1.03)
    if invalidate:
        assert not found
    else:
        assert len(found) == 1
        np.testing.assert_allclose((found[0].x, found[0].y, found[0].r), (126, 231.2, 6.6))


def test_cpu_native_flow_matches_opencv():
    rng = np.random.default_rng(62)
    prev = np.clip(synth.ground_texture(180, 320, rng), 0, 255).astype(np.uint8)
    current = cv2.warpAffine(prev, np.float32([[1, 0, 7], [0, 1, -3]]), (320, 180))
    points = cv2.goodFeaturesToTrack(prev, 80, .01, 10)
    native = edcircles.flow_fb(prev, current, points)
    assert native is not None, '先运行 ./build.sh'
    expected, status, _ = cv2.calcOpticalFlowPyrLK(prev, current, points, None,
        winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
    np.testing.assert_array_equal(native[1], status.ravel())
    np.testing.assert_allclose(native[0][status.ravel() != 0], expected[status.ravel() != 0], atol=1e-4)


def test_native_ellipse_preserves_measurement(monkeypatch):
    from ringdet import verify
    rng = np.random.default_rng(63)
    image = synth.ground_texture(240, 360, rng, amp=8)
    synth.draw_ring(image, 180, 120, 20, 2, 85, axes=(24, 16), angle_deg=28)
    image = synth.degrade(image, rng, blur=.5, noise=1)
    args = (image.astype(np.float32), Camera.default(360, 240), 180, 120, 20, 2, Params().verify)
    accelerated = verify.measure(*args)
    monkeypatch.setattr(verify, 'ellipse_hypotheses', lambda *_: None)
    reference = verify.measure(*args)
    assert accelerated.ok and reference.ok and accelerated.ellipse is not None
    np.testing.assert_allclose(accelerated.ellipse, reference.ellipse, atol=1e-4)
    assert accelerated.sectors == reference.sectors


def test_cuda_auto_falls_back_without_device(monkeypatch):
    from ringdet.motion import GroundMotion
    monkeypatch.setattr(cv2.cuda, 'getCudaEnabledDeviceCount', lambda: 0)
    assert GroundMotion(Params().motion).backend == 'cpu'
    params = Params()
    params.motion.backend = 'cuda'
    with pytest.raises(RuntimeError, match='CUDA'):
        GroundMotion(params.motion)


@pytest.mark.skipif(cv2.cuda.getCudaEnabledDeviceCount() == 0 or not (
    hasattr(cv2, 'cuda_SparsePyrLKOpticalFlow') or hasattr(cv2.cuda, 'SparsePyrLKOpticalFlow_create')),
    reason='需要 CUDA 设备和 OpenCV cudaoptflow 模块')
def test_cuda_motion_tracks_translation():
    from ringdet.motion import GroundMotion
    rng = np.random.default_rng(64)
    image = np.clip(synth.ground_texture(360, 640, rng), 0, 255).astype(np.uint8)
    params = Params()
    params.motion.backend = 'cuda'
    motion = GroundMotion(params.motion)
    motion.focus = (320, 180, 15)
    motion.update(image)
    shifted = cv2.warpAffine(image, np.float32([[1, 0, 9], [0, 1, -4]]), (640, 360))
    transform = motion.update(shifted)
    assert motion.backend == 'cuda' and transform is not None
    assert math.dist(transform.apply(320, 180), (329, 176)) < 1


@pytest.mark.skipif(cv2.cuda.getCudaEnabledDeviceCount() == 0 or not hasattr(cv2.cuda, 'createTemplateMatching'),
    reason='需要 CUDA 设备和 OpenCV CUDA 模板匹配模块')
def test_cuda_ncc_matches_cpu():
    rng = np.random.default_rng(66)
    roi = rng.integers(0, 256, (180, 240), dtype=np.uint8)
    template = roi[40:65, 55:80].copy()
    params = Params()
    params.motion.backend = 'cpu'
    params.track.template_backend = 'cuda'
    tracker = RingTracker(params)
    scores = tracker._match_templates(roi, [template])[0]
    assert tracker.template_backend == 'cuda'
    np.testing.assert_allclose(scores, cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED), atol=1e-3)


def test_record_queue_does_not_block_vision(monkeypatch):
    from tools.run_camera import AsyncVideoWriter
    entered, release = threading.Event(), threading.Event()
    class Writer:
        def isOpened(self): return True
        def set(self, *_): pass
        def write(self, _):
            entered.set()
            release.wait(2)
        def release(self): pass
    monkeypatch.setattr(cv2, 'VideoWriter', lambda *_: Writer())
    recorder = AsyncVideoWriter('/unused.avi', 30, (32, 24))
    try:
        recorder.submit(np.zeros((24, 32, 3), np.uint8))
        assert entered.wait(1)
        for _ in range(6):
            recorder.submit(np.zeros((24, 32, 3), np.uint8))
        assert recorder.dropped >= 2
    finally:
        release.set()
        recorder.close()
    assert not recorder.is_alive()


@pytest.mark.skipif(shutil.which('ffmpeg') is None, reason='需要 ffmpeg 才能直存 MJPEG')
def test_mjpeg_copy_preserves_source_packets(tmp_path):
    from tools.run_camera import AsyncVideoWriter
    image = np.full((48, 64, 3), 125, np.uint8)
    _, packet = cv2.imencode('.jpg', image)
    path = tmp_path / 'raw.avi'
    recorder = AsyncVideoWriter(str(path), 30, (64, 48), copy_mjpeg=True)
    recorder.submit(packet.tobytes())
    recorder.submit(packet.tobytes())
    recorder.close()
    assert recorder.error is None and recorder.written == 2 and recorder.dropped == 0
    assert packet.tobytes() in path.read_bytes()
    cap = cv2.VideoCapture(str(path))
    try:
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 2
        ok, decoded = cap.read()
        assert ok
        np.testing.assert_array_equal(decoded, cv2.imdecode(packet, cv2.IMREAD_COLOR))
    finally:
        cap.release()
