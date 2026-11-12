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
    assert first <= 12
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


def test_global_recovery_retains_growing_target_scale(monkeypatch):
    from ringdet.motion import Motion
    rng = np.random.default_rng(85)
    image = synth.ground_texture(480, 640, rng)
    synth.draw_ring(image, 320, 240, 45, 3, 75)
    image = synth.degrade(image, rng)
    tracker = RingTracker(Params(), Camera.default(640, 480))
    tracker.set_click(320, 240)
    for i in range(24):
        result = tracker.process(image, i / 30, i)
    assert result.state == 'DETECTED'
    stamp = tracker.last_detect
    tracker.phase = 'REACQUIRE'
    tracker.reacq = dict(start=stamp, x=320., y=240., r=tracker.tgt[2])
    A = cv2.getRotationMatrix2D((320, 240), 0, 1.8)
    moved = cv2.warpAffine(image, A, (640, 480))
    monkeypatch.setattr(tracker.motion, 'update', lambda _: Motion(A, 80))
    result = tracker.process(moved, stamp + 3.1, 24)
    assert result.debug['r_range'][1] > 81
    monkeypatch.setattr(tracker.motion, 'update', lambda _: Motion(np.float64([[1, 0, 0], [0, 1, 0]]), 80))
    for i in range(25, 45):
        result = tracker.process(moved, stamp + 3.1 + (i - 24) / 30, i)
        if result.state == 'DETECTED':
            break
    assert result.state == 'DETECTED' and result.target_id == 1
    assert math.dist(result.center_px, (320, 240)) < 2
    assert abs(result.radius_px - 81) < 2
    tracker.close()


def test_initial_identity_rejects_target_moving_relative_to_ground(monkeypatch):
    from ringdet.motion import Motion
    rng = np.random.default_rng(81)
    ground = synth.ground_texture(360, 640, rng)
    tracker = RingTracker(Params(), Camera.default(640, 360))
    tracker.set_click(320, 180)
    monkeypatch.setattr(tracker.motion, 'update', lambda _: Motion(np.float64([[1, 0, 0], [0, 1, 0]]), 80))
    states = []
    for i in range(32):
        image = ground.copy()
        x = 320 + 2 * min(i, 12)
        synth.draw_ring(image, x, 180, 12, 1.5, 80)
        result = tracker.process(synth.degrade(image, rng, noise=0), i / 30, i)
        if i < 12:
            assert result.state == 'LOST' and not tracker.references
        states.append(result.state)
        if result.state == 'DETECTED':
            assert math.dist(result.center_px, (344, 180)) < 1.5
    assert 'DETECTED' in states[12:]
    tracker.close()


def test_keyframe_position_survives_geometry_failure_and_expires(monkeypatch):
    import ringdet.tracker as tracker_module
    from ringdet.verify import Measurement
    rng = np.random.default_rng(82)
    image = synth.ground_texture(360, 640, rng)
    synth.draw_ring(image, 320, 180, 18, 2, 75)
    image = synth.degrade(image, rng)
    tracker = RingTracker(Params(), Camera.default(640, 360))
    tracker.set_click(320, 180)
    for i in range(24):
        result = tracker.process(image, i / 30, i)
    assert result.state == 'DETECTED' and tracker.keyframe is not None
    protected = tracker.references[0]['image'].copy()
    updates = tracker.online['updates']
    detected = tracker.last_detect
    monkeypatch.setattr(tracker.motion, 'update', lambda _: None)
    monkeypatch.setattr(tracker_module, 'measure_channels', lambda *_args, **_kwargs: Measurement(False, 'few_valid'))
    shifted = cv2.warpAffine(image, np.float32([[1, 0, 7], [0, 1, -4]]), (640, 360))
    result = tracker.process(shifted, 24 / 30, 24)
    assert result.state == 'PREDICTED' and result.debug['prediction_source'] == 'appearance'
    assert math.dist(result.center_px, (327, 176)) < 1.5
    assert tracker.last_detect == detected and tracker.online['updates'] == updates
    np.testing.assert_array_equal(tracker.references[0]['image'], protected)
    result = tracker.process(shifted, detected + tracker.P.track.appearance_coast_s + .01, 25)
    assert result.state == 'LOST' and result.center_px is None
    tracker.close()


def test_keyframe_prediction_rejects_equal_appearance_peaks(monkeypatch):
    import ringdet.tracker as tracker_module
    from ringdet.verify import Measurement
    rng = np.random.default_rng(83)
    background = synth.ground_texture(360, 640, rng)
    image = background.copy()
    synth.draw_ring(image, 320, 180, 18, 2, 75)
    image = synth.degrade(image, rng)
    tracker = RingTracker(Params(), Camera.default(640, 360))
    tracker.set_click(320, 180)
    for i in range(24):
        result = tracker.process(image, i / 30, i)
    assert result.state == 'DETECTED'
    canvas = np.clip(background, 0, 255).astype(np.uint8)
    patch = tracker.keyframe['image']
    n = patch.shape[0]
    for x in (250, 390):
        canvas[180 - n // 2:180 - n // 2 + n, x - n // 2:x - n // 2 + n] = patch
    monkeypatch.setattr(tracker.motion, 'update', lambda _: None)
    monkeypatch.setattr(tracker_module, 'measure_channels', lambda *_args, **_kwargs: Measurement(False, 'few_valid'))
    result = tracker.process(canvas, 24 / 30, 24)
    assert result.state == 'LOST' and result.center_px is None
    assert 'prediction_source' not in result.debug
    tracker.close()


def test_keyframe_prediction_rejects_conflict_with_valid_motion(monkeypatch):
    import ringdet.tracker as tracker_module
    from ringdet.motion import Motion
    from ringdet.verify import Measurement
    rng = np.random.default_rng(86)
    image = synth.ground_texture(360, 640, rng)
    synth.draw_ring(image, 320, 180, 18, 2, 75)
    image = synth.degrade(image, rng)
    tracker = RingTracker(Params(), Camera.default(640, 360))
    tracker.set_click(320, 180)
    for i in range(24):
        result = tracker.process(image, i / 30, i)
    assert result.state == 'DETECTED'
    monkeypatch.setattr(tracker.motion, 'update', lambda _: Motion(np.float64([[1, 0, 0], [0, 1, 0]]), 80))
    monkeypatch.setattr(tracker_module, 'measure_channels', lambda *_args, **_kwargs: Measurement(False, 'few_valid'))
    shifted = cv2.warpAffine(image, np.float32([[1, 0, 12], [0, 1, 0]]), (640, 360))
    result = tracker.process(shifted, 24 / 30, 24)
    assert result.state == 'PREDICTED' and 'prediction_source' not in result.debug
    assert math.dist(result.center_px, (320, 180)) < 1.5
    tracker.close()


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


def test_stable_track_waits_through_brief_blur_without_outputting_a_position(monkeypatch):
    rng = np.random.default_rng(57)
    image = synth.ground_texture(360, 640, rng)
    synth.draw_ring(image, 320, 180, 18, 2, 75)
    image = synth.degrade(image, rng)
    tracker = RingTracker(Params(), Camera.default(640, 360))
    tracker.set_click(320, 180)
    for i in range(24):
        result = tracker.process(image, i / 30, i)
    assert result.state == 'DETECTED'
    assert tracker.stable_track_hits >= tracker.P.track.stable_track_frames
    reference = tracker.references[0]['image'].copy()
    online_updates = tracker.online['updates']
    monkeypatch.setattr(tracker.motion, 'update', lambda _: None)
    blurred = cv2.GaussianBlur(image, (0, 0), 16)
    for i in range(24, 26):
        result = tracker.process(blurred, i / 30, i)
        assert result.state == 'LOST' and result.phase == 'COAST'
        assert result.center_px is None and not result.motion_ok
        np.testing.assert_array_equal(tracker.references[0]['image'], reference)
        assert tracker.online['updates'] == online_updates
    result = tracker.process(image, 26 / 30, 26)
    assert result.state == 'DETECTED' and result.target_id == 1
    assert math.dist(result.center_px, (320, 180)) < 1.5
    tracker.close()


def test_broken_motion_chain_requires_identity_and_expires(monkeypatch):
    from ringdet.motion import Motion
    rng = np.random.default_rng(35)
    image = synth.ground_texture(360, 640, rng)
    synth.draw_ring(image, 320, 180, 18, 2, 75)
    image = synth.degrade(image, rng)
    tracker = RingTracker(Params(), Camera.default(640, 360))
    tracker.set_click(320, 180)
    for i in range(24):
        result = tracker.process(image, i / 30, i)
    assert result.state == 'DETECTED' and tracker.stable_track_hits >= 6
    monkeypatch.setattr(tracker.motion, 'update', lambda _: None)
    monkeypatch.setattr(tracker, '_template_measure', lambda *_: None)
    blurred = cv2.GaussianBlur(image, (0, 0), 16)
    result = tracker.process(blurred, 24 / 30, 24)
    assert result.state == 'LOST' and result.phase == 'COAST'
    foreign = synth.ground_texture(360, 640, np.random.default_rng(36))
    synth.draw_ring(foreign, 320, 180, 18, 2, 75)
    foreign = synth.degrade(foreign, rng)
    monkeypatch.setattr(tracker.motion, 'update', lambda _: Motion(np.float64([[1, 0, 0], [0, 1, 0]]), 80))
    result = tracker.process(foreign, 25 / 30, 25)
    assert result.debug['track_meas'].ok
    assert result.state == 'LOST' and result.center_px is None
    result = tracker.process(blurred, 27 / 30, 27)
    assert result.state == 'LOST' and result.phase == 'REACQUIRE'
    tracker.close()


def test_recovery_candidate_guides_search_after_large_jitter(monkeypatch):
    rng = np.random.default_rng(58)
    image = synth.ground_texture(360, 640, rng)
    synth.draw_ring(image, 250, 180, 18, 2, 75)
    image = synth.degrade(image, rng)
    tracker = RingTracker(Params(), Camera.default(640, 360))
    tracker.set_click(250, 180)
    for i in range(24):
        result = tracker.process(image, i / 30, i)
    assert result.state == 'DETECTED' and tracker.stable_track_hits >= 6
    monkeypatch.setattr(tracker.motion, 'update', lambda _: None)
    for i, dx in enumerate((80, 150, 150, 150), 24):
        shifted = cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, 0]]), (640, 360))
        result = tracker.process(shifted, i / 30, i)
        if i < 27:
            assert result.state == 'LOST'
    assert result.state == 'DETECTED' and result.target_id == 1
    assert math.dist(result.center_px, (400, 180)) < 1.5
    tracker.close()


def test_stability_survives_two_weaker_measurements(monkeypatch):
    import ringdet.tracker as tracker_module
    rng = np.random.default_rng(59)
    image = synth.ground_texture(360, 640, rng)
    synth.draw_ring(image, 320, 180, 18, 2, 75)
    image = synth.degrade(image, rng)
    tracker = RingTracker(Params(), Camera.default(640, 360))
    tracker.set_click(320, 180)
    for i in range(24):
        result = tracker.process(image, i / 30, i)
    assert result.state == 'DETECTED' and tracker.stable_track_hits >= 8
    measure = tracker_module.measure_channels
    remaining = [2]

    def weaker(*args, **kwargs):
        m = measure(*args, **kwargs)
        if remaining[0] and args[6] == 'track' and m.ok:
            m.sectors = 12
            remaining[0] -= 1
        return m

    monkeypatch.setattr(tracker_module, 'measure_channels', weaker)
    for i in range(24, 26):
        assert tracker.process(image, i / 30, i).state == 'DETECTED'
    assert tracker.stable_track_hits >= tracker.P.track.stable_track_frames
    monkeypatch.setattr(tracker.motion, 'update', lambda _: None)
    blurred = cv2.GaussianBlur(image, (0, 0), 16)
    result = tracker.process(blurred, 26 / 30, 26)
    assert result.state == 'LOST' and result.phase == 'COAST' and result.center_px is None
    tracker.close()


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
    tracker.P.track.stable_no_flow_s = 0
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


@pytest.mark.parametrize('recent_score', [.76, 1.0])
def test_recent_identity_adapts_only_along_reliable_continuous_measurements(monkeypatch, recent_score):
    from ringdet.motion import Motion
    rng = np.random.default_rng(84)
    image = synth.ground_texture(360, 640, rng)
    synth.draw_ring(image, 320, 180, 18, 2, 75)
    image = synth.degrade(image, rng)
    tracker = RingTracker(Params(), Camera.default(640, 360))
    tracker.set_click(320, 180)
    for i in range(24):
        result = tracker.process(image, i / 30, i)
    assert result.state == 'DETECTED'
    protected = tracker.references[0]['image'].copy()
    original_score = tracker._reference_score

    def score(fr, m, references=None):
        return .7 if references is None else min(recent_score, original_score(fr, m, references))

    monkeypatch.setattr(tracker, '_reference_score', score)
    monkeypatch.setattr(tracker.motion, 'update', lambda _: Motion(np.float64([[1, 0, 2], [0, 1, 0]]), 80))
    for i in range(24, 34):
        dx = 2 * (i - 23)
        moved = cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, 0]]), (640, 360))
        result = tracker.process(moved, i / 30, i)
        assert result.state == 'DETECTED'
        assert math.dist(result.center_px, (320 + dx, 180)) < 1.5
    assert tracker.keyframe['stamp'] >= 33 / 30
    if recent_score >= .9:
        assert tracker.references[-1]['stamp'] >= 31 / 30
    else:
        assert tracker.references[-1]['stamp'] < 24 / 30
    np.testing.assert_array_equal(tracker.references[0]['image'], protected)
    from ringdet.frame import Frame
    stamp = tracker.keyframe['stamp']
    tracker._stamp += 1 / 30
    tracker.dbg['measurement_shift'] = 10
    tracker._remember(Frame(moved, 2), tracker.tgt_meas)
    assert tracker.keyframe['stamp'] == stamp
    tracker._stamp = stamp + tracker.P.track.coast_s + .01
    tracker.dbg['measurement_shift'] = 0
    tracker._remember(Frame(moved, 2), tracker.tgt_meas)
    assert tracker.keyframe['stamp'] == stamp
    tracker.close()


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
    for i in range(24):
        result = tracker.process(image, i / 30, i)
    assert result.state == 'DETECTED'
    monkeypatch.setattr(tracker.motion, 'update', lambda _: None)
    shifted = cv2.warpAffine(image, np.float32([[1, 0, 70], [0, 1, 0]]), (640, 360))
    for i in range(24, 26):
        result = tracker.process(shifted, i / 30, i)
        assert result.state == 'LOST' and result.phase == 'COAST'
    result = tracker.process(shifted, 26 / 30, 26)
    assert result.state == 'DETECTED' and result.target_id == 1
    assert math.dist(result.center_px, (390, 180)) < 1.5


@pytest.mark.parametrize('seeded', [False, True])
def test_reacquire_rechecks_motion_prediction_without_waiting_for_template(monkeypatch, seeded):
    from ringdet.motion import Motion
    rng = np.random.default_rng(87)
    image = synth.ground_texture(360, 640, rng)
    synth.draw_ring(image, 320, 180, 18, 2, 75)
    image = synth.degrade(image, rng)
    tracker = RingTracker(Params(), Camera.default(640, 360))
    tracker.set_click(320, 180)
    for i in range(24):
        result = tracker.process(image, i / 30, i)
    assert result.state == 'DETECTED'
    protected = tracker.references[0]['image'].copy()
    tracker.phase = 'REACQUIRE'
    tracker.reacq = dict(start=24 / 30, x=320., y=180., r=18.)
    tracker.template_hold = tracker.P.track.template_update_hits
    if seeded:
        from ringdet.tracker import _Cand
        candidate = _Cand(tracker.tgt_meas, 24 / 30, 5)
        candidate.needed, candidate.frame = 3, tracker.n_frames
        tracker.cands = [candidate]
    monkeypatch.setattr(tracker.motion, 'update', lambda _: Motion(np.float64([[1, 0, 0], [0, 1, 0]]), 80))

    def unexpected_template(*_):
        raise AssertionError('可靠预测附近的圆应直接复查几何')

    monkeypatch.setattr(tracker, '_template_measure', unexpected_template)
    last = 25 if seeded else 26
    for i in range(24, last + 1):
        result = tracker.process(image, i / 30, i)
        assert result.state == ('DETECTED' if i == last else 'LOST')
    assert result.target_id == 1 and math.dist(result.center_px, (320, 180)) < 1.5
    np.testing.assert_array_equal(tracker.references[0]['image'], protected)
    tracker.close()


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
    tracker.P.track.stable_no_flow_s = 0
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


@pytest.mark.parametrize('radius', [6., 24., 150.])
def test_native_ray_scores_preserve_subpixel_profiles(radius):
    from ringdet import verify
    rng = np.random.default_rng(91)
    image = rng.uniform(0, 255, (480, 720)).astype(np.float32)[:, ::-1]
    params = Params().verify
    assert hasattr(edcircles._load(), 'edc_ray_scores'), '先运行 ./build.sh'
    accelerated = verify._ray_scores(image, 352.3, 237.8, radius, max(3., .15 * radius), 64, params)
    params.backend = 'python'
    reference = verify._ray_scores(image, 352.3, 237.8, radius, max(3., .15 * radius), 64, params)
    for first, second in zip(accelerated['scores'], reference['scores']):
        np.testing.assert_array_equal(first, second)


def test_native_circle_refinement_preserves_robust_fit(monkeypatch):
    from ringdet import verify
    rng = np.random.default_rng(92)
    angles = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    points = np.column_stack((132 + 45 * np.cos(angles), 87 + 45 * np.sin(angles)))
    points += rng.normal(0, .5, points.shape)
    points[::9] += rng.normal(0, 6, points[::9].shape)
    assert edcircles.huber_circle(points, (130., 89.), 43., 1., 10) is not None, '先运行 ./build.sh'
    accelerated = verify._huber_circle(points, (130., 89.), 43., 1.)
    reference = verify._huber_circle(points, (130., 89.), 43., 1., native=False)
    np.testing.assert_allclose(accelerated[0], reference[0], atol=1e-9)
    assert accelerated[1] == pytest.approx(reference[1], abs=1e-9)
    monkeypatch.setattr(verify, 'huber_circle', lambda *_: None)
    fallback = verify._huber_circle(points, (130., 89.), 43., 1.)
    np.testing.assert_array_equal(fallback[0], reference[0])
    assert fallback[1] == reference[1]


@pytest.mark.parametrize('ellipse', [False, True])
def test_native_verification_preserves_decision_and_center(ellipse):
    from ringdet import verify
    rng = np.random.default_rng(97)
    image = synth.ground_texture(240, 360, rng, amp=8)
    radius = 30 if ellipse else 7
    synth.draw_ring(image, 180, 120, radius, 1.5, 85,
                    axes=(36, 24) if ellipse else None, angle_deg=28)
    image = synth.degrade(image, rng, blur=.5, noise=1).astype(np.float32)
    params = Params().verify
    accelerated = verify.measure(image, Camera.default(360, 240), 180, 120, radius, 2, params, mode='track')
    params.backend = 'python'
    reference = verify.measure(image, Camera.default(360, 240), 180, 120, radius, 2, params, mode='track')
    assert (accelerated.ok, accelerated.reason, accelerated.sectors, accelerated.max_gap,
            accelerated.n_inliers) == (reference.ok, reference.reason, reference.sectors,
                                       reference.max_gap, reference.n_inliers)
    np.testing.assert_allclose((accelerated.cx, accelerated.cy, accelerated.r, accelerated.rms),
                               (reference.cx, reference.cy, reference.r, reference.rms), atol=1e-7)


def test_cached_ellipse_samples_preserve_rng_sequence():
    from ringdet.verify import _ellipse_samples
    for n in (24, 64, 128):
        generator = np.random.default_rng(93)
        generator.integers(0, 17)
        state = generator.bit_generator.state
        expected = np.array([generator.choice(n, 5, replace=False) for _ in range(60)], np.int32)
        for _ in range(2):
            samples, end_state = _ellipse_samples(n, state['state']['state'], state['state']['inc'],
                                                 state['has_uint32'], state['uinteger'])
            np.testing.assert_array_equal(samples, expected)
            assert end_state == generator.bit_generator.state


def test_cpu_compact_matches_preserve_original_peaks():
    rng = np.random.default_rng(94)
    roi = rng.integers(0, 256, (180, 240), dtype=np.uint8)
    templates = [roi[40:65, 55:80].copy(), np.full((19, 19), 128, np.uint8)]
    params = Params()
    params.motion.backend = params.track.template_backend = 'cpu'
    tracker = RingTracker(params)
    maps = tracker._match_templates(roi, templates)
    summaries = tracker._match_templates(roi, templates, peak_radius=8)
    for scores, (mean, deviation, peaks) in zip(maps, summaries):
        expected_mean, expected_std = cv2.meanStdDev(scores)
        assert mean == expected_mean[0, 0] and deviation == expected_std[0, 0]
        expected = []
        for _ in range(2):
            _, value, _, point = cv2.minMaxLoc(scores)
            if value < params.track.template_score:
                break
            expected.append((value, point))
            cv2.circle(scores, point, 8, -1, -1)
        assert peaks == expected


@pytest.mark.parametrize('case', ['normal', 'tie', 'unsupported', 'nonfinite', 'low_variance'])
def test_gpu_compact_matching_preserves_peaks_and_fallback(monkeypatch, case):
    uploads, downloads = [], []

    class Device:
        def __init__(self, array=None):
            self.array, self.response = array, False

        def upload(self, array):
            uploads.append(array.shape)
            self.array = array.copy()

        def download(self):
            downloads.append(self.response)
            return self.array.copy()

        def size(self):
            return self.array.shape[1], self.array.shape[0]

        def create(self, rows, cols, _type):
            self.array = np.empty((rows, cols), np.uint8)

        def setTo(self, value, mask=None):
            if mask is None:
                self.array.fill(value)
            else:
                self.array[mask.array != 0] = value

        def rowRange(self, first, last):
            return Device(self.array[first:last])

        def colRange(self, first, last):
            return Device(self.array[:, first:last])

    class Matcher:
        def match(self, image, template, result):
            result.array = cv2.matchTemplate(image.array, template.array, cv2.TM_CCOEFF_NORMED)
            result.response = True
            if case == 'nonfinite':
                result.array[0, :3] = (np.nan, np.inf, -np.inf)
            if case == 'tie':
                result.array[40, 55] = result.array[110, 170] = .999
            if case == 'low_variance':
                result.array = .9 + .001 * result.array
            return result

    def in_range(src, lo, hi):
        return Device(np.where((src.array >= lo[0]) & (src.array <= hi[0]), 255, 0).astype(np.uint8))

    def total(src):
        if case == 'unsupported':
            raise cv2.error('unsupported reduction')
        return (float(np.sum(src.array, dtype=np.float64)), 0, 0, 0)

    monkeypatch.setattr(cv2, 'cuda_GpuMat', Device)
    operations = dict(getCudaEnabledDeviceCount=lambda: 1, createTemplateMatching=lambda *_: Matcher(),
        sum=total, sqrSum=lambda src: (float(np.sum(src.array.astype(np.float64) ** 2)), 0, 0, 0),
        minMaxLoc=lambda src, mask=None: cv2.minMaxLoc(src.array, mask=None if mask is None else mask.array),
        inRange=in_range, bitwise_not=lambda src: Device(cv2.bitwise_not(src.array)),
        bitwise_and=lambda a, b: Device(cv2.bitwise_and(a.array, b.array)),
        countNonZero=lambda src: cv2.countNonZero(src.array))
    for name, operation in operations.items():
        monkeypatch.setattr(cv2.cuda, name, operation, raising=False)
    rng = np.random.default_rng(95)
    roi = rng.integers(0, 256, (180, 240), dtype=np.uint8)
    template = roi[40:65, 55:80].copy()
    copied = template.astype(np.int16)
    if case != 'tie':
        copied += rng.integers(-25, 26, copied.shape)
    roi[110:135, 170:195] = np.clip(copied, 0, 255).astype(np.uint8)
    params = Params()
    params.motion.backend = 'cpu'
    params.track.template_backend = 'cuda'
    tracker = RingTracker(params)
    expected = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
    if case == 'nonfinite':
        expected[0, :3] = 0
    if case == 'tie':
        expected[40, 55] = expected[110, 170] = .999
    if case == 'low_variance':
        expected = .9 + .001 * expected
    mean, deviation = cv2.meanStdDev(expected)
    expected_peaks = []
    for _ in range(2):
        _, value, _, point = cv2.minMaxLoc(expected)
        expected_peaks.append((value, point))
        cv2.circle(expected, point, 8, -1, -1)
    for _ in range(2):
        summaries = tracker._match_templates(roi, [template, template.copy()], peak_radius=8)
        assert tracker.template_backend == 'cuda'
        for actual_mean, actual_std, peaks in summaries:
            assert actual_mean == pytest.approx(mean[0, 0], abs=1e-8)
            assert actual_std == pytest.approx(deviation[0, 0], abs=1e-8)
            assert peaks == expected_peaks
    assert uploads.count(template.shape) == 1
    assert sum(downloads) == (4 if case in ('tie', 'unsupported', 'low_variance') else 0)
    if case == 'normal':
        tracker.warmup()
        assert set(tracker._gpu_peak_routes) == {0, 1, 2}
    tracker._gpu_peak_routes = {0: False, 1: False, 2: False}
    summary = tracker._match_templates(roi, [template], peak_radius=8)[0]
    assert summary[2] == expected_peaks
    assert tracker.template_peak_backend == 'cpu'
    tracker.close()


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


@pytest.mark.skipif(cv2.cuda.getCudaEnabledDeviceCount() == 0 or not hasattr(cv2.cuda, 'createTemplateMatching'),
    reason='需要 CUDA 设备和 OpenCV CUDA 模板匹配模块')
def test_cuda_compact_peaks_preserve_downloaded_response():
    rng = np.random.default_rng(96)
    roi = rng.integers(0, 256, (180, 240), dtype=np.uint8)
    template = roi[40:65, 55:80].copy()
    params = Params()
    params.motion.backend = 'cpu'
    params.track.template_backend = 'cuda'
    tracker = RingTracker(params)
    tracker.warmup()
    response = tracker._match_templates(roi, [template])[0]
    expected_mean, expected_std = cv2.meanStdDev(response)
    _, expected_score, _, expected_point = cv2.minMaxLoc(response)
    mean, deviation, peaks = tracker._match_templates(roi, [template], peak_radius=8)[0]
    assert mean == pytest.approx(expected_mean[0, 0], abs=1e-5)
    assert deviation == pytest.approx(expected_std[0, 0], abs=1e-5)
    assert peaks[0][1] == expected_point
    assert peaks[0][0] == pytest.approx(expected_score, abs=1e-5)
    tracker.close()


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
