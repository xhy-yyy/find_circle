"""调试画图。"""
from __future__ import annotations

import cv2

GREEN, ORANGE, RED, CYAN, YELLOW, GRAY = (0, 220, 0), (0, 160, 255), (0, 0, 255), (255, 255, 0), (0, 255, 255), (150, 150, 150)
SHIFT = 4   # 亚像素画图


def _p(x, y):
    return int(round(x * (1 << SHIFT))), int(round(y * (1 << SHIFT)))


def circle(img, x, y, r, color, th=1):
    cv2.circle(img, _p(x, y), int(round(r * (1 << SHIFT))), color, th, cv2.LINE_AA, SHIFT)


def cross(img, x, y, size, color, th=1):
    cv2.line(img, _p(x - size, y), _p(x + size, y), color, th, cv2.LINE_AA, SHIFT)
    cv2.line(img, _p(x, y - size), _p(x, y + size), color, th, cv2.LINE_AA, SHIFT)


def text(img, s, org, color=(255, 255, 255), scale=0.6):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_measurement(img, m, color):
    if m.ellipse is not None:
        cx, cy, a, b, ang = m.ellipse
        cv2.ellipse(img, ((cx, cy), (2 * a, 2 * b), ang), color, 1, cv2.LINE_AA)
    else:
        circle(img, m.cx, m.cy, m.r, color)
    cross(img, m.cx, m.cy, max(3, m.r * 0.3), color)


def draw_result(frame, res):
    img = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    d = res.debug
    if res.phase == 'SEARCH' and res.click_px is not None and d.get('region') is not None:
        x0, y0, x1, y1 = d['region']
        if x1 > x0 and y1 > y0:
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1) - 1, int(y1) - 1), YELLOW, 1)
    if res.center_px is not None:
        x, y = res.center_px
        color = GREEN if res.state == 'DETECTED' else ORANGE
        if res.ellipse is not None:
            cx, cy, a, b, ang = res.ellipse
            cv2.ellipse(img, ((cx, cy), (2 * a, 2 * b), ang), color, 1, cv2.LINE_AA)
        else:
            circle(img, x, y, res.radius_px, color)
        cross(img, x, y, max(4, res.radius_px * 0.3), color)
    color = {'DETECTED': GREEN, 'PREDICTED': ORANGE}.get(res.state, RED)
    s = f'#{res.frame_id}  {res.state} / {res.phase}  ID={res.target_id}'
    if d.get('prediction_source') == 'appearance':
        s += '  KEYFRAME'
    if res.ambiguous:
        s += '  AMBIGUOUS'
    lines = [(s, color)]
    if res.center_px is not None:
        position = f'Center: ({res.center_px[0]:.1f}, {res.center_px[1]:.1f})  R: {res.radius_px:.1f}px'
    else:
        position = 'Center: --  R: --'
    if res.age_ms is not None:
        position += f'  Last seen: {res.age_ms:.0f}ms'
    lines.append((position, (255, 255, 255)))
    fps = f"{d['fps']:.1f}" if d.get('fps', 0) > 0 else '--'
    timing = f"Process FPS: {fps}  Core: {d.get('ms', 0):.1f}ms"
    if 'frame_age_ms' in d:
        timing += f"  Frame age: {d['frame_age_ms']:.0f}ms"
    elif 'source_fps' in d:
        timing += f"  Video: {d['source_fps']:.1f}fps"
    lines.append((timing, (255, 255, 255)))
    if res.state == 'DETECTED':
        geometry = (f"Geometry: sectors={res.sectors}/{d.get('verify_sectors', 16)}"
                    f'  RMS={res.rms_px:.2f}px  Width={res.line_w_px:.1f}px')
        lines.append((geometry, (255, 255, 255)))
    flow = 'OK' if res.motion_ok else 'FAIL'
    scope = 'local' if d.get('local_motion') else 'global'
    lines.append((f"Flow: {flow} {scope}/{d.get('motion_backend', '--')}"
                  f"  Inliers={d.get('motion_inliers', 0)}  Err={d.get('motion_error', 0):.1f}px",
                  (255, 255, 255) if res.motion_ok else ORANGE))
    score = d.get('target_identity_score') if res.state == 'DETECTED' else d.get('identity_score')
    identity = f'{score:.2f}' if score is not None else '--'
    label = 'Identity' if res.state == 'DETECTED' else 'Last ID check'
    match = d.get('appearance_ncc', d.get('template_score'))
    ncc = f'{match:.2f}' if match is not None else '--'
    psr = d.get('appearance_psr', d.get('online_psr', d.get('online_update_psr')))
    psr = f'{psr:.1f}' if psr is not None else '--'
    lines.append((f"{label}: {identity}  NCC: {ncc}  PSR: {psr}  Refs: {d.get('reference_count', 0)}",
                  (255, 255, 255)))
    age = d.get('online_age_frames')
    if not d.get('online_enabled'):
        learning = 'OFF'
    elif d.get('online_active'):
        learning = 'UPDATED' if age == 0 else 'FROZEN' if res.phase != 'TRACK' or d.get('template_hold') else 'READY'
    elif not d.get('reference_count'):
        learning = 'WAIT LOCK'
    elif res.radius_px is not None and res.radius_px < d.get('online_min_radius', 8):
        learning = 'WAIT SIZE'
    else:
        learning = 'WAIT QUALITY'
    lines.append((f"Online CF: {learning}  Updates={d.get('online_updates', 0)}"
                  f"  Age={str(age) + 'f' if age is not None else '--'}  Hold={d.get('template_hold', 0)}",
                  (255, 255, 255)))
    if res.phase in ('SEARCH', 'REACQUIRE'):
        search = f"Search: raw={len(d.get('found', []))}  candidates={len(d.get('cands', []))}"
        confirmation = d.get('confirmation')
        if confirmation is not None:
            hits, needed, ed = confirmation
            search += f"  Confirm={hits}/{needed}  ED={'yes' if ed else 'no'}"
        if d.get('r_range') is not None:
            lo, hi = d['r_range']
            search += f'  R={lo:.1f}..{hi:.1f}'
        lines.append((search, YELLOW))
    check = d.get('track_meas')
    if res.ambiguous:
        lines.append(('Check: multiple similar candidates', ORANGE))
    elif check is not None and not check.ok:
        lines.append((f'Track check: {check.reason}', ORANGE))
    elif d.get('rejected'):
        reason = max(d['rejected'], key=d['rejected'].get)
        lines.append((f'Search check: {reason}', ORANGE))
    elif res.phase == 'SEARCH' and res.click_px is None and not d.get('reference_count'):
        lines.append(('Waiting for click', YELLOW))
    H, W = img.shape[:2]
    scale = max(0.6, W / 1280)
    font = 0.48 * scale
    margin, gap = int(round(10 * scale)), int(round(21 * scale))
    widths = [cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, font, 1)[0][0] for s, _ in lines]
    panel_w, panel_h = min(W, max(widths) + 2 * margin), min(H, len(lines) * gap + margin)
    panel = img[:panel_h, :panel_w]
    cv2.addWeighted(panel, 0.35, panel * 0, 0.65, 0, dst=panel)
    for row, (s, tint) in enumerate(lines):
        text(img, s, (margin, margin + (row + 1) * gap - int(5 * scale)), tint, font)
    return img
