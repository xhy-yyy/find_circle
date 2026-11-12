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


def text(img, s, org, color=(255, 255, 255), scale=0.6, th=3):
    # 阴影用非 AA：1080p 上 7 行文字从 ~6 ms 降到 ~2 ms，正文仍是 AA
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th, cv2.LINE_8)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_measurement(img, m, color):
    if m.ellipse is not None:
        cx, cy, a, b, ang = m.ellipse
        cv2.ellipse(img, ((cx, cy), (2 * a, 2 * b), ang), color, 1, cv2.LINE_AA)
    else:
        circle(img, m.cx, m.cy, m.r, color)
    cross(img, m.cx, m.cy, max(3, m.r * 0.3), color)


def draw_result(frame, res, scale=1.0):
    """scale < 1：直接按显示分辨率绘制（预览用），省掉全图拷贝和事后缩放。"""
    if scale != 1.0:
        img = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    else:
        img = frame.copy()
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    sp = scale
    d = res.debug
    if res.phase == 'SEARCH' and res.click_px is not None and d.get('region') is not None:
        x0, y0, x1, y1 = d['region']
        if x1 > x0 and y1 > y0:
            cv2.rectangle(img, (int(x0 * sp), int(y0 * sp)), (int(x1 * sp) - 1, int(y1 * sp) - 1), YELLOW, 1)
    if res.center_px is not None:
        x, y = res.center_px[0] * sp, res.center_px[1] * sp
        color = GREEN if res.state == 'DETECTED' else ORANGE
        if res.ellipse is not None:
            cx, cy, a, b, ang = res.ellipse
            cv2.ellipse(img, ((cx * sp, cy * sp), (2 * a * sp, 2 * b * sp), ang), color, 1, cv2.LINE_AA)
        else:
            circle(img, x, y, res.radius_px * sp, color)
        cross(img, x, y, max(4, res.radius_px * 0.3) * sp, color)
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
    if res.age_ms is not None and res.state != 'DETECTED':
        position += f'  Last seen: {res.age_ms:.0f}ms'
    lines.append((position, (255, 255, 255)))
    fps = f"{d['fps']:.1f}" if d.get('fps', 0) > 0 else '--'
    timing = f"FPS: {fps}  Core: {d.get('ms', 0):.1f}ms"
    if 'frame_age_ms' in d:
        timing += f"  Frame age: {d['frame_age_ms']:.0f}ms"
    elif 'source_fps' in d:
        timing += f"  Video: {d['source_fps']:.1f}fps"
    lines.append((timing, (255, 255, 255)))
    if res.state == 'DETECTED':
        lines.append((f"Ing: sectors={res.sectors}/{d.get('verify_sectors', 16)}"
                      f'  RMS={res.rms_px:.2f}px', (255, 255, 255)))
    if res.phase in ('SEARCH', 'REACQUIRE'):
        search = f"Search: candidates={len(d.get('cands', []))}"
        confirmation = d.get('confirmation')
        if confirmation is not None:
            hits, needed, _ = confirmation
            search += f'  Confirm={hits}/{needed}'
        lines.append((search, YELLOW))
    if not res.motion_ok:
        lines.append(('Flow FAIL', ORANGE))
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
    ls = max(0.6, W / 1280)
    font = 0.48 * ls
    margin, gap = int(round(10 * ls)), int(round(21 * ls))
    widths = [cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, font, 1)[0][0] for s, _ in lines]
    panel_w, panel_h = min(W, max(widths) + 2 * margin), min(H, len(lines) * gap + margin)
    panel = img[:panel_h, :panel_w]
    cv2.multiply(panel, 0.35, dst=panel, dtype=cv2.CV_8U)
    th = max(1, int(round(3 * sp)))
    for row, (s, tint) in enumerate(lines):
        text(img, s, (margin, margin + (row + 1) * gap - int(5 * ls)), tint, font, th)
    return img
