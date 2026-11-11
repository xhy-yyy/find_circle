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
    if d.get('region') is not None:
        x0, y0, x1, y1 = d['region']
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1) - 1, int(y1) - 1), YELLOW, 1)
    for x, y, r in d.get('found', []):
        circle(img, x, y, r, GRAY)
    for x, y, r, h in d.get('cands', []):
        circle(img, x, y, r + 3, CYAN)
        text(img, str(h), (int(x + r + 4), int(y)), CYAN, 0.45)
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
    s = f'#{res.frame_id} {res.state} {res.phase} id={res.target_id}'
    if res.center_px is not None:
        s += f' c=({res.center_px[0]:.1f},{res.center_px[1]:.1f}) r={res.radius_px:.1f}'
    if res.ambiguous:
        s += ' AMBIGUOUS'
    text(img, s, (10, 25), color)
    timing = f"{d.get('ms', 0):.0f} ms  noise={d.get('noise', 0):.1f}  flow={d.get('motion_inliers', 0)}"
    if d.get('fps', 0) > 0:
        timing += f"  {d['fps']:.1f} FPS"
    text(img, timing, (10, 50))
    return img
