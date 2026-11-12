#!/usr/bin/env bash
# find_circle 统一入口。用法见 ./run.sh help 或 docs/使用教程.md
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

# ---------------- 可修改的配置（也可以用同名环境变量覆盖，例如 DEV=/dev/video1 ./run.sh camera） ----------------
DEV="${DEV:-/dev/video0}"              # 相机设备
SIZE="${SIZE:-1920x1080}"              # 分辨率
CALIB="${CALIB:-}"                     # 相机标定 yaml（ROS camera_info 格式）；空 = 未标定
EXPOSURE="${EXPOSURE:-keep}"           # keep = 保持相机控制项；auto = 自动曝光；数字单位 100 us
UDP_OUT="${UDP_OUT:-}"                 # 结果发到哪里，如 192.168.1.10:9000；空 = 打印到屏幕
LISTEN="${LISTEN:-9001}"               # 接收点击/高度/重置的端口
REC_DIR="${REC_DIR-$HOME/find_circle_rec}"   # 机载运行时录像和结果日志的目录；空 = 不录
TARGET="${TARGET:-127.0.0.1}"          # click/height/reset 命令发往的视觉机地址
TEST_IMAGES="${TEST_IMAGES:-$HOME/桌面/无人机下视圆环图片}"
# ------------------------------------------------------------------------------------------------------------

if [[ -r /proc/device-tree/model ]] &&
   [[ $(tr -d '\000' </proc/device-tree/model) == *"Jetson Orin Nano"* ]]; then
  FPS="${FPS:-60}"
  if [[ ! -v PARAMS && -f orin_nano_8gb.yaml ]]; then
    PARAMS=orin_nano_8gb.yaml
  fi
fi
FPS="${FPS:-30}"
PARAMS="${PARAMS:-}"                   # 显式置空使用通用参数
if [[ -z "${PY:-}" && -x .venv-cuda/bin/python3 ]]; then
  PY="$PWD/.venv-cuda/bin/python3"
fi
PY="${PY:-python3}"
STAMP="$(date +%Y%m%d_%H%M%S)"
PARAM_ARGS=()
[[ -n "$PARAMS" ]] && PARAM_ARGS=(--params "$PARAMS")

die() { echo "错误：$*" >&2; exit 1; }

need_lib() {
  [[ -f ringdet/_lib/libedcircles.so ]] && return
  echo "还没编译 ED_Lib，先编译……"
  ./build.sh >/dev/null
}

has_arg() {   # has_arg --out "$@"：参数里有没有某个选项
  local k="$1"; shift
  for x in "$@"; do [[ "$x" == "$k" || "$x" == "$k="* ]] && return 0; done
  return 1
}

cmd_help() {
  cat <<HELP
用法：./run.sh <命令> [参数]

  setup                       检查依赖、编译 ED_Lib、跑测试（第一次先运行这个）
  image [图片或文件夹 ...]     测图片；不给路径时测 $TEST_IMAGES。其余参数传给 tools/run_image.py
  video <视频> [x,y]           测视频，x,y 是第 0 帧的点击位置（原图像素）；SHOW=1 时弹窗显示，可用鼠标点
                              --no-record 跳过标注编码；Nano 预览默认启用，--out 指定文件可保存
  camera                       机载运行：设置相机、识别、发结果、录像（配置见脚本开头）
  click <u> <v>                发点击，u、v 为 0~1 的归一化坐标（相对原始图像宽高）
  height <米|null>              发离地高度
  reset                        清除点击和目标
  listen [端口]                 接收并显示视觉的输出（默认 9000），用来联调
  clean [--dry-run]             清理相机测试录像；--dry-run 只预览，正在使用的录像跳过
  eval                         回归测试：冒烟测试、点击评估、合成下降

例子：
  ./run.sh image ~/桌面/无人机下视圆环图片/黄沙与浅色场地_100米俯视
  SHOW=1 ./run.sh video 降落.mp4 1160,600
  UDP_OUT=192.168.1.10:9000 CALIB=cam.yaml ./run.sh camera
  TARGET=192.168.1.20 ./run.sh click 0.52 0.47
HELP
}

cmd_setup() {
  local miss=()
  command -v cmake >/dev/null || miss+=(cmake)
  command -v g++ >/dev/null || miss+=(g++)
  pkg-config --exists opencv4 2>/dev/null || pkg-config --exists opencv 2>/dev/null || miss+=("OpenCV C++ 开发包")
  for m in cv2 numpy yaml pytest; do
    "$PY" -c "import $m" 2>/dev/null || miss+=("python3 模块 $m")
  done
  if ((${#miss[@]})); then
    echo "缺少：${miss[*]}"
    echo "Ubuntu：sudo apt install cmake g++ libopencv-dev python3-opencv python3-numpy python3-yaml python3-pytest"
    echo "Jetson：OpenCV 随 JetPack 安装，不要另装 Ubuntu 源的 OpenCV；其余用上面的命令补齐"
    exit 1
  fi
  ./build.sh
  "$PY" -m ringdet.edcircles 2>&1 | grep -v RuntimeWarning | grep -v "warn(" || true
  "$PY" -m pytest -q tests/
  echo "安装完成。下一步：./run.sh image"
}

cmd_image() {
  need_lib
  local args=("$@")
  if ((${#args[@]} == 0)); then
    [[ -d "$TEST_IMAGES" ]] || die "没给图片路径，默认目录 $TEST_IMAGES 也不存在"
    args=("$TEST_IMAGES")
  fi
  local out="out/images_$STAMP"
  has_arg --out "${args[@]}" || args+=(--out "$out")
  "$PY" tools/run_image.py "${PARAM_ARGS[@]}" "${args[@]}"
}

cmd_video() {
  need_lib
  (($# >= 1)) || die "用法：./run.sh video <视频> [x,y]"
  local video="$1"; shift
  [[ -f "$video" ]] || die "找不到视频 $video"
  local args=()
  if (($# >= 1)) && [[ "$1" =~ ^[0-9.]+,[0-9.]+(,[0-9]+)?$ ]]; then
    args+=(--click "$1"); shift
  fi
  local name; name="$(basename "${video%.*}")"
  local out="out/video_${name}_$STAMP"
  if ! has_arg --out "$@"; then
    if ! has_arg --no-record "$@"; then
      if [[ "${PARAMS##*/}" == orin_nano_8gb.yaml ]] && { [[ "${SHOW:-0}" == 1 ]] || has_arg --show "$@"; }; then
        args+=(--no-record)
      else
        args+=(--out "$out.mp4")
      fi
    fi
    has_arg --jsonl "$@" || args+=(--jsonl "$out.jsonl")
  fi
  [[ "${SHOW:-0}" == 1 ]] && args+=(--show)
  "$PY" tools/run_video.py "$video" "${PARAM_ARGS[@]}" "${args[@]}" "$@"
}

set_camera() {
  if [[ "$EXPOSURE" == keep ]]; then
    echo "保持摄像头原有曝光和对焦设置"
    return
  fi
  if ! command -v v4l2-ctl >/dev/null; then
    echo "[提示] 没有 v4l2-ctl（sudo apt install v4l-utils），相机参数保持原样"
    return
  fi
  # 不同相机、不同内核的控件名不一样，两种写法都试，失败不影响运行
  v4l2-ctl -d "$DEV" -c focus_automatic_continuous=0 2>/dev/null || v4l2-ctl -d "$DEV" -c focus_auto=0 2>/dev/null \
    || echo "[提示] 没能关闭自动对焦（相机可能是定焦）"
  if [[ "$EXPOSURE" == auto ]]; then
    { v4l2-ctl -d "$DEV" -c auto_exposure=3 2>/dev/null \
      || v4l2-ctl -d "$DEV" -c exposure_auto=3 2>/dev/null \
      || v4l2-ctl -d "$DEV" -c auto_exposure=0 2>/dev/null \
      || v4l2-ctl -d "$DEV" -c exposure_auto=0 2>/dev/null; } \
      && echo "已开启自动曝光" \
      || echo "[提示] 没能开启自动曝光，用 v4l2-ctl -d $DEV -l 查看控件名"
  else
    { v4l2-ctl -d "$DEV" -c auto_exposure=1 -c exposure_time_absolute="$EXPOSURE" 2>/dev/null \
      || v4l2-ctl -d "$DEV" -c exposure_auto=1 -c exposure_absolute="$EXPOSURE" 2>/dev/null; } \
      && echo "曝光设为 $EXPOSURE x 100 us" \
      || echo "[提示] 没能设置曝光，用 v4l2-ctl -d $DEV -l 查看控件名"
  fi
}

cmd_camera() {
  need_lib
  [[ -e "$DEV" ]] || die "找不到相机 $DEV（ls /dev/video* 看看有哪些）"
  [[ -z "$CALIB" || -f "$CALIB" ]] || die "找不到标定文件 $CALIB"
  set_camera
  local args=(--dev "$DEV" --size "$SIZE" --fps "$FPS" --listen "$LISTEN")
  [[ -n "$CALIB" ]] && args+=(--calib "$CALIB")
  [[ -n "$UDP_OUT" ]] && args+=(--udp "$UDP_OUT")
  if [[ -n "$REC_DIR" ]]; then
    mkdir -p "$REC_DIR"
    args+=(--record "$REC_DIR/raw_$STAMP.avi" --log "$REC_DIR/result_$STAMP.jsonl")
    echo "录像：$REC_DIR/raw_$STAMP.avi    结果日志：$REC_DIR/result_$STAMP.jsonl"
  fi
  [[ "${SHOW:-0}" == 1 ]] && args+=(--show)
  echo "输出：${UDP_OUT:-屏幕}    输入端口：$LISTEN    Ctrl+C 结束"
  exec "$PY" -u tools/run_camera.py "${args[@]}" "${PARAM_ARGS[@]}" "$@"
}

send() {   # send <click_norm u v | height h|null | reset>：在 Python 里检查数值、生成 JSON
  "$PY" - "$TARGET" "$LISTEN" "$@" <<'PYEOF'
import json, socket, sys
host, port, kind, vals = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4:]
try:
    if kind == 'click_norm':
        u, v = (float(x) for x in vals)
        assert 0 <= u <= 1 and 0 <= v <= 1
        msg = {'click_norm': [u, v]}
    elif kind == 'height':
        (h,) = vals
        msg = {'height': None if h == 'null' else float(h)}
        assert msg['height'] is None or msg['height'] > 0
    else:
        msg = {'reset': True}
except (ValueError, AssertionError):
    sys.exit(f'错误：参数不对：{" ".join(vals)}（点击 u、v 在 0~1 之间；高度为正数或 null）')
s = json.dumps(msg)
socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(s.encode(), (host, port))
print(f'已发送到 {host}:{port}：{s}')
PYEOF
}

cmd_click() { (($# == 2)) || die "用法：./run.sh click <u> <v>（0~1）"; send click_norm "$1" "$2"; }
cmd_height() { (($# == 1)) || die "用法：./run.sh height <米|null>"; send height "$1"; }
cmd_reset() { send reset; }

cmd_listen() {
  local port="${1:-9000}"
  echo "在 UDP $port 等视觉的输出，Ctrl+C 结束（RAW=1 显示原始 JSON）"
  "$PY" -u - "$port" "${RAW:-0}" <<'PYEOF'
import json, socket, sys
port, raw = int(sys.argv[1]), sys.argv[2] == '1'
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(('0.0.0.0', port))
f = lambda v, n=1: '-' if v is None else (f'({v[0]:.{n}f},{v[1]:.{n}f})' if isinstance(v, list) else f'{v:.{n}f}')
while True:
    data, addr = s.recvfrom(65536)
    if raw:
        print(data.decode(errors='replace'))
        continue
    try:
        d = json.loads(data)
    except ValueError:
        print('[非 JSON]', data[:80]); continue
    print(f"#{d.get('frame_id')} {str(d.get('state')):9} {str(d.get('phase')):9} id={d.get('target_id')} "
          f"圆心={f(d.get('center_px'))} r={f(d.get('radius_px'))} bearing={f(d.get('bearing'), 3)} "
          f"点击={f(d.get('click_px'), 0)} 扇区={d.get('sectors')} 延迟={f(d.get('frame_age_ms'), 0)}ms")
PYEOF
}

cmd_clean() {
  [[ -n "$REC_DIR" ]] || die "REC_DIR 为空，没有指定录像清理目录"
  "$PY" - "$REC_DIR" "$@" <<'PYEOF'
import argparse
from pathlib import Path
import re
import shutil
import subprocess

ap = argparse.ArgumentParser(prog='./run.sh clean', usage='%(prog)s [--dry-run]', description='清理相机测试录像，保留日志并跳过正在使用的文件。')
ap.add_argument('directory', help=argparse.SUPPRESS)
ap.add_argument('--dry-run', action='store_true', help='只列出待删除录像')
a = ap.parse_args()
root = Path(a.directory).expanduser().resolve()
if not root.is_dir():
    print(f'录像目录不存在：{root}')
    raise SystemExit(0)
fuser = shutil.which('fuser')
if fuser is None:
    ap.error('缺少 fuser，无法检查录像占用；请安装 psmisc 后重试')

count = size = skipped = failures = 0
print(f'清理目录：{root}')
for p in sorted(root.iterdir()):
    if p.is_symlink() or not p.is_file() or not re.fullmatch(r'raw_\d{8}_\d{6}\.avi', p.name):
        continue
    busy = subprocess.run([fuser, '-s', str(p)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if busy.returncode != 1 or busy.stderr:
        print(f'跳过（正在使用或无法确认占用）：{p.name}')
        skipped += 1
        continue
    try:
        length = p.stat().st_size
        if not a.dry_run:
            p.unlink()
    except OSError as e:
        print(f'处理失败：{p.name}：{e}')
        failures += 1
        continue
    count += 1
    size += length
    print(f'{"将删除" if a.dry_run else "已删除"}：{p.name}（{length / 1024**2:.1f} MiB）')
print(f'{"待删除" if a.dry_run else "已删除"} {count} 个视频，共 {size / 1024**2:.1f} MiB；跳过 {skipped} 个，失败 {failures} 个。')
raise SystemExit(1 if failures else 0)
PYEOF
}

cmd_eval() {
  need_lib
  echo "== 冒烟测试 =="
  "$PY" -m pytest -q tests/
  echo "== 点击评估（桌面图 + 合成图）=="
  "$PY" tools/eval_lock.py
  echo "== 点击评估（先按 JPEG 85 压缩）=="
  "$PY" tools/eval_lock.py --jpeg 85
  echo "== 合成下降视频 =="
  "$PY" tools/sim_descent.py --size 1920x1080 --r1 150 | tail -8
}

case "${1:-help}" in
  setup|image|video|camera|click|height|reset|listen|clean|eval) c="$1"; shift; "cmd_$c" "$@" ;;
  help|-h|--help) cmd_help ;;
  *) echo "未知命令：$1"; echo; cmd_help; exit 1 ;;
esac
