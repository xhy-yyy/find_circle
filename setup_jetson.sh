#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

usage() {
  cat <<'HELP'
用法：./setup_jetson.sh [--jobs N] [--power keep|25w|max] [--offline] [--check]

为 Jetson Orin 配置项目专用 CUDA OpenCV 环境并编译、验证项目。
需要已安装 JetPack CUDA 开发工具，首次配置需要联网和 sudo 权限。

  --jobs N       编译并行数，默认 2
  --power keep   保持现有功耗模式（默认）
  --power 25w    切换到系统已开放的官方 25W 档
  --power max    切换到系统已开放的官方 MAXN_SUPER 档
  --check        只检查已安装环境
  --offline      使用已安装的离线环境，不联网或编译 OpenCV

脚本不更新启动固件，也不会自动重启。
HELP
}

die() { echo "错误：$*" >&2; exit 1; }
jobs=2
power=keep
check=0
offline=0
while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --jobs) (($# >= 2)) || die '--jobs 缺少数值'; jobs="$2"; shift 2 ;;
    --power) (($# >= 2)) || die '--power 缺少档位'; power="$2"; shift 2 ;;
    --check) check=1; shift ;;
    --offline) offline=1; shift ;;
    *) die "未知选项：$1" ;;
  esac
done
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || die '--jobs 必须为正整数'
case "$power" in keep|25w|max) ;; *) die '--power 只能是 keep、25w 或 max' ;; esac
[[ "$(uname -m)" == aarch64 ]] && { [[ -r /etc/nv_tegra_release ]] || [[ -r /etc/nv_boot_control.conf ]]; } \
  || die '此脚本需要在 Jetson Orin 上运行'
model="$(tr -d '\0' </proc/device-tree/model)"
[[ "$model" == *Orin* ]] || die "此脚本面向 Jetson Orin，当前设备：$model"
((EUID != 0)) || die '请以普通用户运行，脚本会按需调用 sudo'

repo="$PWD"
version=4.14.0
cache="$HOME/.cache/find_circle/opencv-cuda-$version"
prefix="$HOME/.local/opt/opencv-cuda-$version"
if [[ -f "/opt/find-circle/opencv-cuda-$version/python/cv2/config.py" ]]; then
  prefix="/opt/find-circle/opencv-cuda-$version"
fi
venv="$repo/.venv-cuda"
system_python=/usr/bin/python3
cuda_root="$(readlink -f /usr/local/cuda 2>/dev/null || true)"
[[ -x "$cuda_root/bin/nvcc" ]] || die '缺少 /usr/local/cuda/bin/nvcc，请先安装当前 JetPack 对应的 CUDA 开发工具'

cuda_ready() {
  [[ -x "$venv/bin/python3" ]] || return 1
  "$venv/bin/python3" - "$prefix" <<'PY'
import sys
from pathlib import Path
import cv2
assert Path(sys.argv[1]).resolve() in Path(cv2.__file__).resolve().parents, '未加载项目专用 OpenCV'
assert cv2.cuda.getCudaEnabledDeviceCount() > 0, 'OpenCV 没有可用的 CUDA 设备'
assert (hasattr(cv2, 'cuda_SparsePyrLKOpticalFlow') or
        hasattr(cv2.cuda, 'SparsePyrLKOpticalFlow_create')), '缺少 CUDA 光流模块'
assert hasattr(cv2.cuda, 'createGoodFeaturesToTrackDetector'), '缺少 CUDA 角点模块'
assert hasattr(cv2.cuda, 'createTemplateMatching'), '缺少 CUDA 模板匹配模块'
print('OpenCV:', cv2.__version__, cv2.__file__)
print('CUDA 设备数:', cv2.cuda.getCudaEnabledDeviceCount())
PY
}

echo "设备：$model"
if ((check)); then
  cuda_ready || die '项目 CUDA 环境未就绪，请先运行 ./setup_jetson.sh'
  /usr/sbin/nvpmodel -q
  exit 0
fi

avail_gb=$(( $(df -Pk "$HOME" | awk 'NR==2 {print $4}') / 1024 / 1024 ))
(( avail_gb >= 15 )) || echo "警告：$HOME 剩余约 ${avail_gb} GB，编译 OpenCV CUDA 建议至少 15 GB" >&2

if ((offline == 0)); then
sudo -v
sudo env DEBIAN_FRONTEND=noninteractive apt-get update
sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  build-essential cmake pkg-config curl ca-certificates \
  python3-dev python3-venv python3-numpy python3-yaml python3-pytest \
  libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  libtbb-dev libjpeg-dev libpng-dev ffmpeg v4l-utils psmisc
fi
mkdir -p "$cache"
((offline)) || mkdir -p "$prefix"
if [[ ! -x "$venv/bin/python3" ]]; then
  "$system_python" -m venv --system-site-packages "$venv"
fi
site="$("$venv/bin/python3" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
printf '%s\n' "$prefix/python" >"$site/opencv_cuda.pth"

if cuda_ready >/dev/null 2>&1; then
  echo '复用已安装的项目 CUDA OpenCV。'
else
  ((offline == 0)) || die '离线 OpenCV 不可用，请先安装与当前 JetPack、CUDA 和 Python 匹配的 deb 及依赖'
  ffmpeg_args=()
  if [[ -f /usr/lib/aarch64-linux-gnu/libavcodec.so.60 ]] && \
      [[ "$(pkg-config --modversion libavcodec)" == 62.* ]]; then
    sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libavdevice60
    mkdir -p "$cache/ffmpeg6"
    "$system_python" - "$cache/ffmpeg6" <<'PY'
from pathlib import Path
import re
import subprocess
import sys
root = Path(sys.argv[1])
for package in ('libavcodec-dev', 'libavformat-dev', 'libavutil-dev', 'libswscale-dev',
                'libavdevice-dev', 'libavfilter-dev', 'libswresample-dev', 'libpostproc-dev'):
    if not list(root.glob(package + '_*.deb')):
        policy = subprocess.check_output(['apt-cache', 'policy', package], text=True)
        version = re.search(r'7:6\.1\.1[^\s]*', policy).group(0)
        subprocess.run(['apt-get', 'download', package + '=' + version], cwd=root, check=True)
for archive in root.glob('*.deb'):
    subprocess.run(['dpkg-deb', '-x', str(archive), str(root / 'root')], check=True)
libraries = (('avcodec', '60', '60.31.102'), ('avformat', '60', '60.16.100'),
             ('avutil', '58', '58.29.100'), ('swscale', '7', '7.5.100'))
lines = ['set(FFMPEG_FOUND TRUE)',
         f'set(FFMPEG_INCLUDE_DIRS "{root}/root/usr/include/aarch64-linux-gnu")',
         'set(FFMPEG_LIBRARIES "' + ';'.join('/usr/lib/aarch64-linux-gnu/lib' + name + '.so.' + major
                                             for name, major, _ in libraries) + '")']
lines += ['set(FFMPEG_lib' + name + '_VERSION ' + version + ')' for name, _, version in libraries]
lines += ['set(FFMPEG_libavdevice_FOUND TRUE)', 'set(FFMPEG_libavdevice_VERSION 60.3.100)',
          'list(APPEND FFMPEG_LIBRARIES /usr/lib/aarch64-linux-gnu/libavdevice.so.60)']
(root / 'FFMPEGConfig.cmake').write_text('\n'.join(lines) + '\n')
PY
    ffmpeg_args=(-DOPENCV_FFMPEG_USE_FIND_PACKAGE=ON "-DFFMPEG_DIR=$cache/ffmpeg6")
  fi
  for name in opencv opencv_contrib; do
    src="$cache/$name-$version"
    archive="$cache/$name-$version.tar.gz"
    if [[ ! -d "$src" ]]; then
      if [[ ! -f "$archive" ]]; then
        url="https://codeload.github.com/opencv/$name/tar.gz/refs/tags/$version"
        if ! curl -fL --retry 3 --connect-timeout 15 "$url" -o "$archive.part"; then
          rm -f "$archive.part"
          die "下载失败：$url
可手动下载后放到 $archive（文件名保持一致）"
        fi
        mv "$archive.part" "$archive"
      fi
      tar -xzf "$archive" -C "$cache"
    fi
  done
  numpy_include="$("$system_python" -c 'import numpy; print(numpy.get_include())')"
  cmake -S "$cache/opencv-$version" -B "$cache/build" \
    "${ffmpeg_args[@]}" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$prefix" \
    -DCMAKE_INSTALL_RPATH="$prefix/lib" -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
    -DOPENCV_EXTRA_MODULES_PATH="$cache/opencv_contrib-$version/modules" \
    -DBUILD_LIST=core,imgproc,imgcodecs,video,videoio,highgui,calib3d,features2d,flann,python3,cudev,cudaarithm,cudafilters,cudaimgproc,cudaoptflow,cudawarping,ximgproc \
    -DWITH_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR="$cuda_root" -DCUDA_ARCH_BIN=8.7 -DCUDA_ARCH_PTX= \
    -DWITH_CUDNN=OFF -DOPENCV_DNN_CUDA=OFF -DWITH_CUBLAS=ON \
    -DENABLE_FAST_MATH=OFF -DCUDA_FAST_MATH=OFF \
    -DWITH_TBB=ON -DWITH_OPENMP=OFF -DWITH_GTK=ON -DWITH_QT=OFF \
    -DWITH_FFMPEG=ON -DWITH_GSTREAMER=ON -DWITH_V4L=ON -DWITH_OPENCL=OFF \
    -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_JAVA=OFF \
    -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=ON \
    -DPYTHON3_EXECUTABLE="$system_python" -DPYTHON3_NUMPY_INCLUDE_DIRS="$numpy_include" \
    -DOPENCV_PYTHON3_INSTALL_PATH="$prefix/python" -DOPENCV_GENERATE_PKGCONFIG=ON
  grep -q '^#define HAVE_CUDA' "$cache/build/cvconfig.h" || die 'OpenCV 配置未启用 CUDA，请检查 CUDA 开发包是否完整'
  cmake --build "$cache/build" --parallel "$jobs"
  cmake --install "$cache/build"
fi
cuda_ready

cmake -S "$repo" -B "$repo/build" -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR="$prefix/lib/cmake/opencv4"
cmake --build "$repo/build" --parallel "$jobs"

"$venv/bin/python3" - <<'PY'
from ringdet import Params, RingTracker
p = Params.load('orin_nano_8gb.yaml')
p.motion.backend = 'cuda'
p.track.template_backend = 'cuda'
tracker = RingTracker(p)
try:
    tracker.warmup()
    assert tracker.motion.backend == tracker.template_backend == 'cuda'
    print('GPU 光流、角点和模板匹配预热通过。')
finally:
    tracker.close()
PY
"$venv/bin/python3" -m pytest -q tests/test_smoke.py \
  -k 'cuda_motion_tracks_translation or cuda_ncc_matches_cpu or cuda_compact_peaks_preserve_downloaded_response'

if [[ "$power" != keep ]]; then
  name=25W
  [[ "$power" == max ]] && name=MAXN_SUPER
  mode="$("$system_python" - "$name" <<'PY'
import pathlib, re, sys
text = pathlib.Path('/etc/nvpmodel.conf').read_text()
for mode, name in re.findall(r'<\s*POWER_MODEL\s+ID=(\d+)\s+NAME=([^\s>]+)', text):
    if name == sys.argv[1]:
        print(mode)
        break
PY
)"
  if [[ -z "$mode" ]]; then
    echo "CUDA 环境已就绪；当前启动配置未开放 $name，功耗模式未切换。" >&2
    echo '需先配置官方 Super 启动固件并重启，脚本不会自动执行。' >&2
    exit 2
  fi
  sudo -v
  if power_output="$(sudo /usr/sbin/nvpmodel -m "$mode" </dev/null 2>&1)"; then
    echo "$power_output"
  else
    echo "$power_output" >&2
    die 'CUDA 环境已就绪，但功耗切换失败'
  fi
  if [[ "${power_output,,}" == *'reboot required'* ]]; then
    echo 'CUDA 环境已就绪；功耗切换需要重启，本次未重启。' >&2
    exit 2
  fi
fi
/usr/sbin/nvpmodel -q
echo '配置完成。run.sh 将自动使用 .venv-cuda，系统 Python 的 OpenCV 保留。'
echo '有显示器：SHOW=1 ./run.sh camera --stats --quiet'
echo '无显示器：UDP_OUT=<地面站IP>:9000 ./run.sh camera --stats --quiet'
