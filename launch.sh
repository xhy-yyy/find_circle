#!/usr/bin/env bash
# 桌面启动入口：先选输入源（摄像头 / 视频文件），再进入识别。
# 对应命令等价于 SHOW=1 ./run.sh camera 或 SHOW=1 ./run.sh video <文件>。
set -u
cd "$(dirname "$(readlink -f "$0")")" || exit 1
command -v zenity >/dev/null || { echo '缺少 zenity：sudo apt install zenity'; exit 1; }

choice=$(zenity --list --title="白环识别" --text="选择输入源" --radiolist \
  --column="" --column="输入源" --print-column=2 \
  TRUE "摄像头 1080p60" FALSE "摄像头 720p60（更快）" FALSE "视频文件" \
  --width=360 --height=220 2>/dev/null) || exit 0

case "$choice" in
  摄像头\ 1080p60)
    # 1080p：MJPG → NVDEC 硬解；识别可到约 60 fps（CUDA/CPU 视场景）
    mkdir -p "$HOME/find_circle_rec"
    FPS=60 SIZE=1920x1080 REC_DIR='' SHOW=1 ./run.sh camera \
      --log "$HOME/find_circle_rec/result_$(date +%Y%m%d_%H%M%S).jsonl"
    ;;
  摄像头\ 720p60（更快）)
    # 720p：MJPG → NVDEC 硬解，识别 55~60 fps；不录像以免拖慢主循环
    mkdir -p "$HOME/find_circle_rec"
    FPS=60 SIZE=1280x720 REC_DIR='' SHOW=1 ./run.sh camera \
      --log "$HOME/find_circle_rec/result_$(date +%Y%m%d_%H%M%S).jsonl"
    ;;
  视频文件)
    file=$(zenity --file-selection --title="选择视频文件" \
      --file-filter="视频文件 | *.mp4 *.avi *.mov *.mkv *.webm" \
      --filename="$HOME/圆测试视频/" 2>/dev/null) || exit 0
    [ -n "$file" ] || exit 0
    SHOW=1 ./run.sh video "$file"
    ;;
  *)
    exit 0
    ;;
esac
rc=$?
if [ "$rc" -ne 0 ]; then
  read -r -p "程序异常退出（代码 $rc），按回车关闭窗口..."
fi
