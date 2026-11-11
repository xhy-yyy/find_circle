#!/usr/bin/env bash
# 编译 ED_Lib EDCircles 动态库 -> ringdet/_lib/libedcircles.so
# 依赖：cmake、g++、OpenCV C++ 开发包（core、imgproc、video）
set -euo pipefail
cd "$(dirname "$0")"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j"$(nproc)"
ls -l ringdet/_lib/
