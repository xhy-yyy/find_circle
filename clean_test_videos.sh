#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

"${PY:-python3}" - "$@" <<'PYEOF'
import argparse
from pathlib import Path
import re
import shutil
import subprocess

ap = argparse.ArgumentParser(prog='./clean_test_videos.sh', description='清理默认生成的测试标注视频。')
ap.add_argument('--dry-run', action='store_true', help='只列出待删除视频')
a = ap.parse_args()
root = Path('out').resolve()
if not root.is_dir():
    print(f'测试输出目录不存在：{root}')
    raise SystemExit(0)
fuser = shutil.which('fuser')
if fuser is None:
    ap.error('缺少 fuser，无法检查视频占用；请安装 psmisc 后重试')

count = size = skipped = failures = 0
print(f'清理目录：{root}')
for p in sorted(root.iterdir()):
    if p.is_symlink() or not p.is_file() or not re.fullmatch(r'video_.+_\d{8}_\d{6}\.mp4', p.name):
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
