#!/usr/bin/env bash
# Bind the bundled Python environment (.python/ + .venv/) to the folder it was extracted to.
# Idempotent; re-run after moving or copying the folder.
#
#   bash setup_env.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"
PY_HOME="$(ls -1d "$ROOT"/.python/cpython-3.10.*-linux-x86_64-gnu 2>/dev/null | tail -n 1)/bin"
SITE="$VENV/lib/python3.10/site-packages"

if [[ ! -x "$PY_HOME/python3.10" || ! -d "$SITE" ]]; then
  echo "ERROR: bundled environment not found (.python/ or .venv/ missing under $ROOT)" >&2
  exit 1
fi

# venv interpreter → bundled standalone CPython (pyvenv.cfg needs an absolute home).
ln -sfn "$PY_HOME/python3.10" "$VENV/bin/python"
ln -sfn python "$VENV/bin/python3"
ln -sfn python "$VENV/bin/python3.10"
sed -i "s|^home = .*|home = $PY_HOME|" "$VENV/pyvenv.cfg"

# Console-script shebangs and activate scripts.
for f in "$VENV"/bin/*; do
  [[ -f "$f" && ! -L "$f" ]] || continue
  if head -n 1 "$f" | grep -qE '^#!.*/bin/python[0-9.]*$'; then
    sed -i "1s|^#!.*$|#!$VENV/bin/python|" "$f"
  fi
done
for f in "$VENV"/bin/activate*; do
  [[ -f "$f" ]] || continue
  sed -i -E "/VIRTUAL_ENV/s#(['\"])/[^'\"]*/\.venv(['\"])#\1$VENV\2#" "$f"
done

# Editable installs of this repo and its local packages.
echo "$ROOT" > "$SITE/_editable_impl_cosmos_predict2.pth"
for pkg in cosmos-oss cosmos-cuda cosmos-gradio; do
  echo "$ROOT/packages/$pkg" > "$SITE/_editable_impl_${pkg//-/_}.pth"
done

# ffmpeg (used to read / write mp4): fall back to the static binary shipped with imageio-ffmpeg.
if ! PATH="${PATH//$VENV\/bin:/}" command -v ffmpeg >/dev/null 2>&1; then
  bin="$(ls -1 "$SITE"/imageio_ffmpeg/binaries/ffmpeg-linux-* 2>/dev/null | head -n 1)"
  if [[ -n "$bin" ]]; then
    chmod +x "$bin"
    ln -sfn "$bin" "$VENV/bin/ffmpeg"
    echo "No system ffmpeg; using $bin"
  fi
fi

"$VENV/bin/python" - <<'EOF'
import sys

import torch

print(f"python {sys.version.split()[0]} ({sys.prefix})")
print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}")
EOF
echo "Environment ready: $VENV"
