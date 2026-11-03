#!/usr/bin/env bash
# One-time environment setup on macOS (Apple Silicon or Intel).
#   bash scripts/setup_macos.sh
# Creates .venv, installs the pinned dependencies and fixes the LightGBM/PyTorch
# OpenMP clash: both wheels ship their own libomp.dylib, and loading two OpenMP
# runtimes in one process makes LightGBM segfault or PyTorch deadlock.  The fix
# points LightGBM's rpath at PyTorch's copy so that only one runtime is loaded.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3.13}"
if [ ! -d .venv ]; then
  "$PYTHON" -m venv .venv
fi
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e .

SITE=$(.venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
LGB="$SITE/lightgbm/lib/lib_lightgbm.dylib"
TORCHLIB="$SITE/torch/lib"
if [ -f "$LGB" ] && [ -f "$TORCHLIB/libomp.dylib" ]; then
  if otool -l "$LGB" | grep -q "/opt/homebrew/opt/libomp/lib"; then
    install_name_tool -rpath /opt/homebrew/opt/libomp/lib "$TORCHLIB" "$LGB"
    codesign -s - -f "$LGB"
    echo "patched LightGBM rpath -> $TORCHLIB"
  else
    echo "LightGBM rpath already patched"
  fi
fi

.venv/bin/python - <<'EOF'
import lightgbm, torch, sklearn, numpy, pandas
print("lightgbm", lightgbm.__version__, "| torch", torch.__version__, "| sklearn", sklearn.__version__,
      "| numpy", numpy.__version__, "| pandas", pandas.__version__)
EOF
echo "ok. next: .venv/bin/python scripts/download_data.py"
