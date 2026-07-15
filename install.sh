#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

"$PYTHON" - <<'PY'
import sys

if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10 or newer is required")
PY

"$PYTHON" -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/python" -m pip install --upgrade pip
"$ROOT/.venv/bin/python" -m pip install --editable "$ROOT"
PYTHON="$ROOT/.venv/bin/python" "$ROOT/scripts/download_hand_model.sh"

(
  cd "$ROOT"
  "$ROOT/.venv/bin/python" -m unittest discover -s tests -v
)

printf '\nSetup complete. Run:\n  %s/run_wall_touch_demo.sh --fresh\n' "$ROOT"
printf 'For an Orbbec camera, install its Linux USB rule once:\n  %s/scripts/install_orbbec_udev.sh\n' "$ROOT"
