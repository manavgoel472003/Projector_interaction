#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT/.venv/bin/python"
APP="$ROOT/wall_touch_paint.py"
MODEL="$ROOT/models/hand_landmarker.task"

# Two-camera mode: `--dual` (or `--dual-camera`) routes to the dual-camera app.
if [[ "${1:-}" == "--dual" || "${1:-}" == "--dual-camera" ]]; then
  APP="$ROOT/wall_touch_dual.py"
  shift
fi

# Run the script directly with the venv interpreter so the project directory is
# on sys.path. This avoids depending on the installed console-script entry point,
# whose editable finder can go stale when new modules are added to the project.
if [[ ! -x "$PYTHON" || ! -f "$APP" || ! -f "$MODEL" ]]; then
  printf 'Setup is incomplete. Run %s/install.sh first.\n' "$ROOT" >&2
  exit 1
fi

# ROS setup files can inject Python 3.10 packages into this Python 3.13 venv.
exec env -u PYTHONPATH "$PYTHON" "$APP" "$@"
