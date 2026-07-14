#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="$ROOT/.venv/bin/wall-touch-demo"
MODEL="$ROOT/models/hand_landmarker.task"

if [[ ! -x "$APP" || ! -f "$MODEL" ]]; then
  printf 'Setup is incomplete. Run %s/install.sh first.\n' "$ROOT" >&2
  exit 1
fi

exec "$APP" "$@"
