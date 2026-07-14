#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
MODEL="$ROOT/models/hand_landmarker.task"
URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
EXPECTED_SHA256="fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"

mkdir -p "$(dirname "$MODEL")"

if [[ -f "$MODEL" ]] && printf '%s  %s\n' "$EXPECTED_SHA256" "$MODEL" | sha256sum --check --status; then
  printf 'MediaPipe model already verified: %s\n' "$MODEL"
  exit 0
fi

TEMP_MODEL="$(mktemp "$MODEL.tmp.XXXXXX")"
trap 'rm -f "$TEMP_MODEL"' EXIT

"$PYTHON" - "$URL" "$TEMP_MODEL" <<'PY'
import pathlib
import sys
import urllib.request

url, destination = sys.argv[1:]
print(f"Downloading MediaPipe Hand Landmarker from {url}")
with urllib.request.urlopen(url, timeout=60) as response:
    pathlib.Path(destination).write_bytes(response.read())
PY

printf '%s  %s\n' "$EXPECTED_SHA256" "$TEMP_MODEL" | sha256sum --check --status || {
  printf 'Downloaded model failed SHA-256 verification.\n' >&2
  exit 1
}
mv "$TEMP_MODEL" "$MODEL"
trap - EXIT
printf 'Model installed: %s\n' "$MODEL"
