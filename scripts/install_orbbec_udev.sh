#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RULE_SOURCE="$(find "$ROOT/.venv/lib" -path '*/site-packages/pyorbbecsdk/shared/99-obsensor-libusb.rules' -print -quit)"

if [[ -z "$RULE_SOURCE" ]]; then
  printf 'Orbbec SDK rule not found. Run ./install.sh first.\n' >&2
  exit 1
fi

sudo install -m 0644 "$RULE_SOURCE" /etc/udev/rules.d/99-obsensor-libusb.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
printf 'Orbbec USB permissions installed. Unplug and reconnect the camera.\n'
