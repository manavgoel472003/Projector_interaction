# Contributing

## Setup

Run `./install.sh` from the repository root. It creates an isolated virtual
environment, retrieves the model, and runs the unit suite.

## Before a Change

- Keep external-camera selection explicit. Do not add webcam fallback behavior.
- Keep geometry and touch decisions in `wall_touch_core.py` so they remain
  testable without camera hardware.
- Keep visual simulations in `wall_touch_effects.py` or
  `wall_touch_ambient_effects.py`.
- Do not commit `models/hand_landmarker.task` or local calibration data.

## Validation

```bash
make test
.venv/bin/python -m py_compile wall_touch_*.py
.venv/bin/wall-touch-demo --help
```

Camera or projector changes also require a manual hardware test. Confirm all
four calibration corners, touch calibration, every interaction mode, clear,
reset, fullscreen, and quit.
