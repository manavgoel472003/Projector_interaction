# Projector Interaction

Turn a projector and an Orbbec RGB-D camera into an interactive wall. The
camera maps a fingertip into projector coordinates with a four-point
homography, while aligned depth measures its distance from the calibrated wall
plane in millimeters. A standard external RGB camera remains available as a
fallback.

The application prefers a connected Orbbec automatically. It never selects the
known PC webcam as its RGB fallback.

## Modes

- `paint`: textured soft brush strokes, pigment flecks, and a paper-like canvas
- `spill`: luminous watercolor pigment mixing across a full-bleed surface
- `ripple`: reflective crimson liquid with pale highlights and touch-driven waves
- `pulse`: layered luminous rings across a reactive jewel-tone grid
- `constellation`: fading stars connected by fine luminous lines
- `sand`: metallic grains attracted into touch-driven vortices

`spill` is the default. Depth contact accepts a fingertip from `-15 mm` behind
the fitted wall through `45 mm` in front, with a `50 ms` dwell.

## Requirements

- Linux with Video4Linux2; USB 3 recommended
- Python 3.10 or newer
- Projector configured as a display
- Orbbec Gemini RGB-D camera, preferably Gemini 336/335 series
- Optional external RGB camera fallback

## Quick Start

```bash
git clone https://github.com/manavgoel472003/Projector_interaction.git
cd Projector_interaction
./install.sh
./scripts/install_orbbec_udev.sh
# Unplug and reconnect the Orbbec after installing its USB rule.
./run_wall_touch_demo.sh --fresh
```

`install.sh` creates `.venv`, installs pinned runtime dependencies, downloads
the official MediaPipe Hand Landmarker model, verifies its SHA-256 checksum,
and runs the tests. The udev command installs Orbbec's official Linux USB
permissions and requires your sudo password once.

With the Gemini connected, `--sensor auto` selects synchronized, hardware
aligned color and depth. Force it when diagnosing setup:

```bash
./run_wall_touch_demo.sh --sensor orbbec --fresh
```

The Gemini must appear in `lsusb`; Gemini 336 reports `2bc5:0803`. If it does
not, reconnect it with the supplied USB 3 data cable before debugging software.
On USB 2.1, the app automatically selects the tested bandwidth-safe hardware
alignment profile at `640x480/15 FPS`; USB 3 permits higher-bandwidth profiles.

### RGB fallback

The launcher automatically discovers the primary video stream of a connected
external V4L2 camera. It prefers stable `/dev/v4l/by-id` paths, so replacing a
camera does not require editing the launcher. If several external cameras are
connected, select one explicitly:

```bash
./run_wall_touch_demo.sh \
  --sensor rgb \
  --camera /dev/v4l/by-id/<external-camera>-video-index0 \
  --fresh
```

Do not use `/dev/video0` style numeric indexes unless you have independently
verified the device. Explicit paths such as `--camera /dev/video4` work, but
ambiguous numeric arguments such as `--camera 4` are intentionally refused.
The camera can also be selected persistently with `WALL_TOUCH_CAMERA`.

Camera format is selected automatically. The Logitech `046d:0825` uses raw
`YUYV 640x480/30` because its MJPEG stream produces corrupt-frame warnings.
Format and stream settings can be overridden when testing other hardware:

```bash
./run_wall_touch_demo.sh \
  --camera-format yuyv --camera-width 640 --camera-height 480 --camera-fps 30
```

## Calibration

1. Fix the projector and camera in place.
2. Click the four projected targets in the laptop debug window in this order:
   top-left, top-right, bottom-right, bottom-left.
3. Keep the projected area empty while 12 wall-depth frames are collected.
4. Touch and drag inside the projected region. No fingertip depth calibration
   target is needed.

Calibration is stored locally in `wall_touch_calibration.json` and is ignored
by Git. Use `--fresh` or press `r` after moving the camera, projector, or wall.
See [docs/hardware-setup.md](docs/hardware-setup.md) for placement and display
details.

## Controls

| Key | Action |
| --- | --- |
| `1`-`6` | Select any mode directly |
| `]` / `m` | Next mode |
| `[` | Previous mode |
| `c` | Clear artwork and keep calibration |
| `t` | Relearn wall depth (or RGB touch scale in fallback mode) |
| `r` | Clear artwork and choose new projection points |
| `f` | Toggle projector fullscreen |
| `q` / `Esc` | Quit |

## Development

```bash
./install.sh
make test
.venv/bin/wall-touch-demo --help
```

Repository layout:

```text
wall_touch_paint.py    camera, calibration, interaction loop
wall_touch_orbbec.py  synchronized Orbbec RGB-D capture
wall_touch_core.py     geometry, wall plane, and touch gates
wall_touch_effects.py  original visual simulations
wall_touch_ambient_effects.py  ambient and field simulations
tests/                 deterministic unit tests
scripts/               model setup utility
docs/                  hardware and calibration notes
```

## Limitations

Depth accuracy is limited at fingertip silhouettes and by reflective or
transparent walls. The app samples a patch just inside the fingertip and uses a
robust wall plane, but `--touch-max-gap-mm` may need adjustment for camera
placement and pointing posture. RGB fallback still uses approximate hand size.

No software license has been selected for this repository yet. Add one before
publishing if others should be allowed to copy, modify, or redistribute it.
