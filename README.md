# Projector Interaction

Turn a projector and one external RGB camera into an interactive wall. The
camera maps a fingertip into projector coordinates with a four-point
homography. A learned apparent-hand-size threshold provides approximate touch
detection without a depth sensor.

The application never selects a PC webcam by numeric camera index. It requires
an explicit external V4L2 device path and rejects known built-in webcam names.

## Modes

- `paint`: textured soft brush strokes, pigment flecks, and a paper-like canvas
- `spill`: luminous watercolor pigment mixing across a full-bleed surface
- `ripple`: reflective crimson liquid with pale highlights and touch-driven waves
- `pulse`: layered luminous rings across a reactive jewel-tone grid
- `constellation`: fading stars connected by fine luminous lines
- `sand`: metallic grains attracted into touch-driven vortices

`spill` is the default. The tested touch defaults are a scale range of
`0.50-1.60` with a `50 ms` dwell.

## Requirements

- Linux with Video4Linux2
- Python 3.10 or newer
- Projector configured as a display
- External RGB camera that can see the complete projected rectangle

## Quick Start

```bash
git clone https://github.com/manavgoel472003/Projector_interaction.git
cd Projector_interaction
./install.sh
./run_wall_touch_demo.sh --fresh
```

`install.sh` creates `.venv`, installs pinned runtime dependencies, downloads
the official MediaPipe Hand Landmarker model, verifies its SHA-256 checksum,
and runs the tests.

The launcher automatically discovers the primary video stream of a connected
external V4L2 camera. It prefers stable `/dev/v4l/by-id` paths, so replacing a
camera does not require editing the launcher. If several external cameras are
connected, select one explicitly:

```bash
./run_wall_touch_demo.sh \
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
3. Walk to the wall and hold your index fingertip on the projected center
   target while the touch samples are collected.
4. Touch and drag inside the projected region.

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
| `t` | Relearn the touch plane |
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
wall_touch_core.py     geometry and touch gate
wall_touch_effects.py  original visual simulations
wall_touch_ambient_effects.py  ambient and field simulations
tests/                 deterministic unit tests
scripts/               model setup utility
docs/                  hardware and calibration notes
```

## Limitations

A monocular RGB camera estimates wall contact from apparent hand size. It can
reject obvious near-camera motion, but cannot reliably distinguish a fingertip
touching the wall from one hovering a few centimeters above it. Precise contact
detection requires a side camera, depth camera, or optical touch plane.

No software license has been selected for this repository yet. Add one before
publishing if others should be allowed to copy, modify, or redistribute it.
