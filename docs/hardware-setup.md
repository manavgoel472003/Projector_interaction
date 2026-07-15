# Orbbec RGB-D Wall Touch Demo

The preferred sensor is an Orbbec Gemini 336/335-series RGB-D camera. The app
uses synchronized hardware depth-to-color alignment and retains standard
external V4L2 cameras as a fallback.

## Camera placement

1. Put the projector in its final position facing the wall.
2. Put the Orbbec near the projector, preferably within 20-50 cm of
   the projector lens.
3. Aim the camera at the center of the projected rectangle.
4. Make sure the camera sees the entire rectangle with some wall around every
   edge and can see a hand at all four corners.
5. Fix both devices in place for the session. Recalibrate after either one
   moves or when changing walls.

A mostly front-facing view gives the cleanest coordinate mapping. A slight
horizontal offset is fine; the four-point homography corrects that perspective.

## Run

The current display layout was detected as:

- Projector/HDMI: `1920x1200` at desktop position `(0, 0)`
- Laptop: starts at desktop position `(1920, 0)`

Run:

```bash
./run_wall_touch_demo.sh --fresh
```

Before the first Orbbec run:

```bash
./install.sh
./scripts/install_orbbec_udev.sh
# Unplug and reconnect the camera, then verify Gemini 336 PID 0803:
lsusb | grep 2bc5:0803
```

Use the supplied USB 3 data cable when possible. USB 3 is preferred, but the
app also detects USB 2.1 and automatically selects the tested bandwidth-safe
hardware-aligned `640x480/15 FPS` profile.

To force Orbbec depth:

```bash
./run_wall_touch_demo.sh --sensor orbbec --fresh
```

To force RGB fallback and choose a camera explicitly:

```bash
./run_wall_touch_demo.sh --sensor rgb --camera /dev/video4 --fresh
# Or persist the selection for future launches:
WALL_TOUCH_CAMERA=/dev/video4 ./run_wall_touch_demo.sh --fresh
```

The launcher automatically avoids the malformed MJPEG stream on Logitech
`046d:0825` cameras and selects clean `YUYV 640x480/30` capture. Use
`--camera-format`, `--camera-width`, `--camera-height`, and `--camera-fps` only
when overriding the automatic profile for other hardware.

For a different projector layout, provide its resolution and desktop origin:

```bash
./run_wall_touch_demo.sh \
  --projector-width 1920 --projector-height 1080 \
  --projector-x 0 --projector-y 0 --fresh
```

## First calibration

1. The projector shows four numbered crosshairs.
2. In the camera window on the laptop, click their centers in order:
   top-left, top-right, bottom-right, bottom-left.
3. Keep people and objects out of the projected area while the green depth
   calibration ring fills for 45 frames.
4. Touch and hold the center target, then the upper-left, upper-right,
   lower-right, and lower-left targets. Each target advances automatically
   after collecting a stable positive contact.
5. Touch and drag within the projected area.
6. Press `t` only when intentionally relearning the wall and touch profile.

Calibration accepts a projected quadrilateral covering at least 1.25% of the
camera frame. Orbbec mode processes only that quadrilateral; RGB fallback uses
an expanded crop around it for hand tracking.

## Controls

- `]` or `m`: select the next reactive mode
- `[`: select the previous reactive mode
- `1`-`6`: select any mode directly
- `c`: clear all artwork while keeping the current calibration
- `t`: relearn wall depth in Orbbec mode or hand size in RGB fallback
- `r`: clear artwork and discard calibration so you can choose new projection points
- `f`: toggle projector fullscreen
- `q` or `Esc`: quit

Watercolor spill is the default mode. Brush mode can be selected at startup with:

```bash
./run_wall_touch_demo.sh --mode paint
```

In spill mode, each touch pours pigment into a full-bleed water surface. Horizontal
touch position chooses the hue. Pigment spreads, swirls, mixes with existing
colors, and creates a small surface ripple.

The six modes are:

1. `paint`: soft color-mixing brush strokes
2. `spill`: watercolor pigment diffusing across a full-bleed surface
3. `ripple`: a reflective crimson liquid surface disturbed by touch
4. `pulse`: expanding colored waves lighting a reactive grid
5. `constellation`: connected fading stars
6. `sand`: touch-attracted metallic grains

Geometry is saved in `wall_touch_calibration.json`. The empty-wall depth and
noise map are saved in `wall_touch_calibration.depth.npz`. Start without
`--fresh` to reuse them only when the camera, projector, wall, and display
layout have not moved.

## What counts as touch

In Orbbec mode, 45 empty-wall frames produce a per-pixel median depth and noise
map. A three-frame temporal median is compared with that reference, and direct
depth foreground components provide the interaction coordinate. Five guided
touches learn accepted contact gap and component area. A component must match
that profile for three spatially consistent frames before it becomes a cursor.
MediaPipe is not used in this mode. The pre-calibration defaults are:

```bash
./run_wall_touch_demo.sh \
  --touch-max-gap-mm 30 --depth-noise-multiplier 0.75
```

Lower `--touch-max-gap-mm` to reject hovering more strictly. Raise it if real
touches are missed, but raising it can also admit depth noise. Moving the depth
camera closer to the wall is preferable when possible. The debug window reports
the live wall gap.

### RGB fallback

One normal camera cannot measure absolute depth. This demo learns the apparent
hand size while your finger is touching the wall. A hand closer to the camera
looks larger after wall mapping and is rejected. The status window reports the
live wall ratio; approximately `1.00` means the learned wall distance.

The default accepted range is `0.50-1.60`. It is intentionally broad enough for
normal hand rotation. It can be tightened after testing:

```bash
./run_wall_touch_demo.sh \
  --touch-scale-min 0.82 --touch-scale-max 1.15
```

This rejects obvious near-camera movement, but a finger hovering a few
centimeters above the wall can still look like contact. Reliable centimeter
level contact detection requires a second side camera, a depth camera, or an
optical touch plane.

The tracker tolerates brief hand-landmark losses for 220 ms, smooths the hand
scale measurement, and does not require a rigid index-finger pose by default.
Use `--require-index-extension` only when stricter gesture filtering is more
important than touch continuity.

## Technical basis

- OpenCV computes a planar homography from the four camera clicks to known
  projector pixels.
- Orbbec hardware D2C provides synchronized aligned color and depth.
- Per-pixel background subtraction, temporal filtering, and connected
  components locate depth contact without RGB landmarks.
- MediaPipe provides fingertip landmarks only for RGB fallback, which uses
  palm size in mapped projector coordinates.
- A short dwell rejects fly-by motion, then dragging paints continuously.

Primary references:

- https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python
- https://docs.opencv.org/master/d9/dab/tutorial_homography.html
- https://github.com/orbbec/pyorbbecsdk
