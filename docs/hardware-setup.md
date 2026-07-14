# External-Camera Wall Touch Demo

This demo supports standard external V4L2 cameras. It discovers the primary
image stream automatically, prefers stable device paths, and explicitly
refuses the known Shinetech PC webcam.

## Camera placement

1. Put the projector in its final position facing the wall.
2. Put the external camera near the projector, preferably within 20-50 cm of
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

To choose a camera explicitly or override automatic selection:

```bash
./run_wall_touch_demo.sh --camera /dev/video4 --fresh
# Or persist the selection for future launches:
WALL_TOUCH_CAMERA=/dev/video4 ./run_wall_touch_demo.sh --fresh
```

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
3. A target appears at the center of the projected area.
4. Put your index fingertip physically on that target, using the same pointing
   posture you will use while painting.
5. Keep the fingertip there while the green outer ring fills. Sampling starts
   automatically after geometric calibration.
6. Touch and drag within the projected area to mix paint. Use `t` only when you
   intentionally want to relearn the touch distance later.

The projection may occupy a small part of the full camera frame. After the four
clicks, the hand detector automatically crops around that region. Calibration
accepts a projected quadrilateral covering at least 1.25% of the camera frame.

## Controls

- `]` or `m`: select the next reactive mode
- `[`: select the previous reactive mode
- `1`-`6`: select any mode directly
- `c`: clear all artwork while keeping the current calibration
- `t`: relearn the touch-plane hand size
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

Calibration is saved in `wall_touch_calibration.json`. Start without `--fresh`
to reuse it only when the camera, projector, wall, and display layout have not
moved.

## What counts as touch

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

- MediaPipe Hand Landmarker provides the 21 hand landmarks and fingertip.
- OpenCV computes a planar homography from the four camera clicks to known
  projector pixels.
- The fingertip is mapped through that homography.
- Palm size in mapped projector coordinates estimates whether the hand is at
  the calibrated wall plane.
- A short dwell rejects fly-by motion, then dragging paints continuously.

Primary references:

- https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python
- https://docs.opencv.org/master/d9/dab/tutorial_homography.html
