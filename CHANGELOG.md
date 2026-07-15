# Changelog

## 3.2.0

- Add guided positive-touch calibration at the projection center and four
  interior corners after empty-wall capture.
- Learn and persist real contact gap and component-area ranges instead of
  relying only on fixed depth thresholds.
- Require three spatially consistent profile-matching frames before exposing
  an Orbbec cursor, preventing raw depth noise from bouncing around the wall.
- Add `--recalibrate-depth` to preserve projection points while relearning the
  wall and guided touch profile.
- Restart the Gemini pipeline once when initial USB 2.1 frame synchronization
  times out.

## 3.1.0

- Replace Orbbec RGB hand-landmark tracking with direct depth-background
  contact tracking inside the calibrated projection.
- Persist a per-pixel empty-wall depth reference and noise map in a compressed
  calibration sidecar.
- Add three-frame temporal depth filtering and pixel-specific noise rejection.
- Tune the measured Gemini 336 setup to a `30 mm` contact limit and `0.75`
  noise multiplier, while retaining command-line overrides.
- Tolerate one dropped synchronized Orbbec frame before failing capture.

## 3.0.0

- Add synchronized Orbbec Gemini RGB-D capture through the official
  `pyorbbecsdk2` package with hardware depth-to-color alignment.
- Fit an empty-wall reciprocal-depth plane and classify touch from the measured
  fingertip-to-wall gap in millimeters.
- Prefer Orbbec automatically while retaining the external RGB fallback.
- Match color/depth frame rates and cap USB 2.1 links at the tested
  hardware-aligned `640x480/15 FPS` profile.
- Add robust depth patch sampling, wall-depth persistence, depth recalibration,
  Linux udev setup, and deterministic depth tests.

## 2.7.1

- Avoid the repeated libjpeg corruption warnings produced by Logitech
  `046d:0825` MJPEG streams by automatically selecting clean `YUYV`
  `640x480/30` capture.
- Add explicit `--camera-format`, `--camera-width`, `--camera-height`, and
  `--camera-fps` overrides and report the negotiated stream at startup.

## 2.7.0

- Discover any connected external V4L2 camera instead of depending on one
  hard-coded Innomaker device path.
- Prefer stable `by-id` and `by-path` primary image streams, skip metadata
  nodes, and retain protection against the known PC webcam.
- Support explicit camera paths and the `WALL_TOUCH_CAMERA` environment
  variable for multi-camera setups.
- Add luminous constellation connections and a wider three-tier star-size
  distribution.

## 2.6.0

- Keep Paint, Spill, Ripple, and Pulse plus Constellation and Magnetic Sand;
  remove the other experimental modes.
- Give each constellation star a stable randomized size and twinkle phase.
- Retheme Ripple as reflective crimson liquid with highlights driven by the
  active wave surface.
- Remove the experimental Particles mode.
- Increase Ripple simulation detail and add cubic scaling plus a full-resolution
  smoothing pass for cleaner projected wave edges.

## 2.5.0

- Add paint, spill, ripple, particles, and pulse interaction modes.
- Make spill mode the default with a `0.50-1.60` touch scale and `50 ms` dwell.
- Add vibrant mode-specific complementary backgrounds.
- Remove decorative water frames and make particles behave like fireworks.
- Add curated pigment colors, paper grain, water caustics, spark twinkles, and pulse halos.
- Add clear, touch relearn, geometry reset, and direct mode controls.
- Require an explicit external V4L2 camera and reject known PC webcams.
- Add reproducible setup, model verification, tests, and CI.
