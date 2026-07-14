# Changelog

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
