import unittest

import numpy as np

from wall_touch_core import DepthContactObservation, build_homography
from wall_touch_dual import (
    learn_wall_reference,
    prepare_detection_rgb,
    touch_contacts_in_projector,
)


class _StubTracker:
    def __init__(self, observations):
        self._observations = observations

    def observations(self, depth_mm):  # noqa: ARG002 - matches DepthContactTracker
        return self._observations


class DualHelperTests(unittest.TestCase):
    def test_prepare_detection_rgb_resizes_to_target_long_side(self):
        crop = np.zeros((120, 60, 3), dtype=np.uint8)
        rgb = prepare_detection_rgb(crop, detection_size=240, clahe=None)
        # Upscaled so the long side hits the target, aspect preserved, 3 channels.
        self.assertEqual(rgb.shape, (240, 120, 3))

    def test_prepare_detection_rgb_downscales_large_crop(self):
        crop = np.zeros((800, 400, 3), dtype=np.uint8)
        rgb = prepare_detection_rgb(crop, detection_size=384, clahe=None)
        self.assertEqual(max(rgb.shape[:2]), 384)

    def test_learn_wall_reference_medians_and_drops_invalid(self):
        frames = [
            np.array([[1000.0, 0.0]], dtype=np.float32),
            np.array([[1010.0, 0.0]], dtype=np.float32),
            np.array([[1020.0, 0.0]], dtype=np.float32),
        ]
        reference = learn_wall_reference(frames)
        # Median of valid wall samples; the all-invalid (<100) column collapses to 0.
        self.assertAlmostEqual(reference[0, 0], 1010.0, places=3)
        self.assertEqual(reference[0, 1], 0.0)

    def test_touch_contacts_filtered_and_mapped_to_projector(self):
        # Square camera quad -> a 100x100 projector plane (identity-like mapping).
        camera_quad = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        output_points = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        homography = build_homography(camera_quad, output_points)
        observations = [
            DepthContactObservation(np.array([50.0, 50.0]), gap_mm=8.0,
                                    component_area=200, contact_area=30),   # keep
            DepthContactObservation(np.array([20.0, 20.0]), gap_mm=40.0,
                                    component_area=200, contact_area=30),   # gap too large
            DepthContactObservation(np.array([70.0, 70.0]), gap_mm=5.0,
                                    component_area=200, contact_area=2),     # too small
        ]
        contacts = touch_contacts_in_projector(
            _StubTracker(observations), depth_mm=np.zeros((100, 100), np.float32),
            homography=homography, gap_mm=25.0, min_contact_area=6, output_size=(100, 100),
        )
        self.assertEqual(len(contacts), 1)
        np.testing.assert_allclose(contacts[0], [50.0, 50.0], atol=1e-3)


if __name__ == "__main__":
    unittest.main()
