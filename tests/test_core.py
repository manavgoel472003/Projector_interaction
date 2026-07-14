import unittest

import numpy as np

from wall_touch_core import (
    TouchGate,
    build_homography,
    camera_detection_roi,
    hand_plane_scale,
    point_in_output,
    projection_near_frame_edge,
    transform_points,
    validate_camera_quad,
)


class GeometryTests(unittest.TestCase):
    def test_homography_maps_camera_quad_to_output_quad(self):
        camera = np.array([[100, 80], [900, 120], [850, 650], [130, 620]], dtype=np.float32)
        output = np.array([[50, 50], [1870, 50], [1870, 1150], [50, 1150]], dtype=np.float32)
        matrix = build_homography(camera, output)
        mapped = transform_points(matrix, camera)
        np.testing.assert_allclose(mapped, output, atol=0.02)

    def test_output_boundary(self):
        self.assertTrue(point_in_output(np.array([100, 100]), 1920, 1200))
        self.assertFalse(point_in_output(np.array([-1, 100]), 1920, 1200))
        self.assertFalse(point_in_output(np.array([1920, 100]), 1920, 1200))

    def test_small_or_crossed_calibration_is_rejected(self):
        small = np.array([[20, 600], [80, 600], [80, 650], [20, 650]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "too small"):
            validate_camera_quad(small, (1280, 720))

        crossed = np.array([[100, 100], [1100, 600], [1100, 100], [100, 600]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "cross"):
            validate_camera_quad(crossed, (1280, 720))

    def test_users_projection_is_accepted_and_cropped(self):
        points = np.array(
            [[216, 541], [391, 535], [367, 677], [100, 675]],
            dtype=np.float32,
        )
        validate_camera_quad(points, (1280, 720))
        x0, y0, x1, y1 = camera_detection_roi(points, (1280, 720))
        self.assertLessEqual(x0, 100)
        self.assertLessEqual(y0, 535)
        self.assertGreater(x1, 391)
        self.assertGreater(y1, 677)
        self.assertLess((x1 - x0) * (y1 - y0), 1280 * 720)
        self.assertTrue(projection_near_frame_edge(points, (1280, 720)))


class HandScaleTests(unittest.TestCase):
    def test_scale_tracks_apparent_hand_size(self):
        landmarks = np.zeros((21, 2), dtype=np.float32)
        landmarks[0] = (10, 30)
        landmarks[5] = (0, 0)
        landmarks[9] = (10, 0)
        landmarks[17] = (30, 0)
        scale = hand_plane_scale(landmarks)
        doubled = hand_plane_scale(landmarks * 2)
        self.assertAlmostEqual(doubled / scale, 2.0, places=5)


class TouchGateTests(unittest.TestCase):
    def test_near_camera_hand_is_rejected(self):
        gate = TouchGate(reference_scale=100, maximum_ratio=1.25, dwell_seconds=0)
        decision = gate.update(
            scale=150, point=np.array([500, 400]), timestamp=1.0,
            inside=True, index_extended=True,
        )
        self.assertFalse(decision.active)
        self.assertEqual(decision.reason, "hand too close to camera")

    def test_wall_scale_activates_after_dwell(self):
        gate = TouchGate(reference_scale=100, dwell_seconds=0.1)
        first = gate.update(
            scale=102, point=np.array([500, 400]), timestamp=1.0,
            inside=True, index_extended=True,
        )
        active = gate.update(
            scale=101, point=np.array([503, 401]), timestamp=1.11,
            inside=True, index_extended=True,
        )
        self.assertFalse(first.active)
        self.assertTrue(active.active)

    def test_outside_point_resets_gate(self):
        gate = TouchGate(reference_scale=100, dwell_seconds=0)
        gate.update(
            scale=100, point=np.array([500, 400]), timestamp=1.0,
            inside=True, index_extended=True,
        )
        active = gate.update(
            scale=100, point=np.array([500, 400]), timestamp=1.01,
            inside=True, index_extended=True,
        )
        outside = gate.update(
            scale=100, point=np.array([2000, 400]), timestamp=1.02,
            inside=False, index_extended=True,
        )
        self.assertTrue(active.active)
        self.assertFalse(outside.active)

    def test_short_tracking_loss_does_not_require_another_dwell(self):
        gate = TouchGate(reference_scale=100, dwell_seconds=0.05, tracking_grace_seconds=0.2)
        gate.update(
            scale=100, point=np.array([500, 400]), timestamp=1.0,
            inside=True, index_extended=True,
        )
        active = gate.update(
            scale=100, point=np.array([500, 400]), timestamp=1.06,
            inside=True, index_extended=True,
        )
        lost = gate.update(
            scale=None, point=None, timestamp=1.12,
            inside=False, index_extended=False,
        )
        resumed = gate.update(
            scale=103, point=np.array([502, 400]), timestamp=1.18,
            inside=True, index_extended=True,
        )
        self.assertTrue(active.active)
        self.assertFalse(lost.active)
        self.assertTrue(lost.candidate)
        self.assertTrue(resumed.active)


if __name__ == "__main__":
    unittest.main()
