import unittest

import numpy as np

from wall_touch_core import (
    DepthContactTracker,
    DepthTouchGate,
    TouchGate,
    WallDepthModel,
    build_homography,
    camera_detection_roi,
    combine_wall_depth_models,
    fit_wall_depth_model,
    hand_plane_scale,
    point_in_output,
    projection_near_frame_edge,
    sample_fingertip_depth,
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


class DepthTouchTests(unittest.TestCase):
    def test_depth_tracker_finds_near_wall_end_of_connected_hand(self):
        reference = np.full((120, 180), 1200, np.float32)
        current = reference.copy()
        current[45:75, 70:125] = 1110
        current[55:66, 45:75] = 1176
        quad = np.array([[10, 10], [170, 10], [170, 110], [10, 110]], np.float32)

        contact = DepthContactTracker(reference, quad).detect(current)

        self.assertIsNotNone(contact)
        self.assertAlmostEqual(contact.gap_mm, 24.0, delta=1.0)
        self.assertLess(contact.camera_point[0], 76)
        self.assertGreater(contact.component_area, 1500)

    def test_depth_tracker_reports_hover_gap_for_gate_rejection(self):
        reference = np.full((100, 140), 1500, np.float32)
        current = reference.copy()
        current[30:75, 45:100] = 1400
        quad = np.array([[5, 5], [134, 5], [134, 94], [5, 94]], np.float32)

        hover = DepthContactTracker(reference, quad).detect(current)

        self.assertIsNotNone(hover)
        self.assertAlmostEqual(hover.gap_mm, 100.0, delta=1.0)

    def test_depth_tracker_rejects_wall_noise_and_small_speckles(self):
        rng = np.random.default_rng(7)
        reference = np.full((100, 140), 1100, np.float32)
        noisy = reference + rng.normal(0, 4, reference.shape).astype(np.float32)
        noisy[40:44, 60:64] -= 30
        quad = np.array([[5, 5], [134, 5], [134, 94], [5, 94]], np.float32)

        contact = DepthContactTracker(reference, quad).detect(noisy)

        self.assertIsNone(contact)

    def test_depth_tracker_uses_calibrated_pixel_noise(self):
        reference = np.full((100, 140), 1200, np.float32)
        noise = np.full(reference.shape, 30.0, np.float32)
        current = reference.copy()
        current[25:75, 40:100] -= 35
        quad = np.array([[5, 5], [134, 5], [134, 94], [5, 94]], np.float32)

        observation = DepthContactTracker(
            reference,
            quad,
            wall_noise_mm=noise,
            noise_multiplier=1.5,
            temporal_frames=1,
        ).detect(current)

        self.assertIsNone(observation)

    def test_reciprocal_depth_plane_fits_slanted_wall_with_outliers(self):
        width, height = 320, 200
        yy, xx = np.mgrid[:height, :width]
        inverse = (
            0.00082
            + 0.00011 * ((xx - 159.5) / width)
            - 0.00007 * ((yy - 99.5) / height)
        )
        depth = (1.0 / inverse).astype(np.float32)
        depth[80:105, 130:165] = 650
        depth[::17, ::13] = 0
        quad = np.array([[20, 15], [300, 18], [295, 185], [24, 181]], np.float32)

        model = fit_wall_depth_model(depth, quad)

        self.assertLess(model.rmse_mm, 1.0)
        expected = float(1.0 / inverse[120, 240])
        actual = model.expected_depth(np.array([240, 120]), (width, height))
        self.assertAlmostEqual(actual, expected, delta=2.0)

    def test_wall_model_round_trips_and_combines(self):
        first = WallDepthModel(np.array([0.1, 0.2, 0.001]), 3.0, 100)
        second = WallDepthModel(np.array([0.2, 0.3, 0.0012]), 5.0, 120)
        restored = WallDepthModel.from_dict(first.to_dict())
        combined = combine_wall_depth_models([first, second])
        np.testing.assert_allclose(restored.coefficients, first.coefficients)
        np.testing.assert_allclose(combined.coefficients, [0.15, 0.25, 0.0011])
        self.assertEqual(combined.sample_count, 220)

    def test_fingertip_depth_prefers_foreground_in_patch(self):
        depth = np.full((80, 100), 1200, np.float32)
        depth[36:45, 46:55] = 1168
        sampled = sample_fingertip_depth(depth, np.array([50, 40]), radius=7)
        self.assertAlmostEqual(sampled, 1168, delta=1)

    def test_depth_gate_rejects_hover_and_accepts_contact(self):
        gate = DepthTouchGate(maximum_gap_mm=45, dwell_seconds=0.05)
        hover = gate.update(
            gap_mm=90,
            point=np.array([400, 300]),
            timestamp=1.0,
            inside=True,
            index_extended=True,
        )
        first = gate.update(
            gap_mm=24,
            point=np.array([400, 300]),
            timestamp=1.1,
            inside=True,
            index_extended=True,
        )
        contact = gate.update(
            gap_mm=22,
            point=np.array([402, 301]),
            timestamp=1.16,
            inside=True,
            index_extended=True,
        )
        self.assertEqual(hover.reason, "finger above wall")
        self.assertFalse(first.active)
        self.assertTrue(contact.active)
        self.assertEqual(contact.distance_mm, 22)


if __name__ == "__main__":
    unittest.main()
