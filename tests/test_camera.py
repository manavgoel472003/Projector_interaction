import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from wall_touch_core import DepthTouchProfile, WallDepthModel
from wall_touch_paint import camera_stream_profile, depth_reference_path, validate_camera
from wall_touch_paint import load_calibration, save_calibration


class CameraSelectionTests(unittest.TestCase):
    @patch(
        "wall_touch_paint.discover_external_cameras",
        return_value=[("/dev/v4l/by-id/example-video-index0", "External UVC Camera")],
    )
    def test_auto_selects_discovered_external_camera(self, _discover):
        self.assertEqual(
            validate_camera("auto"),
            ("/dev/v4l/by-id/example-video-index0", "External UVC Camera"),
        )

    @patch("wall_touch_paint.discover_external_cameras", return_value=[])
    def test_auto_reports_when_no_external_camera_exists(self, _discover):
        with self.assertRaisesRegex(RuntimeError, "No external camera"):
            validate_camera("auto")

    def test_explicit_device_path_is_supported(self):
        with TemporaryDirectory() as directory:
            device = Path(directory) / "custom-camera"
            device.touch()
            with patch("wall_touch_paint.camera_name", return_value="Custom Camera"):
                self.assertEqual(validate_camera(str(device)), (str(device), "Custom Camera"))

    def test_problematic_logitech_mjpeg_uses_clean_raw_profile(self):
        self.assertEqual(
            camera_stream_profile("UVC Camera (046d:0825)", "auto", None, None, None),
            ("YUYV", 640, 480, 30),
        )

    def test_stream_profile_allows_explicit_overrides(self):
        self.assertEqual(
            camera_stream_profile("Other Camera", "mjpg", 1920, 1080, 25),
            ("MJPG", 1920, 1080, 25),
        )

    def test_depth_calibration_round_trips_and_is_sensor_specific(self):
        corners = np.array([[10, 10], [300, 10], [300, 180], [10, 180]], np.float32)
        model = WallDepthModel(np.array([0.0001, -0.0002, 0.001]), 4.2, 800)
        reference = np.full((200, 320), 1234.0, np.float32)
        noise = np.full((200, 320), 8.0, np.float32)
        profile = DepthTouchProfile(12.0, 42.0, 300, 2500, 50)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            save_calibration(
                path,
                "Gemini 336 serial",
                (320, 200),
                (1920, 1200),
                corners,
                corners,
                None,
                model,
                "orbbec-depth",
                reference,
                noise,
                profile,
            )
            loaded = load_calibration(
                path,
                "Gemini 336 serial",
                (320, 200),
                (1920, 1200),
                "orbbec-depth",
            )
            rejected = load_calibration(
                path,
                "Gemini 336 serial",
                (320, 200),
                (1920, 1200),
                "rgb",
            )
            self.assertTrue(depth_reference_path(path).exists())

        self.assertIsInstance(loaded["wall_depth_model"], WallDepthModel)
        np.testing.assert_allclose(loaded["wall_depth_model"].coefficients, model.coefficients)
        np.testing.assert_allclose(loaded["wall_depth_reference"], reference)
        np.testing.assert_allclose(loaded["wall_depth_noise"], noise)
        self.assertEqual(loaded["depth_touch_profile"], profile)
        self.assertIsNone(rejected)


if __name__ == "__main__":
    unittest.main()
