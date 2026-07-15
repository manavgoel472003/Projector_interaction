import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from wall_touch_paint import camera_stream_profile, validate_camera


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


if __name__ == "__main__":
    unittest.main()
