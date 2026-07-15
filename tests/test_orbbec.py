import unittest

import cv2
import numpy as np

from wall_touch_orbbec import OBFormat, color_frame_to_bgr, depth_frame_to_mm


class FakeFrame:
    def __init__(self, data, width, height, pixel_format, scale=1.0):
        self.data = np.asarray(data)
        self.width = width
        self.height = height
        self.pixel_format = pixel_format
        self.scale = scale

    def get_data(self):
        return self.data

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def get_format(self):
        return self.pixel_format

    def get_depth_scale(self):
        return self.scale


class OrbbecConversionTests(unittest.TestCase):
    def test_rgb_frame_converts_to_bgr(self):
        rgb = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
        frame = FakeFrame(rgb, 2, 1, OBFormat.RGB)
        bgr = color_frame_to_bgr(frame)
        np.testing.assert_array_equal(bgr, rgb[..., ::-1])

    def test_yuyv_frame_converts_to_three_channels(self):
        yuyv = np.array([[[100, 128], [100, 128]]], dtype=np.uint8)
        frame = FakeFrame(yuyv, 2, 1, OBFormat.YUYV)
        bgr = color_frame_to_bgr(frame)
        self.assertEqual(bgr.shape, (1, 2, 3))

    def test_mjpeg_frame_decodes(self):
        image = np.full((8, 10, 3), (20, 80, 150), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        frame = FakeFrame(encoded, 10, 8, OBFormat.MJPG)
        self.assertEqual(color_frame_to_bgr(frame).shape, image.shape)

    def test_depth_scale_produces_millimeters(self):
        raw = np.array([[1000, 1500], [2000, 0]], dtype=np.uint16)
        frame = FakeFrame(raw, 2, 2, OBFormat.Y16, scale=0.5)
        np.testing.assert_array_equal(
            depth_frame_to_mm(frame),
            np.array([[500, 750], [1000, 0]], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
