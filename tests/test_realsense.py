import unittest

import numpy as np

from wall_touch_realsense import depth_values_to_mm


class RealSenseConversionTests(unittest.TestCase):
    def test_native_depth_units_convert_to_millimeters(self):
        raw = np.array([[0, 1000], [1500, 2500]], dtype=np.uint16)
        np.testing.assert_allclose(
            depth_values_to_mm(raw, 0.001),
            np.array([[0, 1000], [1500, 2500]], dtype=np.float32),
            atol=0.001,
        )

    def test_nonstandard_depth_scale_is_preserved(self):
        raw = np.array([[100, 400]], dtype=np.uint16)
        np.testing.assert_allclose(
            depth_values_to_mm(raw, 0.00025),
            np.array([[25, 100]], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
