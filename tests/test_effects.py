import unittest

import numpy as np

from wall_touch_effects import PulseGrid, WatercolorPool


class WatercolorPoolTests(unittest.TestCase):
    def test_drop_changes_pool_and_clear_removes_pigment(self):
        pool = WatercolorPool(640, 400, simulation_width=160)
        empty = pool.render()
        added = pool.add_drop(np.array([320, 200]), np.array([30, 30, 240]))
        for _ in range(4):
            pool.step()
        mixed = pool.render()

        self.assertTrue(added)
        self.assertGreater(pool.pigment_mass, 0)
        self.assertEqual(mixed.shape, (400, 640, 3))
        self.assertGreater(np.abs(mixed.astype(int) - empty.astype(int)).sum(), 0)

        pool.clear()
        self.assertEqual(pool.pigment_mass, 0)

    def test_water_surface_is_full_bleed(self):
        pool = WatercolorPool(640, 400, simulation_width=160)
        self.assertTrue(pool.add_drop(np.array([0, 0]), np.array([240, 30, 30])))
        self.assertGreater(pool.pigment_mass, 0)

    def test_pure_ripple_changes_wave_without_pigment(self):
        pool = WatercolorPool(640, 400, simulation_width=160)
        self.assertTrue(pool.add_ripple(np.array([320, 200])))
        pool.step()
        self.assertEqual(len(pool.ripple_events), 1)
        self.assertGreater(np.abs(pool.render().astype(int) - pool.water_color).sum(), 0)
        self.assertEqual(pool.pigment_mass, 0)

    def test_crimson_ripple_has_reflective_highlights(self):
        pool = WatercolorPool(
            640,
            400,
            simulation_width=160,
            ripple_contrast=1.55,
            water_color=(20, 15, 105),
            reflection_color=(205, 226, 255),
            reflection_gain=2.35,
        )
        still_image = pool.render()
        pool.add_ripple(np.array([320, 200]), strength=0.09)
        for _ in range(18):
            pool.step()
        image = pool.render()

        self.assertGreater(image[..., 2].mean(), image[..., 0].mean() * 2)
        self.assertLess(int(still_image.max()), 130)
        self.assertGreater(int(image.max()), 190)


class PulseGridTests(unittest.TestCase):
    def test_pulses_are_throttled_and_expire(self):
        grid = PulseGrid(640, 400)
        point = np.array([320, 200])
        color = np.array([220, 60, 180])
        self.assertTrue(grid.add_pulse(point, color, 1.0))
        self.assertFalse(grid.add_pulse(point, color, 1.05))
        image = grid.render()
        self.assertEqual(image.shape, (400, 640, 3))
        self.assertGreater(np.ptp(image.mean(axis=(0, 1))), 10)
        for _ in range(20):
            grid.step(0.1)
        self.assertEqual(grid.count, 0)


if __name__ == "__main__":
    unittest.main()
