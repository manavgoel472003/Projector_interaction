import unittest

import numpy as np

from wall_touch_ambient_effects import ConstellationField, MagneticSand


POINT = np.array([160, 100], dtype=np.float32)
COLOR = np.array([220, 90, 170], dtype=np.float32)


class AmbientEffectTests(unittest.TestCase):
    def test_constellation_connects_persistent_stars(self):
        effect = ConstellationField(320, 200)
        self.assertTrue(effect.add(POINT, COLOR, 1.0))
        self.assertTrue(effect.add(POINT + (30, 0), COLOR, 1.2))
        effect.step(0.1)
        self.assertEqual(effect.count, 2)
        sizes = [float(star["size"]) for star in effect.stars]
        self.assertNotEqual(sizes[0], sizes[1])
        image = effect.render()
        self.assertEqual(image.shape, (200, 320, 3))
        self.assertGreater(image[100, 175].sum(), effect.background[100, 175].sum() + 20)

    def test_constellation_uses_distinct_star_size_tiers(self):
        effect = ConstellationField(320, 200)
        for index in range(30):
            point = np.array([40 + (index % 2) * 80, 80], dtype=np.float32)
            self.assertTrue(effect.add(point, COLOR, 1.0 + index * 0.2))

        sizes = np.array([float(star["size"]) for star in effect.stars])
        self.assertLess(sizes.min(), 0.9)
        self.assertTrue(np.any((sizes >= 1.35) & (sizes <= 2.15)))
        self.assertGreater(sizes.max(), 3.0)

    def test_magnetic_sand_moves_toward_touch(self):
        effect = MagneticSand(320, 200, count=200)
        initial = effect.positions.copy()
        effect.attract(POINT)
        effect.step(1 / 30)
        self.assertFalse(np.allclose(initial, effect.positions))
        image = effect.render()
        self.assertEqual(image.shape, (200, 320, 3))
        changed = np.any(image != effect.background, axis=2)
        self.assertGreater(np.count_nonzero(changed), 500)
        self.assertGreater(image.max(), 245)

if __name__ == "__main__":
    unittest.main()
