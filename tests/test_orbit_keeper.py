import unittest

import numpy as np

from wall_touch_orbit_keeper import OrbitKeeper


class OrbitKeeperTests(unittest.TestCase):
    def setUp(self):
        self.game = OrbitKeeper(
            960,
            640,
            release_seconds=0.10,
            round_seconds=10.0,
            well_lifetime=0.50,
            seed=17,
        )

    def safe_point(self):
        return np.array(
            [self.game.play_left + 120, self.game.play_top + 120],
            dtype=np.float32,
        )

    def test_press_adds_one_well_until_release_and_wells_expire(self):
        point = self.safe_point()
        self.assertTrue(self.game.update(point, True, 1.0, 0.0))
        self.assertFalse(self.game.update(point + 40, True, 1.2, 0.10))
        self.assertEqual(len(self.game.wells), 1)

        self.game.update(None, False, 1.35, 0.0)
        self.assertTrue(self.game.update(point + 40, True, 1.36, 0.0))
        self.assertEqual(len(self.game.wells), 2)

        self.game.step(0.60)
        self.assertEqual(len(self.game.wells), 0)

    def test_collecting_beacon_increases_score_and_combo(self):
        self.game.beacon_position = self.game.comet_position.copy()
        self.game.step(0.01)

        self.assertEqual(self.game.score, 100)
        self.assertEqual(self.game.combo, 1)
        self.assertEqual(self.game.best_score, 100)

    def test_gravity_well_changes_comet_trajectory(self):
        baseline = OrbitKeeper(960, 640, seed=31)
        steered = OrbitKeeper(960, 640, seed=31)
        steered.add_well(
            steered.comet_position + np.array([0.0, -120.0], dtype=np.float64)
        )

        for _ in range(60):
            baseline.step(1.0 / 60.0)
            steered.step(1.0 / 60.0)

        displacement = np.linalg.norm(
            baseline.comet_position - steered.comet_position
        )
        self.assertGreater(displacement, 25.0)

    def test_core_collisions_consume_lives_and_end_round(self):
        for expected_lives in (2, 1, 0):
            self.game.comet_position = self.game.center.copy()
            self.game.comet_velocity[:] = 0.0
            self.game.step(0.01)
            self.assertEqual(self.game.lives, expected_lives)

        self.assertTrue(self.game.game_over)

    def test_timer_reset_control_and_render(self):
        self.game.step(0.25)
        self.assertAlmostEqual(self.game.remaining_seconds, 9.75, places=2)
        self.game.score = 230
        self.game.best_score = 230

        frame = self.game.render(2.0)
        self.assertEqual(frame.shape, (640, 960, 3))
        self.assertGreater(float(frame.std()), 5.0)

        reset = np.asarray(self.game.reset_center, dtype=np.float32)
        self.assertTrue(self.game.update(reset, True, 2.1, 0.0))
        self.assertEqual(self.game.score, 0)
        self.assertEqual(self.game.best_score, 230)
        self.assertEqual(self.game.lives, 3)


if __name__ == "__main__":
    unittest.main()
