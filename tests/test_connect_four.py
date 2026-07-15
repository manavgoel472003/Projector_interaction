import unittest

import numpy as np

from wall_touch_connect_four import PrismConnectFour


class PrismConnectFourTests(unittest.TestCase):
    def setUp(self):
        self.game = PrismConnectFour(1280, 720, release_seconds=0.10)

    def column_point(self, column):
        return np.array(
            [
                self.game.board_left + (column + 0.5) * self.game.cell_size,
                self.game.board_top + self.game.cell_size,
            ],
            dtype=np.float32,
        )

    def test_pieces_drop_from_bottom_and_full_column_is_rejected(self):
        for index in range(self.game.ROWS):
            self.assertTrue(self.game.drop(0, float(index)))
        self.assertFalse(self.game.drop(0, 7.0))
        self.assertTrue(np.all(self.game.board[:, 0] != 0))
        self.assertEqual(self.game.board[-1, 0], PrismConnectFour.CYAN)

    def test_detects_horizontal_and_vertical_wins(self):
        for column in (0, 0, 1, 1, 2, 2, 3):
            self.assertTrue(self.game.drop(column, 1.0))
        self.assertEqual(self.game.winner, PrismConnectFour.CYAN)
        self.assertEqual(len(self.game.winning_cells), 4)

        self.game.clear()
        for column in (0, 1, 0, 1, 0, 1, 0):
            self.assertTrue(self.game.drop(column, 2.0))
        self.assertEqual(self.game.winner, PrismConnectFour.CYAN)

    def test_detects_diagonal_win(self):
        for column in (0, 1, 1, 2, 3, 2, 2, 3, 4, 3, 3):
            self.assertTrue(self.game.drop(column, 3.0))
        self.assertEqual(self.game.winner, PrismConnectFour.CYAN)
        self.assertEqual(len(self.game.winning_cells), 4)

    def test_touch_latch_reset_and_render(self):
        self.assertTrue(self.game.update(self.column_point(2), True, 4.0))
        self.assertFalse(self.game.update(self.column_point(3), True, 4.2))
        self.assertEqual(np.count_nonzero(self.game.board), 1)
        self.game.update(None, False, 4.35)
        self.assertTrue(self.game.update(self.column_point(3), True, 4.36))

        frame = self.game.render(4.5)
        self.assertEqual(frame.shape, (720, 1280, 3))
        self.assertGreater(float(frame.std()), 5.0)

        self.game.update(None, False, 4.6)
        reset = np.asarray(self.game.reset_center, dtype=np.float32)
        self.assertTrue(self.game.update(reset, True, 4.72))
        self.assertFalse(np.any(self.game.board))


if __name__ == "__main__":
    unittest.main()
