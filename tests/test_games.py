import unittest

import numpy as np

from wall_touch_games import TicTacToe


class TicTacToeTests(unittest.TestCase):
    def setUp(self):
        self.game = TicTacToe(900, 700, release_seconds=0.10)

    def point(self, row, column):
        return np.asarray(self.game._cell_center(row, column), dtype=np.float32)

    def test_places_alternating_marks_and_rejects_occupied_cell(self):
        self.assertTrue(self.game.place(self.point(0, 0), 1.0))
        self.assertEqual(self.game.board[0, 0], TicTacToe.X)
        self.assertFalse(self.game.place(self.point(0, 0), 1.1))
        self.assertTrue(self.game.place(self.point(1, 1), 1.2))
        self.assertEqual(self.game.board[1, 1], TicTacToe.O)

    def test_detects_win_and_keeps_score_for_new_round(self):
        for row, column in ((0, 0), (1, 0), (0, 1), (1, 1), (0, 2)):
            self.assertTrue(self.game.place(self.point(row, column), 2.0))

        self.assertEqual(self.game.winner, TicTacToe.X)
        self.assertEqual(self.game.scores[TicTacToe.X], 1)
        self.assertIsNotNone(self.game.winning_line)

        self.game.reset_round()
        self.assertFalse(np.any(self.game.board))
        self.assertEqual(self.game.scores[TicTacToe.X], 1)
        self.assertEqual(self.game.current_player, TicTacToe.O)

    def test_one_move_requires_press_then_release(self):
        self.assertTrue(self.game.update(self.point(0, 0), True, 3.0))
        self.assertFalse(self.game.update(self.point(0, 1), True, 3.2))
        self.assertEqual(np.count_nonzero(self.game.board), 1)

        self.game.update(None, False, 3.35)
        self.assertTrue(self.game.update(self.point(0, 1), True, 3.36))
        self.assertEqual(np.count_nonzero(self.game.board), 2)

    def test_projected_reset_control_and_render(self):
        self.game.place(self.point(2, 2), 4.0)
        reset = np.asarray(self.game.reset_center, dtype=np.float32)
        self.assertTrue(self.game.update(reset, True, 4.2))
        self.assertFalse(np.any(self.game.board))

        frame = self.game.render(4.3)
        self.assertEqual(frame.shape, (700, 900, 3))
        self.assertGreater(float(frame.std()), 5.0)


if __name__ == "__main__":
    unittest.main()
