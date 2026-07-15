from __future__ import annotations

import math
import time

import cv2
import numpy as np


WINNING_LINES = (
    ((0, 0), (0, 1), (0, 2)),
    ((1, 0), (1, 1), (1, 2)),
    ((2, 0), (2, 1), (2, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 1), (1, 1), (2, 1)),
    ((0, 2), (1, 2), (2, 2)),
    ((0, 0), (1, 1), (2, 2)),
    ((0, 2), (1, 1), (2, 0)),
)


class TicTacToe:
    X = 1
    O = -1

    def __init__(
        self,
        width: int,
        height: int,
        release_seconds: float = 0.16,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.release_seconds = float(release_seconds)
        self.board_size = int(min(self.width * 0.52, self.height * 0.64))
        self.board_size -= self.board_size % 3
        self.cell_size = self.board_size // 3
        self.board_left = (self.width - self.board_size) // 2
        self.board_top = int(self.height * 0.23)
        self.board_right = self.board_left + self.board_size
        self.board_bottom = self.board_top + self.board_size
        self.reset_center = (
            min(self.width - 62, self.board_right + 86),
            self.board_top + 44,
        )
        self.reset_hit_radius = max(44, int(self.cell_size * 0.18))
        self._background = self._make_background()
        self.clear()

    def clear(self) -> None:
        self.scores = {self.X: 0, self.O: 0}
        self.draws = 0
        self._starting_player = self.X
        self._touch_latched = False
        self._last_active_time: float | None = None
        self.reset_round(alternate_start=False)

    def reset_round(self, alternate_start: bool = True) -> None:
        if alternate_start:
            self._starting_player *= -1
        self.board = np.zeros((3, 3), dtype=np.int8)
        self.current_player = self._starting_player
        self.winner = 0
        self.winning_line: tuple[tuple[int, int], ...] | None = None
        self.is_draw = False
        self.last_move: tuple[int, int] | None = None
        self.last_move_time = -1e9

    def cell_at(self, point: np.ndarray) -> tuple[int, int] | None:
        x, y = np.asarray(point, dtype=float)
        if not (
            self.board_left <= x < self.board_right
            and self.board_top <= y < self.board_bottom
        ):
            return None
        column = min(2, int((x - self.board_left) // self.cell_size))
        row = min(2, int((y - self.board_top) // self.cell_size))
        return row, column

    def place(self, point: np.ndarray, timestamp: float | None = None) -> bool:
        if self.winner or self.is_draw:
            return False
        cell = self.cell_at(point)
        if cell is None or self.board[cell] != 0:
            return False

        self.board[cell] = self.current_player
        self.last_move = cell
        self.last_move_time = time.monotonic() if timestamp is None else timestamp
        self._update_result()
        if not self.winner and not self.is_draw:
            self.current_player *= -1
        return True

    def update(
        self,
        point: np.ndarray | None,
        active: bool,
        timestamp: float,
    ) -> bool:
        if not active or point is None:
            if (
                self._last_active_time is None
                or timestamp - self._last_active_time >= self.release_seconds
            ):
                self._touch_latched = False
            return False

        self._last_active_time = timestamp
        if self._touch_latched:
            return False
        self._touch_latched = True

        reset_distance = np.linalg.norm(
            np.asarray(point, dtype=float) - self.reset_center
        )
        if reset_distance <= self.reset_hit_radius:
            self.reset_round()
            return True
        return self.place(point, timestamp)

    def render(self, timestamp: float | None = None) -> np.ndarray:
        now = time.monotonic() if timestamp is None else timestamp
        frame = self._background.copy()
        glow = np.zeros_like(frame)

        self._draw_header(frame)
        self._draw_board(frame, glow, now)
        self._draw_reset(frame, glow)
        if self.winning_line is not None:
            self._draw_winning_line(frame, glow)

        blurred = cv2.GaussianBlur(glow, (0, 0), sigmaX=14, sigmaY=14)
        frame = cv2.addWeighted(frame, 1.0, blurred, 0.72, 0.0)
        frame = cv2.addWeighted(frame, 1.0, glow, 0.90, 0.0)
        return frame

    def _update_result(self) -> None:
        for line in WINNING_LINES:
            values = [int(self.board[cell]) for cell in line]
            if values[0] != 0 and values.count(values[0]) == 3:
                self.winner = values[0]
                self.winning_line = line
                self.scores[self.winner] += 1
                return
        if np.all(self.board != 0):
            self.is_draw = True
            self.draws += 1

    def _make_background(self) -> np.ndarray:
        yy, xx = np.mgrid[: self.height, : self.width]
        nx = (xx - self.width * 0.5) / max(self.width, 1)
        ny = (yy - self.height * 0.48) / max(self.height, 1)
        vignette = np.clip(1.0 - 1.15 * np.sqrt(nx * nx + ny * ny), 0.0, 1.0)
        background = np.empty((self.height, self.width, 3), dtype=np.float32)
        background[..., 0] = 18 + 12 * vignette
        background[..., 1] = 14 + 8 * vignette
        background[..., 2] = 24 + 12 * vignette

        rng = np.random.default_rng(20260715)
        grain = rng.normal(0.0, 2.2, (self.height, self.width, 1))
        background += grain
        return np.clip(background, 0, 255).astype(np.uint8)

    def _draw_header(self, frame: np.ndarray) -> None:
        self._center_text(frame, "TIC TAC TOE", int(self.height * 0.075), 1.15, (242, 238, 246), 2)
        if self.winner:
            symbol = "X" if self.winner == self.X else "O"
            status = f"{symbol} TAKES THE ROUND"
            color = self._player_color(self.winner)
        elif self.is_draw:
            status = "ROUND DRAW"
            color = (208, 218, 224)
        else:
            symbol = "X" if self.current_player == self.X else "O"
            status = f"{symbol} TO MOVE"
            color = self._player_color(self.current_player)
        self._center_text(frame, status, int(self.height * 0.145), 0.80, color, 2)

        score_y = int(self.height * 0.195)
        score_text = (
            f"X  {self.scores[self.X]}     "
            f"DRAWS  {self.draws}     "
            f"O  {self.scores[self.O]}"
        )
        self._center_text(frame, score_text, score_y, 0.58, (174, 178, 190), 1)

    def _draw_board(self, frame: np.ndarray, glow: np.ndarray, now: float) -> None:
        gap = max(7, int(self.cell_size * 0.035))
        cell_colors = ((33, 30, 43), (38, 31, 39))
        for row in range(3):
            for column in range(3):
                x0 = self.board_left + column * self.cell_size + gap
                y0 = self.board_top + row * self.cell_size + gap
                x1 = self.board_left + (column + 1) * self.cell_size - gap
                y1 = self.board_top + (row + 1) * self.cell_size - gap
                cv2.rectangle(frame, (x0, y0), (x1, y1), cell_colors[(row + column) % 2], -1)
                player = int(self.board[row, column])
                if player:
                    emphasis = 1.0
                    if self.last_move == (row, column):
                        age = max(0.0, now - self.last_move_time)
                        emphasis += 0.12 * math.exp(-age * 2.2) * (1.0 + math.sin(age * 9.0))
                    self._draw_mark(
                        frame,
                        glow,
                        player,
                        (x0, y0, x1, y1),
                        emphasis,
                    )

    def _draw_mark(
        self,
        frame: np.ndarray,
        glow: np.ndarray,
        player: int,
        bounds: tuple[int, int, int, int],
        emphasis: float,
    ) -> None:
        x0, y0, x1, y1 = bounds
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        radius = int(min(x1 - x0, y1 - y0) * 0.29 * emphasis)
        color = self._player_color(player)
        glow_color = tuple(int(channel * 0.72) for channel in color)
        thick = max(8, int(self.cell_size * 0.052))
        if player == self.X:
            offset = int(radius * 0.72)
            first = ((cx - offset, cy - offset), (cx + offset, cy + offset))
            second = ((cx + offset, cy - offset), (cx - offset, cy + offset))
            for start, end in (first, second):
                cv2.line(glow, start, end, glow_color, thick * 3, cv2.LINE_AA)
                cv2.line(frame, start, end, color, thick, cv2.LINE_AA)
                cv2.line(frame, start, end, (246, 247, 250), max(2, thick // 5), cv2.LINE_AA)
        else:
            cv2.circle(glow, (cx, cy), radius, glow_color, thick * 3, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), radius, color, thick, cv2.LINE_AA)
            cv2.ellipse(
                frame,
                (cx, cy),
                (radius, radius),
                0,
                205,
                305,
                (250, 250, 248),
                max(2, thick // 4),
                cv2.LINE_AA,
            )

    def _draw_winning_line(self, frame: np.ndarray, glow: np.ndarray) -> None:
        start_cell = self.winning_line[0]
        end_cell = self.winning_line[-1]
        start = self._cell_center(*start_cell)
        end = self._cell_center(*end_cell)
        color = self._player_color(self.winner)
        cv2.line(glow, start, end, color, max(22, self.cell_size // 10), cv2.LINE_AA)
        cv2.line(frame, start, end, (248, 244, 235), max(5, self.cell_size // 35), cv2.LINE_AA)

    def _draw_reset(self, frame: np.ndarray, glow: np.ndarray) -> None:
        center = self.reset_center
        radius = max(24, int(self.cell_size * 0.12))
        color = (196, 202, 214)
        cv2.ellipse(glow, center, (radius, radius), 0, 35, 320, color, 9, cv2.LINE_AA)
        cv2.ellipse(frame, center, (radius, radius), 0, 35, 320, color, 3, cv2.LINE_AA)
        angle = math.radians(35)
        tip = (
            int(center[0] + radius * math.cos(angle)),
            int(center[1] + radius * math.sin(angle)),
        )
        wing = max(8, radius // 3)
        triangle = np.array(
            [tip, (tip[0] - wing, tip[1] - 2), (tip[0] - 2, tip[1] + wing)],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(frame, triangle, color, cv2.LINE_AA)
        self._center_text(
            frame,
            "NEW ROUND",
            center[1] + radius + 34,
            0.40,
            (150, 154, 166),
            1,
            center_x=center[0],
        )

    def _cell_center(self, row: int, column: int) -> tuple[int, int]:
        return (
            self.board_left + column * self.cell_size + self.cell_size // 2,
            self.board_top + row * self.cell_size + self.cell_size // 2,
        )

    @staticmethod
    def _player_color(player: int) -> tuple[int, int, int]:
        return (255, 191, 55) if player == TicTacToe.X else (92, 92, 255)

    @staticmethod
    def _center_text(
        image: np.ndarray,
        text: str,
        baseline_y: int,
        scale: float,
        color: tuple[int, int, int],
        thickness: int,
        center_x: int | None = None,
    ) -> None:
        size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thickness)
        center = image.shape[1] // 2 if center_x is None else center_x
        origin = (center - size[0] // 2, baseline_y)
        cv2.putText(
            image,
            text,
            origin,
            cv2.FONT_HERSHEY_DUPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
