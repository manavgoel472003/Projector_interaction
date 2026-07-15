from __future__ import annotations

import math
import time

import cv2
import numpy as np


class PrismConnectFour:
    ROWS = 6
    COLUMNS = 7
    CYAN = 1
    MAGENTA = -1

    def __init__(
        self,
        width: int,
        height: int,
        release_seconds: float = 0.16,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.release_seconds = float(release_seconds)
        self.cell_size = max(
            42,
            int(
                min(
                    self.width * 0.70 / self.COLUMNS,
                    self.height * 0.62 / self.ROWS,
                )
            ),
        )
        self.board_width = self.cell_size * self.COLUMNS
        self.board_height = self.cell_size * self.ROWS
        self.board_left = (self.width - self.board_width) // 2
        self.board_top = int(self.height * 0.255)
        self.board_right = self.board_left + self.board_width
        self.board_bottom = self.board_top + self.board_height
        self.reset_center = (
            min(self.width - 62, self.board_right + 82),
            self.board_top + 42,
        )
        self.reset_hit_radius = max(42, int(self.cell_size * 0.38))
        self._background = self._make_background()
        self.clear()

    def clear(self) -> None:
        self.scores = {self.CYAN: 0, self.MAGENTA: 0}
        self.draws = 0
        self._starting_player = self.CYAN
        self._touch_latched = False
        self._last_active_time: float | None = None
        self.reset_round(alternate_start=False)

    def reset_round(self, alternate_start: bool = True) -> None:
        if alternate_start:
            self._starting_player *= -1
        self.board = np.zeros((self.ROWS, self.COLUMNS), dtype=np.int8)
        self.current_player = self._starting_player
        self.winner = 0
        self.is_draw = False
        self.winning_cells: tuple[tuple[int, int], ...] | None = None
        self.last_drop: tuple[int, int, int] | None = None
        self.last_drop_time = -1e9

    def column_at(self, point: np.ndarray) -> int | None:
        x, y = np.asarray(point, dtype=float)
        if not (
            self.board_left <= x < self.board_right
            and self.board_top <= y < self.board_bottom
        ):
            return None
        return min(
            self.COLUMNS - 1,
            int((x - self.board_left) // self.cell_size),
        )

    def drop(self, column: int, timestamp: float | None = None) -> bool:
        if self.winner or self.is_draw or not 0 <= column < self.COLUMNS:
            return False
        empty_rows = np.flatnonzero(self.board[:, column] == 0)
        if empty_rows.size == 0:
            return False

        row = int(empty_rows[-1])
        player = self.current_player
        self.board[row, column] = player
        self.last_drop = (row, column, player)
        self.last_drop_time = time.monotonic() if timestamp is None else timestamp
        self._update_result(row, column, player)
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
        column = self.column_at(point)
        return column is not None and self.drop(column, timestamp)

    def render(self, timestamp: float | None = None) -> np.ndarray:
        now = time.monotonic() if timestamp is None else timestamp
        frame = self._background.copy()
        glow = np.zeros_like(frame)
        self._draw_header(frame)
        self._draw_board(frame, glow, now)
        self._draw_reset(frame, glow)
        blurred = cv2.GaussianBlur(glow, (0, 0), sigmaX=13, sigmaY=13)
        frame = cv2.addWeighted(frame, 1.0, blurred, 0.74, 0.0)
        return cv2.addWeighted(frame, 1.0, glow, 0.88, 0.0)

    def _update_result(self, row: int, column: int, player: int) -> None:
        for delta_row, delta_column in ((0, 1), (1, 0), (1, 1), (1, -1)):
            cells = [(row, column)]
            for direction in (-1, 1):
                candidate_row = row + direction * delta_row
                candidate_column = column + direction * delta_column
                directional: list[tuple[int, int]] = []
                while (
                    0 <= candidate_row < self.ROWS
                    and 0 <= candidate_column < self.COLUMNS
                    and self.board[candidate_row, candidate_column] == player
                ):
                    directional.append((candidate_row, candidate_column))
                    candidate_row += direction * delta_row
                    candidate_column += direction * delta_column
                if direction < 0:
                    cells = list(reversed(directional)) + cells
                else:
                    cells.extend(directional)
            if len(cells) >= 4:
                self.winner = player
                self.winning_cells = tuple(cells)
                self.scores[player] += 1
                return
        if np.all(self.board != 0):
            self.is_draw = True
            self.draws += 1

    def _make_background(self) -> np.ndarray:
        yy, xx = np.mgrid[: self.height, : self.width]
        nx = (xx - self.width * 0.5) / max(self.width, 1)
        ny = (yy - self.height * 0.52) / max(self.height, 1)
        light = np.clip(1.0 - 1.35 * np.sqrt(nx * nx + ny * ny), 0.0, 1.0)
        background = np.empty((self.height, self.width, 3), dtype=np.float32)
        background[..., 0] = 15 + 16 * light
        background[..., 1] = 18 + 12 * light
        background[..., 2] = 22 + 10 * light
        rng = np.random.default_rng(2407)
        background += rng.normal(0.0, 1.8, (self.height, self.width, 1))
        return np.clip(background, 0, 255).astype(np.uint8)

    def _draw_header(self, frame: np.ndarray) -> None:
        self._center_text(
            frame,
            "PRISM FOUR",
            int(self.height * 0.075),
            1.14,
            (244, 241, 247),
            2,
        )
        if self.winner:
            name = "CYAN" if self.winner == self.CYAN else "MAGENTA"
            status = f"{name} CONNECTS FOUR"
            color = self._player_color(self.winner)
        elif self.is_draw:
            status = "BOARD DRAW"
            color = (215, 220, 228)
        else:
            name = "CYAN" if self.current_player == self.CYAN else "MAGENTA"
            status = f"{name} TO DROP"
            color = self._player_color(self.current_player)
        self._center_text(
            frame,
            status,
            int(self.height * 0.145),
            0.76,
            color,
            2,
        )
        score = (
            f"CYAN  {self.scores[self.CYAN]}     "
            f"DRAWS  {self.draws}     "
            f"MAGENTA  {self.scores[self.MAGENTA]}"
        )
        self._center_text(
            frame,
            score,
            int(self.height * 0.205),
            0.52,
            (176, 182, 192),
            1,
        )

    def _draw_board(
        self,
        frame: np.ndarray,
        glow: np.ndarray,
        now: float,
    ) -> None:
        cv2.rectangle(
            frame,
            (self.board_left - 10, self.board_top - 10),
            (self.board_right + 10, self.board_bottom + 10),
            (29, 32, 40),
            -1,
        )
        cv2.rectangle(
            frame,
            (self.board_left - 10, self.board_top - 10),
            (self.board_right + 10, self.board_bottom + 10),
            (168, 176, 188),
            2,
            cv2.LINE_AA,
        )

        animation_age = now - self.last_drop_time
        animating = self.last_drop is not None and 0.0 <= animation_age < 0.42
        for row in range(self.ROWS):
            for column in range(self.COLUMNS):
                center = self._cell_center(row, column)
                radius = int(self.cell_size * 0.37)
                cv2.circle(frame, center, radius, (14, 16, 22), -1, cv2.LINE_AA)
                cv2.circle(
                    frame,
                    center,
                    radius,
                    (105, 112, 126),
                    max(2, self.cell_size // 42),
                    cv2.LINE_AA,
                )
                player = int(self.board[row, column])
                if not player:
                    continue
                if animating and self.last_drop[:2] == (row, column):
                    continue
                self._draw_piece(frame, glow, center, radius, player)

        if animating:
            row, column, player = self.last_drop
            target_x, target_y = self._cell_center(row, column)
            start_y = self.board_top - self.cell_size // 2
            progress = np.clip(animation_age / 0.42, 0.0, 1.0)
            eased = 1.0 - (1.0 - progress) ** 3
            moving_y = int(start_y + (target_y - start_y) * eased)
            self._draw_piece(
                frame,
                glow,
                (target_x, moving_y),
                int(self.cell_size * 0.37),
                player,
            )

        if self.winning_cells:
            start = self._cell_center(*self.winning_cells[0])
            end = self._cell_center(*self.winning_cells[-1])
            color = self._player_color(self.winner)
            cv2.line(
                glow,
                start,
                end,
                color,
                max(20, self.cell_size // 5),
                cv2.LINE_AA,
            )
            cv2.line(
                frame,
                start,
                end,
                (250, 248, 240),
                max(4, self.cell_size // 22),
                cv2.LINE_AA,
            )

    def _draw_piece(
        self,
        frame: np.ndarray,
        glow: np.ndarray,
        center: tuple[int, int],
        radius: int,
        player: int,
    ) -> None:
        color = self._player_color(player)
        cv2.circle(glow, center, radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, center, radius, color, -1, cv2.LINE_AA)
        cv2.circle(
            frame,
            center,
            int(radius * 0.68),
            tuple(max(0, channel - 34) for channel in color),
            -1,
            cv2.LINE_AA,
        )
        cv2.ellipse(
            frame,
            center,
            (int(radius * 0.70), int(radius * 0.70)),
            0,
            205,
            302,
            (250, 250, 246),
            max(2, radius // 9),
            cv2.LINE_AA,
        )

    def _draw_reset(self, frame: np.ndarray, glow: np.ndarray) -> None:
        center = self.reset_center
        radius = max(23, int(self.cell_size * 0.27))
        color = (202, 207, 216)
        cv2.ellipse(glow, center, (radius, radius), 0, 35, 320, color, 8, cv2.LINE_AA)
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
            center[1] + radius + 31,
            0.38,
            (154, 160, 170),
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
        return (255, 196, 48) if player == PrismConnectFour.CYAN else (210, 70, 252)

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
        size, _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_DUPLEX,
            scale,
            thickness,
        )
        center = image.shape[1] // 2 if center_x is None else center_x
        cv2.putText(
            image,
            text,
            (center - size[0] // 2, baseline_y),
            cv2.FONT_HERSHEY_DUPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
