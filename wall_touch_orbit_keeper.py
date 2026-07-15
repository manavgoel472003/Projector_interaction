from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class GravityWell:
    position: np.ndarray
    life_remaining: float


class OrbitKeeper:
    def __init__(
        self,
        width: int,
        height: int,
        release_seconds: float = 0.16,
        round_seconds: float = 45.0,
        well_lifetime: float = 2.2,
        seed: int = 472003,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.scale = float(min(width, height))
        self.release_seconds = float(release_seconds)
        self.round_seconds = float(round_seconds)
        self.well_lifetime = float(well_lifetime)
        self.play_left = int(self.width * 0.065)
        self.play_right = int(self.width * 0.935)
        self.play_top = int(self.height * 0.19)
        self.play_bottom = int(self.height * 0.93)
        self.center = np.array(
            [self.width * 0.50, self.height * 0.57],
            dtype=np.float64,
        )
        self.core_radius = max(24.0, self.scale * 0.050)
        self.beacon_radius = max(25.0, self.scale * 0.038)
        self.comet_radius = max(9, int(self.scale * 0.011))
        self.softening = max(55.0, self.scale * 0.060)
        self.central_strength = self.scale**3 * 0.0078
        self.well_strength = self.scale**3 * 0.0135
        self.maximum_speed = self.scale * 0.36
        self.reset_center = (
            self.width - max(68, int(self.width * 0.055)),
            max(62, int(self.height * 0.095)),
        )
        self.reset_hit_radius = max(42, int(self.scale * 0.042))
        self._rng = np.random.default_rng(seed)
        self._background = self._make_background()
        self.clear()

    def clear(self) -> None:
        self.best_score = 0
        self._touch_latched = False
        self._last_active_time: float | None = None
        self.reset_round()

    def reset_round(self) -> None:
        self.score = 0
        self.combo = 0
        self.lives = 3
        self.elapsed = 0.0
        self.game_over = False
        self.wells: list[GravityWell] = []
        self.trail: deque[np.ndarray] = deque(maxlen=72)
        self._spawn_comet()
        self._spawn_beacon()

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.round_seconds - self.elapsed)

    def add_well(self, point: np.ndarray) -> bool:
        position = np.asarray(point, dtype=np.float64)
        if not self._inside_playfield(position) or self.game_over:
            return False
        if len(self.wells) >= 3:
            self.wells.pop(0)
        self.wells.append(GravityWell(position.copy(), self.well_lifetime))
        return True

    def update(
        self,
        point: np.ndarray | None,
        active: bool,
        timestamp: float,
        delta_seconds: float,
    ) -> bool:
        self.step(delta_seconds)
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
        return self.add_well(point)

    def step(self, delta_seconds: float) -> None:
        if self.game_over:
            return
        elapsed_delta = float(np.clip(delta_seconds, 0.0, 0.50))
        if elapsed_delta <= 0.0:
            return

        self.elapsed += elapsed_delta
        for well in self.wells:
            well.life_remaining -= elapsed_delta
        self.wells = [well for well in self.wells if well.life_remaining > 0.0]
        if self.elapsed >= self.round_seconds:
            self._finish_round()
            return

        delta = min(elapsed_delta, 0.12)
        steps = max(1, int(math.ceil(delta / 0.014)))
        step_delta = delta / steps
        for _ in range(steps):
            acceleration = self._gravity_from(
                self.center,
                self.central_strength,
            )
            for well in self.wells:
                life_ratio = np.clip(
                    well.life_remaining / self.well_lifetime,
                    0.0,
                    1.0,
                )
                acceleration += self._gravity_from(
                    well.position,
                    self.well_strength * (0.35 + 0.65 * life_ratio),
                )

            self.comet_velocity += acceleration * step_delta
            speed = float(np.linalg.norm(self.comet_velocity))
            if speed > self.maximum_speed:
                self.comet_velocity *= self.maximum_speed / speed
            self.comet_velocity *= 0.9994 ** (step_delta * 60.0)
            self.comet_position += self.comet_velocity * step_delta

            if (
                np.linalg.norm(self.comet_position - self.beacon_position)
                <= self.beacon_radius + self.comet_radius
            ):
                self._collect_beacon()
            if np.linalg.norm(self.comet_position - self.center) <= self.core_radius:
                self._lose_life()
                break
            if not self._inside_playfield(self.comet_position, margin=-self.comet_radius):
                self._lose_life()
                break

        self.trail.append(self.comet_position.copy())

    def render(self, timestamp: float) -> np.ndarray:
        frame = self._background.copy()
        glow = np.zeros_like(frame)
        self._draw_header(frame)
        self._draw_playfield(frame)
        self._draw_orbits(frame, timestamp)
        self._draw_wells(frame, glow, timestamp)
        self._draw_beacon(frame, glow, timestamp)
        self._draw_comet(frame, glow)
        self._draw_core(frame, glow, timestamp)
        self._draw_reset(frame, glow)
        if self.game_over:
            self._draw_game_over(frame)

        blurred = cv2.GaussianBlur(glow, (0, 0), sigmaX=14, sigmaY=14)
        frame = cv2.addWeighted(frame, 1.0, blurred, 0.76, 0.0)
        return cv2.addWeighted(frame, 1.0, glow, 0.86, 0.0)

    def _gravity_from(self, source: np.ndarray, strength: float) -> np.ndarray:
        offset = source - self.comet_position
        softened = float(np.dot(offset, offset) + self.softening**2)
        return strength * offset / (softened**1.5)

    def _spawn_comet(self) -> None:
        orbit_radius = self.scale * 0.21
        speed = self.scale * 0.19
        self.comet_position = self.center + np.array(
            [orbit_radius, 0.0],
            dtype=np.float64,
        )
        self.comet_velocity = np.array([0.0, -speed], dtype=np.float64)
        if hasattr(self, "trail"):
            self.trail.clear()

    def _spawn_beacon(self) -> None:
        minimum_core_distance = self.scale * 0.19
        minimum_comet_distance = self.scale * 0.16
        margin = self.beacon_radius * 1.8
        for _ in range(80):
            candidate = np.array(
                [
                    self._rng.uniform(self.play_left + margin, self.play_right - margin),
                    self._rng.uniform(self.play_top + margin, self.play_bottom - margin),
                ],
                dtype=np.float64,
            )
            if (
                np.linalg.norm(candidate - self.center) >= minimum_core_distance
                and np.linalg.norm(candidate - self.comet_position)
                >= minimum_comet_distance
            ):
                self.beacon_position = candidate
                return
        self.beacon_position = np.array(
            [self.play_left + margin, self.play_top + margin],
            dtype=np.float64,
        )

    def _collect_beacon(self) -> None:
        self.score += 100 + self.combo * 30
        self.combo += 1
        self.best_score = max(self.best_score, self.score)
        self._spawn_beacon()

    def _lose_life(self) -> None:
        self.lives -= 1
        self.combo = 0
        if self.lives <= 0:
            self._finish_round()
            return
        self._spawn_comet()

    def _finish_round(self) -> None:
        self.game_over = True
        self.best_score = max(self.best_score, self.score)

    def _inside_playfield(self, point: np.ndarray, margin: float = 0.0) -> bool:
        x, y = np.asarray(point, dtype=float)
        return bool(
            self.play_left - margin <= x <= self.play_right + margin
            and self.play_top - margin <= y <= self.play_bottom + margin
        )

    def _make_background(self) -> np.ndarray:
        yy, xx = np.mgrid[: self.height, : self.width]
        nx = (xx - self.width * 0.48) / max(self.width, 1)
        ny = (yy - self.height * 0.55) / max(self.height, 1)
        light = np.clip(1.0 - 1.45 * np.sqrt(nx * nx + ny * ny), 0.0, 1.0)
        background = np.empty((self.height, self.width, 3), dtype=np.float32)
        background[..., 0] = 21 + 11 * light
        background[..., 1] = 13 + 12 * light
        background[..., 2] = 20 + 8 * light
        background += self._rng.normal(0.0, 1.7, (self.height, self.width, 1))
        result = np.clip(background, 0, 255).astype(np.uint8)

        star_count = max(90, int(self.width * self.height / 13_000))
        for _ in range(star_count):
            x = int(self._rng.integers(0, self.width))
            y = int(self._rng.integers(self.play_top, self.height))
            brightness = int(self._rng.integers(80, 170))
            radius = 1 if self._rng.random() < 0.88 else 2
            cv2.circle(
                result,
                (x, y),
                radius,
                (brightness, brightness, min(220, brightness + 22)),
                -1,
                cv2.LINE_AA,
            )
        return result

    def _draw_header(self, frame: np.ndarray) -> None:
        self._center_text(
            frame,
            "ORBIT KEEPER",
            int(self.height * 0.070),
            1.10,
            (244, 241, 247),
            2,
        )
        status = (
            f"SCORE  {self.score:04d}     "
            f"BEST  {self.best_score:04d}     "
            f"TIME  {int(math.ceil(self.remaining_seconds)):02d}"
        )
        self._center_text(
            frame,
            status,
            int(self.height * 0.132),
            0.58,
            (182, 190, 204),
            1,
        )
        life_y = int(self.height * 0.165)
        spacing = max(24, int(self.scale * 0.025))
        start_x = self.width // 2 - spacing
        for index in range(3):
            color = (255, 196, 70) if index < self.lives else (60, 58, 68)
            x = start_x + index * spacing
            diamond = np.array(
                [(x, life_y - 7), (x + 7, life_y), (x, life_y + 7), (x - 7, life_y)],
                dtype=np.int32,
            )
            cv2.fillConvexPoly(frame, diamond, color, cv2.LINE_AA)

    def _draw_playfield(self, frame: np.ndarray) -> None:
        cv2.rectangle(
            frame,
            (self.play_left, self.play_top),
            (self.play_right, self.play_bottom),
            (96, 94, 112),
            2,
            cv2.LINE_AA,
        )

    def _draw_orbits(self, frame: np.ndarray, timestamp: float) -> None:
        center = tuple(np.rint(self.center).astype(int))
        for index, ratio in enumerate((0.13, 0.21, 0.30)):
            radius = int(self.scale * ratio)
            angle = (timestamp * (8 + index * 4)) % 360
            cv2.ellipse(
                frame,
                center,
                (radius, int(radius * 0.68)),
                angle,
                0,
                360,
                (52 + index * 7, 48 + index * 5, 68 + index * 7),
                1,
                cv2.LINE_AA,
            )

    def _draw_wells(
        self,
        frame: np.ndarray,
        glow: np.ndarray,
        timestamp: float,
    ) -> None:
        del timestamp
        for well in self.wells:
            center = tuple(np.rint(well.position).astype(int))
            life_ratio = float(
                np.clip(well.life_remaining / self.well_lifetime, 0.0, 1.0)
            )
            phase = 1.0 - life_ratio
            color = (
                int(170 + 70 * life_ratio),
                int(80 + 85 * life_ratio),
                int(205 + 45 * life_ratio),
            )
            for ring in range(3):
                radius = int(self.scale * (0.025 + 0.024 * ((phase + ring / 3) % 1.0)))
                cv2.circle(glow, center, radius, color, 7, cv2.LINE_AA)
                cv2.circle(frame, center, radius, color, 2, cv2.LINE_AA)
            cv2.circle(frame, center, 5, (252, 248, 255), -1, cv2.LINE_AA)

    def _draw_beacon(
        self,
        frame: np.ndarray,
        glow: np.ndarray,
        timestamp: float,
    ) -> None:
        center = tuple(np.rint(self.beacon_position).astype(int))
        pulse = 1.0 + 0.08 * math.sin(timestamp * 4.5)
        radius = int(self.beacon_radius * pulse)
        color = (95, 235, 255)
        cv2.circle(glow, center, radius, color, 10, cv2.LINE_AA)
        cv2.circle(frame, center, radius, color, 3, cv2.LINE_AA)
        cv2.circle(frame, center, max(4, radius // 5), (245, 255, 255), -1, cv2.LINE_AA)
        for angle in (timestamp * 70, timestamp * 70 + 180):
            radians = math.radians(angle)
            satellite = (
                int(center[0] + radius * 1.35 * math.cos(radians)),
                int(center[1] + radius * 1.35 * math.sin(radians)),
            )
            cv2.circle(frame, satellite, 4, (140, 215, 255), -1, cv2.LINE_AA)

    def _draw_comet(self, frame: np.ndarray, glow: np.ndarray) -> None:
        if len(self.trail) >= 2:
            points = np.rint(np.stack(self.trail)).astype(np.int32)
            cv2.polylines(
                glow,
                [points],
                False,
                (255, 130, 80),
                max(8, self.comet_radius),
                cv2.LINE_AA,
            )
            cv2.polylines(
                frame,
                [points],
                False,
                (210, 110, 75),
                max(2, self.comet_radius // 3),
                cv2.LINE_AA,
            )
        center = tuple(np.rint(self.comet_position).astype(int))
        cv2.circle(glow, center, self.comet_radius * 2, (255, 155, 82), -1, cv2.LINE_AA)
        cv2.circle(frame, center, self.comet_radius, (255, 182, 96), -1, cv2.LINE_AA)
        cv2.circle(
            frame,
            center,
            max(3, self.comet_radius // 3),
            (255, 252, 242),
            -1,
            cv2.LINE_AA,
        )

    def _draw_core(
        self,
        frame: np.ndarray,
        glow: np.ndarray,
        timestamp: float,
    ) -> None:
        center = tuple(np.rint(self.center).astype(int))
        radius = int(self.core_radius)
        cv2.circle(glow, center, radius * 2, (90, 55, 235), -1, cv2.LINE_AA)
        cv2.circle(frame, center, radius, (40, 24, 78), -1, cv2.LINE_AA)
        cv2.circle(frame, center, radius, (175, 85, 250), 3, cv2.LINE_AA)
        angle = timestamp * 55
        cv2.ellipse(
            frame,
            center,
            (int(radius * 1.45), int(radius * 0.52)),
            angle,
            0,
            360,
            (230, 150, 255),
            2,
            cv2.LINE_AA,
        )

    def _draw_reset(self, frame: np.ndarray, glow: np.ndarray) -> None:
        center = self.reset_center
        radius = max(22, int(self.scale * 0.027))
        color = (198, 202, 214)
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

    def _draw_game_over(self, frame: np.ndarray) -> None:
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (self.play_left, int(self.height * 0.43)),
            (self.play_right, int(self.height * 0.69)),
            (15, 12, 22),
            -1,
        )
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0.0, frame)
        self._center_text(
            frame,
            "ORBIT ENDED",
            int(self.height * 0.535),
            1.25,
            (244, 239, 248),
            3,
        )
        self._center_text(
            frame,
            f"FINAL SCORE  {self.score:04d}",
            int(self.height * 0.615),
            0.68,
            (115, 220, 255),
            2,
        )

    @staticmethod
    def _center_text(
        image: np.ndarray,
        text: str,
        baseline_y: int,
        scale: float,
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        size, _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_DUPLEX,
            scale,
            thickness,
        )
        cv2.putText(
            image,
            text,
            (image.shape[1] // 2 - size[0] // 2, baseline_y),
            cv2.FONT_HERSHEY_DUPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
