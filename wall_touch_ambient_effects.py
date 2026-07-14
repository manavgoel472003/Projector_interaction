from __future__ import annotations

import cv2
import numpy as np


def gradient_background(
    width: int,
    height: int,
    top: tuple[int, int, int],
    bottom: tuple[int, int, int],
) -> np.ndarray:
    blend = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    top_color = np.asarray(top, dtype=np.float32)
    bottom_color = np.asarray(bottom, dtype=np.float32)
    image = top_color + blend * (bottom_color - top_color)
    return np.broadcast_to(image, (height, width, 3)).astype(np.uint8).copy()


def bgr(color: np.ndarray, scale: float = 1.0) -> tuple[int, int, int]:
    return tuple(int(value) for value in np.clip(np.asarray(color) * scale, 0, 255))


class ConstellationField:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(27)
        self.stars: list[dict[str, np.ndarray | float]] = []
        self.phase = 0.0
        self.last_emit = -1e9
        self.last_point: np.ndarray | None = None
        self.background = gradient_background(width, height, (30, 17, 7), (48, 20, 30))

    @property
    def count(self) -> int:
        return len(self.stars)

    def clear(self) -> None:
        self.stars.clear()
        self.last_point = None
        self.last_emit = -1e9

    def add(self, point: np.ndarray, color: np.ndarray, timestamp: float) -> bool:
        point = np.asarray(point, dtype=np.float32)
        moved = self.last_point is None or np.linalg.norm(point - self.last_point) > 42
        if timestamp - self.last_emit < 0.12 and not moved:
            return False
        size_roll = float(self.rng.random())
        if size_roll < 0.55:
            size = 0.34 + size_roll / 0.55 * 0.42
        elif size_roll < 0.88:
            size = 0.84 + (size_roll - 0.55) / 0.33 * 0.54
        else:
            size = 1.65 + (size_roll - 0.88) / 0.12 * 0.85
        self.stars.append(
            {
                "point": point.copy(),
                "color": np.asarray(color, dtype=np.float32).copy(),
                "life": 1.0,
                "phase": float(self.rng.uniform(0, np.pi * 2)),
                "size": size,
            }
        )
        self.stars = self.stars[-90:]
        self.last_point = point.copy()
        self.last_emit = timestamp
        return True

    def step(self, delta_seconds: float) -> None:
        delta = min(max(delta_seconds, 0.001), 0.1)
        self.phase += delta * 2.0
        for star in self.stars:
            star["life"] = float(star["life"]) - delta / 11.0
        self.stars = [star for star in self.stars if float(star["life"]) > 0]

    def render(self) -> np.ndarray:
        image = self.background.copy()
        if not self.stars:
            return image
        points = np.array([star["point"] for star in self.stars], dtype=np.float32)
        connection_distance = min(self.width, self.height) * 0.18
        connection_light = np.zeros_like(image)
        for index, star in enumerate(self.stars):
            distances = np.linalg.norm(points[index + 1 :] - points[index], axis=1)
            for offset in np.where(distances < connection_distance)[0]:
                other = index + 1 + int(offset)
                alpha = (1 - distances[offset] / connection_distance) * 0.36
                color = (
                    np.asarray(star["color"], dtype=np.float32)
                    + np.asarray(self.stars[other]["color"], dtype=np.float32)
                ) * 0.5
                start = tuple(np.rint(points[index]).astype(int))
                end = tuple(np.rint(points[other]).astype(int))
                cv2.line(
                    connection_light,
                    start,
                    end,
                    bgr(color, alpha * 0.52),
                    5,
                    cv2.LINE_AA,
                )
                bright_core = color * 0.72 + 255 * 0.28
                cv2.line(
                    connection_light,
                    start,
                    end,
                    bgr(bright_core, min(1.0, alpha * 1.35)),
                    1,
                    cv2.LINE_AA,
                )
        image = cv2.add(image, connection_light)
        for star in self.stars:
            point = tuple(np.rint(np.asarray(star["point"])).astype(int))
            twinkle = 0.65 + 0.35 * np.sin(self.phase + float(star["phase"])) ** 2
            size = float(star["size"])
            color = np.asarray(star["color"])
            cv2.circle(
                image,
                point,
                max(2, int(7 * twinkle * size)),
                bgr(color, 0.20),
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(
                image,
                point,
                max(1, int(2.5 * twinkle * size)),
                bgr(color, 0.92),
                -1,
                cv2.LINE_AA,
            )
        return image


class MagneticSand:
    def __init__(self, width: int, height: int, count: int = 1400) -> None:
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(31)
        self.positions = np.column_stack(
            (self.rng.uniform(0, width, count), self.rng.uniform(0, height, count))
        ).astype(np.float32)
        self.velocities = np.zeros_like(self.positions)
        self.target: np.ndarray | None = None
        self.target_age = 0.0
        self.phase = 0.0
        self.background = gradient_background(width, height, (35, 39, 38), (22, 27, 31))

    def clear(self) -> None:
        self.positions[:, 0] = self.rng.uniform(0, self.width, len(self.positions))
        self.positions[:, 1] = self.rng.uniform(0, self.height, len(self.positions))
        self.velocities.fill(0)
        self.target = None

    def attract(self, point: np.ndarray) -> None:
        self.target = np.asarray(point, dtype=np.float32).copy()
        self.target_age = 0.28

    def step(self, delta_seconds: float) -> None:
        delta = min(max(delta_seconds, 0.001), 0.08)
        self.phase += delta
        self.target_age = max(0.0, self.target_age - delta)
        if self.target is not None and self.target_age > 0:
            offset = self.target[None, :] - self.positions
            distance = np.linalg.norm(offset, axis=1) + 12
            direction = offset / distance[:, None]
            tangent = np.column_stack((-direction[:, 1], direction[:, 0]))
            self.velocities += direction * (1100 / distance)[:, None] * delta
            self.velocities += tangent * (460 / distance)[:, None] * delta
        else:
            self.velocities[:, 0] += np.sin(self.positions[:, 1] * 0.012 + self.phase) * delta * 1.2
            self.velocities[:, 1] += np.cos(self.positions[:, 0] * 0.010 - self.phase) * delta * 1.2
        self.velocities *= 0.965
        self.positions += self.velocities * delta * 60
        self.positions[:, 0] %= self.width
        self.positions[:, 1] %= self.height

    def render(self) -> np.ndarray:
        image = self.background.copy()
        layer = np.zeros_like(image)
        points = np.rint(self.positions).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, self.width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, self.height - 1)
        split = np.arange(len(points)) % 5 == 0
        layer[points[~split, 1], points[~split, 0]] = (120, 182, 214)
        layer[points[split, 1], points[split, 0]] = (205, 166, 98)
        glow = cv2.dilate(layer, np.ones((3, 3), dtype=np.uint8))
        return cv2.add(image, cv2.addWeighted(glow, 0.38, layer, 0.92, 0))
