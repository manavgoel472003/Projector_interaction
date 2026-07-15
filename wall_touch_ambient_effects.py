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
        self.render_scale = float(np.clip(min(width, height) / 900.0, 0.75, 1.4))
        self.background = gradient_background(width, height, (20, 10, 7), (34, 12, 29))

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
            size = 0.65 + size_roll / 0.55 * 0.55
        elif size_roll < 0.88:
            size = 1.35 + (size_roll - 0.55) / 0.33 * 0.80
        else:
            size = 2.55 + (size_roll - 0.88) / 0.12 * 1.65
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
        connection_glow = np.zeros_like(image)
        connection_core = np.zeros_like(image)
        for index, star in enumerate(self.stars):
            distances = np.linalg.norm(points[index + 1 :] - points[index], axis=1)
            for offset in np.where(distances < connection_distance)[0]:
                other = index + 1 + int(offset)
                alpha = (1 - distances[offset] / connection_distance) * 0.62
                color = (
                    np.asarray(star["color"], dtype=np.float32)
                    + np.asarray(self.stars[other]["color"], dtype=np.float32)
                ) * 0.5
                start = tuple(np.rint(points[index]).astype(int))
                end = tuple(np.rint(points[other]).astype(int))
                cv2.line(
                    connection_glow,
                    start,
                    end,
                    bgr(color * 0.72 + 255 * 0.28, alpha * 0.72),
                    max(5, int(8 * self.render_scale)),
                    cv2.LINE_AA,
                )
                bright_core = color * 0.52 + 255 * 0.48
                cv2.line(
                    connection_core,
                    start,
                    end,
                    bgr(bright_core, min(1.0, alpha * 1.7)),
                    max(1, int(1.6 * self.render_scale)),
                    cv2.LINE_AA,
                )
        connection_bloom = cv2.GaussianBlur(
            connection_glow,
            (0, 0),
            sigmaX=max(2.0, 4.5 * self.render_scale),
        )
        image = cv2.add(image, connection_bloom)
        image = cv2.add(image, connection_glow)
        image = cv2.add(image, connection_core)
        star_glow = np.zeros_like(image)
        for star in self.stars:
            point = tuple(np.rint(np.asarray(star["point"])).astype(int))
            twinkle = 0.65 + 0.35 * np.sin(self.phase + float(star["phase"])) ** 2
            size = float(star["size"])
            color = np.asarray(star["color"])
            cv2.circle(
                star_glow,
                point,
                max(3, int(10 * self.render_scale * twinkle * size)),
                bgr(color * 0.68 + 255 * 0.32, 0.62),
                -1,
                cv2.LINE_AA,
            )
        star_bloom = cv2.GaussianBlur(
            star_glow,
            (0, 0),
            sigmaX=max(2.0, 5.5 * self.render_scale),
        )
        image = cv2.add(image, star_bloom)
        image = cv2.addWeighted(image, 1.0, star_glow, 0.50, 0)
        for star in self.stars:
            point = tuple(np.rint(np.asarray(star["point"])).astype(int))
            twinkle = 0.65 + 0.35 * np.sin(self.phase + float(star["phase"])) ** 2
            size = float(star["size"])
            color = np.asarray(star["color"])
            core_radius = max(2, int(3.4 * self.render_scale * twinkle * size))
            cv2.circle(
                image,
                point,
                core_radius,
                bgr(color * 0.38 + 255 * 0.62, 1.0),
                -1,
                cv2.LINE_AA,
            )
            if size >= 1.35:
                ray = max(5, int(7 * self.render_scale * size * twinkle))
                ray_color = bgr(color * 0.25 + 255 * 0.75, 0.88)
                cv2.line(image, (point[0] - ray, point[1]), (point[0] + ray, point[1]), ray_color, 1, cv2.LINE_AA)
                cv2.line(image, (point[0], point[1] - ray), (point[0], point[1] + ray), ray_color, 1, cv2.LINE_AA)
            cv2.circle(image, point, max(1, core_radius // 3), (255, 255, 255), -1, cv2.LINE_AA)
        return image


class MagneticSand:
    def __init__(self, width: int, height: int, count: int = 2800) -> None:
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
        self.render_scale = float(np.clip(min(width, height) / 900.0, 0.8, 1.5))
        self.background = gradient_background(width, height, (24, 27, 30), (9, 12, 20))

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
        fine = np.zeros_like(image)
        large = np.zeros_like(image)
        cores = np.zeros_like(image)
        points = np.rint(self.positions).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, self.width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, self.height - 1)
        indices = np.arange(len(points))
        large_mask = indices % 7 == 0
        gold_mask = indices % 5 == 0
        cool_mask = ~gold_mask & ~large_mask
        fine[points[cool_mask, 1], points[cool_mask, 0]] = (255, 230, 185)
        fine[points[gold_mask, 1], points[gold_mask, 0]] = (105, 215, 255)
        large[points[large_mask, 1], points[large_mask, 0]] = (220, 245, 255)
        cores[points[:, 1], points[:, 0]] = (245, 250, 255)

        fine_radius = max(1, int(round(1.5 * self.render_scale)))
        large_radius = max(3, int(round(3.2 * self.render_scale)))
        fine_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (fine_radius * 2 + 1, fine_radius * 2 + 1)
        )
        large_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (large_radius * 2 + 1, large_radius * 2 + 1)
        )
        fine_body = cv2.dilate(fine, fine_kernel)
        large_body = cv2.dilate(large, large_kernel)
        combined = cv2.add(fine_body, large_body)
        glow = cv2.GaussianBlur(
            combined,
            (0, 0),
            sigmaX=max(2.0, 3.6 * self.render_scale),
        )
        image = cv2.addWeighted(image, 1.0, glow, 0.82, 0)
        image = cv2.add(image, fine_body)
        image = cv2.add(image, large_body)
        image = cv2.addWeighted(image, 1.0, cores, 0.94, 0)
        return image
