from __future__ import annotations

import cv2
import numpy as np


class WatercolorPool:
    """Small fluid-like pigment simulation rendered at projector resolution."""

    def __init__(
        self,
        output_width: int,
        output_height: int,
        simulation_width: int = 360,
        ripple_contrast: float = 1.0,
        water_color: tuple[int, int, int] = (250, 246, 228),
        reflection_color: tuple[int, int, int] = (255, 255, 255),
        reflection_gain: float = 1.0,
    ) -> None:
        self.output_width = output_width
        self.output_height = output_height
        self.width = min(simulation_width, output_width)
        self.height = max(96, int(round(self.width * output_height / output_width)))
        self.ripple_contrast = ripple_contrast
        self.water_color = np.asarray(water_color, dtype=np.float32)
        self.reflection_color = np.asarray(reflection_color, dtype=np.float32) / 255.0
        self.reflection_gain = reflection_gain

        self.pigment = np.zeros((self.height, self.width, 3), dtype=np.float32)
        self.wave = np.zeros((self.height, self.width), dtype=np.float32)
        self.wave_velocity = np.zeros_like(self.wave)
        self.ripple_events: list[dict[str, float]] = []
        self.phase = 0.0

        self.grid_x, self.grid_y = np.meshgrid(
            np.arange(self.width, dtype=np.float32),
            np.arange(self.height, dtype=np.float32),
        )
        self.mask = self._make_pool_mask()
        self.caustic_scale = 2.4 if ripple_contrast <= 1.5 else 3.2

    def _make_pool_mask(self) -> np.ndarray:
        return np.ones((self.height, self.width), dtype=np.float32)

    @property
    def pigment_mass(self) -> float:
        return float(self.pigment.sum())

    def clear(self) -> None:
        self.pigment.fill(0)
        self.wave.fill(0)
        self.wave_velocity.fill(0)
        self.ripple_events.clear()

    def _simulation_point(self, output_point: np.ndarray) -> tuple[int, int]:
        x = int(np.clip(output_point[0] / self.output_width * self.width, 0, self.width - 1))
        y = int(np.clip(output_point[1] / self.output_height * self.height, 0, self.height - 1))
        return x, y

    def add_drop(
        self,
        output_point: np.ndarray,
        bgr_color: np.ndarray,
        output_radius: int = 52,
        amount: float = 0.14,
    ) -> bool:
        x, y = self._simulation_point(np.asarray(output_point, dtype=np.float32))
        if self.mask[y, x] < 0.5:
            return False

        radius = max(3, int(round(output_radius / self.output_width * self.width)))
        spread = radius * 2
        x0, x1 = max(0, x - spread), min(self.width, x + spread + 1)
        y0, y1 = max(0, y - spread), min(self.height, y + spread + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        falloff = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * radius * radius)).astype(np.float32)
        falloff *= self.mask[y0:y1, x0:x1]

        reflectance = 0.08 + 0.92 * np.clip(np.asarray(bgr_color, dtype=np.float32) / 255.0, 0, 1)
        absorbance = -np.log(reflectance)
        self.pigment[y0:y1, x0:x1] += falloff[..., None] * absorbance * amount
        self.wave_velocity[y0:y1, x0:x1] += falloff * 0.025
        return True

    def add_ripple(
        self,
        output_point: np.ndarray,
        output_radius: int = 64,
        strength: float = 0.075,
    ) -> bool:
        x, y = self._simulation_point(np.asarray(output_point, dtype=np.float32))
        if self.mask[y, x] < 0.5:
            return False

        radius = max(2.0, output_radius / self.output_width * self.width * 0.22)
        self.ripple_events.append(
            {
                "x": float(x),
                "y": float(y),
                "age": 0.0,
                "radius": float(radius),
                "strength": float(np.clip(strength / 0.12, 0.35, 1.4)),
            }
        )
        self.ripple_events = self.ripple_events[-6:]
        return True

    def step(self) -> None:
        self.phase += 0.075
        for event in self.ripple_events:
            event["age"] += 1.0
        self.ripple_events = [event for event in self.ripple_events if event["age"] < 64]
        flow_x = (
            0.34 * np.sin(self.grid_y * 0.055 + self.phase)
            + 0.16 * np.cos(self.grid_x * 0.035 - self.phase * 0.8)
        )
        flow_y = (
            0.26 * np.cos(self.grid_x * 0.045 + self.phase * 0.7)
            + 0.12 * np.sin(self.grid_y * 0.07 - self.phase)
        )
        self.pigment = cv2.remap(
            self.pigment,
            self.grid_x - flow_x.astype(np.float32),
            self.grid_y - flow_y.astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        self.pigment = cv2.GaussianBlur(self.pigment, (0, 0), 0.62)
        self.pigment *= 0.998
        self.pigment *= self.mask[..., None]

        laplacian = (
            np.roll(self.wave, 1, axis=0)
            + np.roll(self.wave, -1, axis=0)
            + np.roll(self.wave, 1, axis=1)
            + np.roll(self.wave, -1, axis=1)
            - 4.0 * self.wave
        )
        self.wave_velocity = (self.wave_velocity + 0.16 * laplacian) * 0.955
        self.wave = (self.wave + self.wave_velocity) * 0.992
        self.wave *= self.mask
        self.wave_velocity *= self.mask

    def render(self) -> np.ndarray:
        display_wave = self.wave.copy()
        warped_x = self.grid_x + np.sin(self.grid_y * 0.055 + self.phase) * 1.25
        warped_y = self.grid_y + np.cos(self.grid_x * 0.047 - self.phase * 0.8) * 1.0
        for event in self.ripple_events:
            age = event["age"]
            radius = event["radius"] + age * 1.22
            offset_x = warped_x - event["x"]
            offset_y = warped_y - event["y"]
            distance = np.hypot(offset_x, offset_y)
            behind = np.maximum(radius - distance, 0)
            front = np.cos((distance - radius) * 0.92) * np.exp(
                -((distance - radius) / 4.8) ** 2
            )
            trailing = (
                (distance < radius)
                * np.sin(behind * 0.78)
                * np.exp(-behind / 6.5)
                * 0.18
            )
            fade = max(0.0, 1.0 - age / 64.0)
            display_wave += (front + trailing) * event["strength"] * fade

        grad_x = cv2.Sobel(display_wave, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(display_wave, cv2.CV_32F, 0, 1, ksize=3)
        refracted = cv2.remap(
            self.pigment,
            self.grid_x - grad_x * 1.6,
            self.grid_y - grad_y * 1.6,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        water = self.water_color.reshape(1, 1, 3) * np.exp(
            -np.clip(refracted, 0, 6) * 1.05
        )
        depth = 0.94 + 0.06 * (
            0.5
            + 0.5
            * np.sin(self.grid_x * 0.021 + self.phase * 0.45)
            * np.cos(self.grid_y * 0.018 - self.phase * 0.32)
        )
        water *= depth[..., None]
        saturated = np.clip(water, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(saturated, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1].astype(np.float32) * 1.22, 0, 255).astype(np.uint8)
        water = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)
        shimmer = np.clip(
            ((grad_x - grad_y) * 4.2 + display_wave * 3.1) * self.ripple_contrast,
            -8 * self.ripple_contrast,
            8 * self.ripple_contrast,
        )
        caustics = (
            np.sin(self.grid_x * 0.105 + self.phase * 1.7)
            * np.cos(self.grid_y * 0.082 - self.phase)
            + 0.45 * np.sin((self.grid_x + self.grid_y) * 0.047 - self.phase * 1.4)
        ) * self.caustic_scale
        lighting = shimmer + caustics
        shadow = np.minimum(lighting, 0)
        reflected = np.maximum(lighting, 0)[..., None] * self.reflection_color
        directional = np.maximum(grad_x * 0.72 - grad_y * 0.48, 0)
        specular = np.clip(directional * directional * 24.0, 0, 52) * self.reflection_gain
        water = np.clip(
            water
            + shadow[..., None]
            + reflected
            + specular[..., None] * self.reflection_color,
            0,
            255,
        ).astype(np.uint8)

        surface = np.full(
            (self.height, self.width, 3), self.water_color, dtype=np.uint8
        )
        inside = self.mask >= 0.5
        surface[inside] = water[inside]
        surface = cv2.GaussianBlur(surface, (0, 0), 0.45)
        output = cv2.resize(
            surface,
            (self.output_width, self.output_height),
            interpolation=cv2.INTER_CUBIC,
        )
        return cv2.GaussianBlur(output, (0, 0), 1.4)


class PulseGrid:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.spacing = max(75, min(width, height) // 12)
        self.points = np.array(
            [
                (x, y)
                for y in range(self.spacing // 2, height, self.spacing)
                for x in range(self.spacing // 2, width, self.spacing)
            ],
            dtype=np.float32,
        )
        self.pulses: list[dict[str, np.ndarray | float]] = []
        self.last_emit_time = -1e9
        self.last_point: np.ndarray | None = None
        self.phase = 0.0
        self.background = self._make_background()

    def _make_background(self) -> np.ndarray:
        vertical = np.linspace(0.0, 1.0, self.height, dtype=np.float32)[:, None, None]
        top = np.array((54, 48, 8), dtype=np.float32)
        bottom = np.array((74, 28, 34), dtype=np.float32)
        image = np.broadcast_to(top + vertical * (bottom - top), (self.height, self.width, 3))
        image = image.astype(np.uint8).copy()
        for x in range(self.spacing // 2, self.width, self.spacing):
            cv2.line(image, (x, 0), (x, self.height), (72, 62, 92), 1)
        for y in range(self.spacing // 2, self.height, self.spacing):
            cv2.line(image, (0, y), (self.width, y), (38, 78, 104), 1)
        return image

    @property
    def count(self) -> int:
        return len(self.pulses)

    def clear(self) -> None:
        self.pulses.clear()
        self.last_emit_time = -1e9
        self.last_point = None

    def add_pulse(
        self,
        point: np.ndarray,
        color: np.ndarray,
        timestamp: float,
    ) -> bool:
        point = np.asarray(point, dtype=np.float32)
        moved = self.last_point is None or np.linalg.norm(point - self.last_point) > 90
        if timestamp - self.last_emit_time < 0.14 and not moved:
            return False
        self.pulses.append(
            {"center": point.copy(), "color": np.asarray(color, dtype=np.float32).copy(), "age": 0.0}
        )
        self.pulses = self.pulses[-14:]
        self.last_emit_time = timestamp
        self.last_point = point.copy()
        return True

    def step(self, delta_seconds: float) -> None:
        self.phase += min(delta_seconds, 0.1) * 3.0
        for pulse in self.pulses:
            pulse["age"] = float(pulse["age"]) + min(delta_seconds, 0.1)
        self.pulses = [pulse for pulse in self.pulses if float(pulse["age"]) < 1.8]

    def render(self) -> np.ndarray:
        image = self.background.copy()

        point_colors = np.repeat(
            np.array([[105.0, 122.0, 178.0]], dtype=np.float32), len(self.points), axis=0
        )
        point_sizes = np.full(len(self.points), 3.0, dtype=np.float32)
        for pulse in self.pulses:
            age = float(pulse["age"])
            center = np.asarray(pulse["center"], dtype=np.float32)
            color = np.asarray(pulse["color"], dtype=np.float32)
            fade = max(0.0, 1.0 - age / 1.8)
            radius = 35 + age * max(self.width, self.height) * 0.42
            ring_color = tuple(
                int(value) for value in np.clip(color * (0.35 + 0.65 * fade), 0, 255)
            )
            center_int = tuple(np.rint(center).astype(int))
            glow_color = tuple(int(value) for value in np.clip(color * fade * 0.24, 0, 255))
            cv2.circle(image, center_int, int(radius), glow_color, 18, cv2.LINE_AA)
            cv2.circle(image, center_int, int(radius), ring_color, 5, cv2.LINE_AA)
            cv2.circle(image, center_int, int(radius + 8), (225, 235, 255), 1, cv2.LINE_AA)

            distance = np.linalg.norm(self.points - center[None, :], axis=1)
            influence = np.exp(-((distance - radius) ** 2) / (2 * (self.spacing * 0.42) ** 2)) * fade
            point_colors += influence[:, None] * color[None, :] * 1.05
            point_sizes += influence * 9.0

        for point, color, size in zip(self.points, point_colors, point_sizes):
            center = tuple(np.rint(point).astype(int))
            glow_radius = int(max(5, size + 5))
            glow_color = tuple(int(value) for value in np.clip(color * 0.30, 0, 255))
            cv2.circle(image, center, glow_radius, glow_color, -1, cv2.LINE_AA)
            cv2.circle(
                image,
                center,
                int(max(2, size)),
                tuple(int(value) for value in np.clip(color, 0, 255)),
                -1,
                cv2.LINE_AA,
            )
            if size > 5:
                cv2.circle(image, center, max(1, int(size * 0.28)), (250, 250, 255), -1, cv2.LINE_AA)
        return image
