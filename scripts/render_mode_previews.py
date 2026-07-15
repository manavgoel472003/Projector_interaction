#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wall_touch_ambient_effects import ConstellationField, MagneticSand
from wall_touch_connect_four import PrismConnectFour
from wall_touch_effects import PulseGrid, WatercolorPool
from wall_touch_games import TicTacToe
from wall_touch_orbit_keeper import OrbitKeeper
from wall_touch_paint import PaintBrush, make_base_canvas, paint_color


WIDTH = 960
HEIGHT = 600
OUTPUT_DIR = ROOT / "docs" / "images" / "modes"


def save(name: str, frame: np.ndarray) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.png"
    if not cv2.imwrite(
        str(path),
        frame,
        [cv2.IMWRITE_PNG_COMPRESSION, 8],
    ):
        raise RuntimeError(f"Could not write {path}")


def render_paint() -> np.ndarray:
    canvas = make_base_canvas(WIDTH, HEIGHT)
    brush = PaintBrush(radius=24, alpha=0.50)
    x_values = np.linspace(90, WIDTH - 90, 150)
    for band, phase in enumerate((0.0, 1.7, 3.4)):
        y_values = HEIGHT * (0.30 + band * 0.20) + np.sin(
            x_values / WIDTH * np.pi * 3.2 + phase
        ) * 54
        for x, y in zip(x_values, y_values):
            point = np.array([x, y], dtype=np.float32)
            brush.apply(canvas, point, paint_color(point, WIDTH, HEIGHT))
    return canvas


def render_spill() -> np.ndarray:
    pool = WatercolorPool(WIDTH, HEIGHT, simulation_width=320)
    drops = (
        ((180, 210), (75, 105, 245)),
        ((350, 330), (225, 80, 160)),
        ((510, 195), (65, 190, 250)),
        ((675, 350), (235, 155, 55)),
        ((790, 230), (145, 80, 230)),
    )
    for step in range(72):
        if step % 12 == 0:
            point, color = drops[(step // 12) % len(drops)]
            pool.add_drop(
                np.array(point, dtype=np.float32),
                np.array(color, dtype=np.float32),
                output_radius=72,
                amount=0.22,
            )
        pool.step()
    return pool.render()


def render_ripple() -> np.ndarray:
    ripple = WatercolorPool(
        WIDTH,
        HEIGHT,
        simulation_width=360,
        ripple_contrast=1.55,
        water_color=(20, 15, 105),
        reflection_color=(205, 226, 255),
        reflection_gain=2.35,
    )
    points = ((250, 260), (510, 340), (730, 225))
    for step in range(34):
        if step in (0, 7, 15):
            ripple.add_ripple(
                np.array(points[(0, 7, 15).index(step)], dtype=np.float32),
                output_radius=82,
                strength=0.11,
            )
        ripple.step()
    return ripple.render()


def render_pulse() -> np.ndarray:
    pulse = PulseGrid(WIDTH, HEIGHT)
    entries = (
        ((210, 190), (255, 185, 70)),
        ((480, 315), (100, 215, 255)),
        ((745, 405), (225, 90, 220)),
    )
    for index, (point, color) in enumerate(entries):
        pulse.add_pulse(
            np.array(point, dtype=np.float32),
            np.array(color, dtype=np.float32),
            index * 0.2,
        )
    for _ in range(11):
        pulse.step(1.0 / 30.0)
    return pulse.render()


def render_constellation() -> np.ndarray:
    field = ConstellationField(WIDTH, HEIGHT)
    rng = np.random.default_rng(91)
    centers = np.array(
        [[230, 220], [430, 350], [650, 205], [745, 395]],
        dtype=np.float32,
    )
    timestamp = 0.0
    for index in range(34):
        center = centers[index % len(centers)]
        point = center + rng.normal(0, (72, 58), 2)
        point = np.clip(point, (50, 60), (WIDTH - 50, HEIGHT - 60))
        color = np.array(
            (255, 175 + index % 70, 105 + (index * 17) % 130),
            dtype=np.float32,
        )
        field.add(point.astype(np.float32), color, timestamp)
        timestamp += 0.13
    field.step(0.15)
    return field.render()


def render_sand() -> np.ndarray:
    sand = MagneticSand(WIDTH, HEIGHT, count=3600)
    for point in ((300, 250), (640, 350), (480, 300)):
        sand.attract(np.array(point, dtype=np.float32))
        for _ in range(16):
            sand.step(1.0 / 30.0)
    return sand.render()


def render_tic_tac_toe() -> np.ndarray:
    game = TicTacToe(WIDTH, HEIGHT)
    for index, cell in enumerate(((0, 0), (1, 0), (0, 1), (1, 1), (0, 2))):
        game.place(np.array(game._cell_center(*cell), dtype=np.float32), index * 0.2)
    return game.render(1.4)


def render_connect_four() -> np.ndarray:
    game = PrismConnectFour(WIDTH, HEIGHT)
    for index, column in enumerate((0, 0, 1, 1, 2, 2, 3)):
        game.drop(column, index * 0.15)
    return game.render(1.4)


def render_orbit_keeper() -> np.ndarray:
    game = OrbitKeeper(WIDTH, HEIGHT, seed=8)
    game.add_well(game.comet_position + np.array([0.0, -105.0]))
    for _ in range(22):
        game.step(1.0 / 15.0)
    return game.render(1.5)


def main() -> None:
    previews = {
        "paint": render_paint(),
        "spill": render_spill(),
        "ripple": render_ripple(),
        "pulse": render_pulse(),
        "constellation": render_constellation(),
        "sand": render_sand(),
        "tic-tac-toe": render_tic_tac_toe(),
        "connect-four": render_connect_four(),
        "orbit-keeper": render_orbit_keeper(),
    }
    for name, frame in previews.items():
        save(name, frame)
    print(f"Rendered {len(previews)} previews to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
