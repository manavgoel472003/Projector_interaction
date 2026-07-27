from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque
from pathlib import Path

SYSTEM_FONT_DIR = "/usr/share/fonts/truetype/dejavu"
os.environ["QT_QPA_FONTDIR"] = SYSTEM_FONT_DIR
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import mediapipe as mp
import numpy as np

# OpenCV's Qt-enabled wheel replaces this with a bundled path that no longer exists.
os.environ["QT_QPA_FONTDIR"] = SYSTEM_FONT_DIR

from wall_touch_ambient_effects import ConstellationField, MagneticSand
from wall_touch_connect_four import PrismConnectFour
from wall_touch_core import (
    CORNER_NAMES,
    DepthContactLock,
    DepthContactObservation,
    DepthContactTracker,
    SpatialTouchCalibration,
    DepthTouchProfile,
    DepthTouchGate,
    TouchGate,
    WallDepthModel,
    build_homography,
    camera_detection_roi,
    combine_wall_depth_models,
    depth_target_foreground_metrics,
    fit_wall_depth_model,
    hand_plane_scale,
    index_is_extended,
    point_in_output,
    sample_fingertip_depth,
    projection_near_frame_edge,
    projector_targets,
    transform_points,
    validate_camera_quad,
)
from wall_touch_effects import PulseGrid, WatercolorPool
from wall_touch_games import TicTacToe
from wall_touch_orbit_keeper import OrbitKeeper
from wall_touch_orbbec import OrbbecCamera, orbbec_device_count
from wall_touch_realsense import RealSenseCamera, realsense_device_count


ROOT = Path(__file__).resolve().parent
APP_VERSION = "3.5.1"
DEPTH_TOUCH_MODE = "fingertip-3d-plane-v4"
DEFAULT_CAMERA = "auto"
DEFAULT_MODEL = ROOT / "models/hand_landmarker.task"
DEFAULT_CALIBRATION = ROOT / "wall_touch_calibration.json"
BLOCKED_CAMERA_NAMES = ("usb2.0 fhd uvc webcam", "shinetech")
YUYV_CAMERA_NAMES = ("046d:0825",)
MODE_ORDER = (
    "paint",
    "spill",
    "ripple",
    "pulse",
    "constellation",
    "sand",
    "tic-tac-toe",
    "connect-four",
    "orbit-keeper",
)
MODE_KEYS = {ord(str(index + 1)): mode for index, mode in enumerate(MODE_ORDER)}

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrated projector touch demo using an RGB-D or external RGB camera."
    )
    parser.add_argument(
        "--sensor",
        choices=("auto", "realsense", "orbbec", "rgb"),
        default="auto",
        help="Prefer RealSense/Orbbec depth when available, or force one capture backend.",
    )
    parser.add_argument(
        "--close-bottom",
        action="store_true",
        help=(
            "Preset for a RealSense mounted close to the wall below the projection "
            "and aimed upward."
        ),
    )
    parser.add_argument(
        "--realsense-preset",
        choices=("high-accuracy", "high-density"),
        default="high-accuracy",
        help="RealSense stereo preset (close-bottom forces high-density).",
    )
    parser.add_argument(
        "--camera",
        default=os.environ.get("WALL_TOUCH_CAMERA", DEFAULT_CAMERA),
        help="External V4L2 device path, or 'auto' to discover one (default).",
    )
    parser.add_argument("--camera-width", type=int, help="Requested width; defaults depend on camera format.")
    parser.add_argument("--camera-height", type=int, help="Requested height; defaults depend on camera format.")
    parser.add_argument("--camera-fps", type=int, help="Requested FPS; defaults depend on camera format.")
    parser.add_argument(
        "--camera-format",
        choices=("auto", "mjpg", "yuyv"),
        default="auto",
        help="V4L2 pixel format (default: camera-specific automatic selection).",
    )
    parser.add_argument("--projector-width", type=int, default=1920)
    parser.add_argument("--projector-height", type=int, default=1200)
    parser.add_argument("--projector-x", type=int, default=0)
    parser.add_argument("--projector-y", type=int, default=0)
    parser.add_argument("--debug-x", type=int, default=1980)
    parser.add_argument("--debug-y", type=int, default=60)
    parser.add_argument("--windowed", action="store_true", help="Do not fullscreen the projector output.")
    parser.add_argument("--fresh", action="store_true", help="Ignore saved geometry and touch calibration.")
    parser.add_argument(
        "--recalibrate-depth",
        action="store_true",
        help="Keep saved projection points but relearn the RGB-D wall and hand contact.",
    )
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--touch-samples", type=int, default=24)
    parser.add_argument("--touch-scale-min", type=float, default=0.50)
    parser.add_argument("--touch-scale-max", type=float, default=1.60)
    parser.add_argument(
        "--touch-dwell-ms",
        type=int,
        default=0,
        help="Contact hold time before activation (default: immediate).",
    )
    parser.add_argument("--touch-min-gap-mm", type=float, default=-30.0)
    parser.add_argument("--touch-max-gap-mm", type=float, default=45.0)
    parser.add_argument(
        "--touch-plane-tolerance-mm",
        type=float,
        default=18.0,
        help=(
            "How far in front of the calibrated 3D plane activates touch "
            "(default: 18 mm). Lower is closer/stricter."
        ),
    )
    parser.add_argument(
        "--fingertip-depth-radius",
        type=int,
        default=7,
        help="Radius of the aligned depth patch sampled around the fingertip.",
    )
    parser.add_argument("--depth-calibration-frames", type=int, default=45)
    parser.add_argument("--depth-change-min-mm", type=float, default=15.0)
    parser.add_argument("--depth-component-min-area", type=int, default=80)
    parser.add_argument("--depth-noise-multiplier", type=float, default=0.75)
    parser.add_argument("--depth-touch-samples", type=int, default=10)
    parser.add_argument(
        "--depth-calibration-max-gap-mm",
        type=float,
        default=60.0,
        help="Largest wall gap accepted while learning a guided hand press.",
    )
    parser.add_argument("--require-index-extension", action="store_true")
    parser.add_argument(
        "--legacy-depth-blob",
        action="store_true",
        help=(
            "Use the old depth blob tracker and guided-touch calibration instead of "
            "MediaPipe fingertip detection fused with the wall-depth plane."
        ),
    )
    parser.add_argument("--brush-radius", type=int, default=46)
    parser.add_argument("--paint-alpha", type=float, default=0.46)
    parser.add_argument("--mode", choices=MODE_ORDER, default="spill")
    parser.add_argument("--detection-confidence", type=float, default=0.40)
    parser.add_argument(
        "--detection-size",
        type=int,
        default=384,
        help=(
            "Target long-side (px) the detection crop is resized to before MediaPipe. "
            "A tight/far crop is upscaled so the hand is big enough to detect; a large "
            "crop is downscaled to cut lag. Lower = faster, higher = longer range."
        ),
    )
    parser.add_argument(
        "--depth-roi",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the near-wall depth foreground to crop the color image tightly around "
            "the hand before MediaPipe (depth sees the hand regardless of its color pixel "
            "size, so this greatly extends range when the camera is far). Depth mode only."
        ),
    )
    parser.add_argument(
        "--enhance-contrast",
        action="store_true",
        help=(
            "CLAHE-normalize the detection ROI so a brightly projected / light-colored "
            "wall does not wash the hand out for MediaPipe."
        ),
    )
    parser.add_argument(
        "--dual-detect",
        action="store_true",
        help=(
            "When the normal detector finds no hand, retry on the inverted (negative) "
            "ROI. Helps on light/washed-out walls where the hand only stands out in the "
            "negative. Runs only on missed frames, so it adds no lag while a hand is tracked."
        ),
    )
    return parser.parse_args()


def apply_close_bottom_preset(args: argparse.Namespace) -> None:
    if not args.close_bottom:
        return
    if args.sensor == "auto":
        args.sensor = "realsense"
    elif args.sensor != "realsense":
        raise ValueError("--close-bottom requires --sensor realsense")
    args.realsense_preset = "high-density"
    args.camera_width = args.camera_width or 640
    args.camera_height = args.camera_height or 480
    args.camera_fps = args.camera_fps or 30
    args.depth_calibration_frames = max(args.depth_calibration_frames, 60)
    args.depth_touch_samples = max(args.depth_touch_samples, 14)
    args.touch_plane_tolerance_mm = max(args.touch_plane_tolerance_mm, 12.0)
    args.touch_dwell_ms = 0
    args.fingertip_depth_radius = max(args.fingertip_depth_radius, 12)
    # Keep the detector well-fed at close range (was --detection-scale >= 0.85 before
    # that flag became the target-size --detection-size).
    args.detection_size = max(args.detection_size, 448)
    args.require_index_extension = False
    args.enhance_contrast = True
    args.dual_detect = True


def camera_name(device_path: Path) -> str:
    resolved = device_path.resolve()
    if not resolved.name.startswith("video"):
        return "unknown"
    name_file = Path("/sys/class/video4linux") / resolved.name / "name"
    return name_file.read_text().strip() if name_file.exists() else "unknown"


def is_primary_video_node(device_path: Path) -> bool:
    resolved = device_path.resolve()
    if not resolved.name.startswith("video"):
        return False
    index_file = Path("/sys/class/video4linux") / resolved.name / "index"
    try:
        return index_file.read_text().strip() == "0"
    except OSError:
        return True


def discover_external_cameras() -> list[tuple[str, str]]:
    candidates: list[Path] = []
    for directory in (Path("/dev/v4l/by-id"), Path("/dev/v4l/by-path")):
        if directory.exists():
            candidates.extend(sorted(directory.glob("*-video-index0")))
    candidates.extend(
        sorted(
            Path("/dev").glob("video*"),
            key=lambda path: int(path.name[5:]) if path.name[5:].isdigit() else 10_000,
        )
    )

    cameras: list[tuple[str, str]] = []
    seen_devices: set[Path] = set()
    for path in candidates:
        if not path.exists() or not is_primary_video_node(path):
            continue
        resolved = path.resolve()
        if resolved in seen_devices:
            continue
        seen_devices.add(resolved)
        name = camera_name(path)
        if any(token in name.lower() for token in BLOCKED_CAMERA_NAMES):
            continue
        cameras.append((str(path), name))
    return cameras


def validate_camera(requested: str) -> tuple[str, str]:
    if requested.strip().lower() == "auto":
        cameras = discover_external_cameras()
        if not cameras:
            raise RuntimeError(
                "No external camera was found. Connect a V4L2 camera or pass "
                "--camera /dev/videoN explicitly."
            )
        return cameras[0]
    if requested.isdigit():
        raise RuntimeError("Camera indexes are disabled. Pass an explicit /dev/video* or /dev/v4l/by-id/* path.")
    path = Path(requested)
    if not path.exists():
        raise RuntimeError(
            f"External camera path does not exist: {requested}. "
            "Use --camera auto to discover the currently connected camera."
        )
    name = camera_name(path)
    if any(token in name.lower() for token in BLOCKED_CAMERA_NAMES):
        raise RuntimeError(f"Refusing PC webcam device {requested!r}: {name}")
    return str(path), name


def camera_stream_profile(
    identity: str,
    requested_format: str,
    width: int | None,
    height: int | None,
    fps: int | None,
) -> tuple[str, int, int, int]:
    use_yuyv = requested_format == "yuyv" or (
        requested_format == "auto"
        and any(token in identity.lower() for token in YUYV_CAMERA_NAMES)
    )
    if use_yuyv:
        return "YUYV", width or 640, height or 480, fps or 30
    return "MJPG", width or 1280, height or 720, fps or 30


def open_camera(
    path: str,
    width: int,
    height: int,
    fps: int,
    pixel_format: str,
) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open external camera {path}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*pixel_format))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def read_camera_frame(cap: cv2.VideoCapture, timeout_seconds: float = 2.5) -> np.ndarray:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame
        time.sleep(0.04)
    raise RuntimeError("External camera did not return a frame before the timeout")


def capture_fourcc(cap: cv2.VideoCapture) -> str:
    value = int(cap.get(cv2.CAP_PROP_FOURCC))
    return "".join(chr((value >> (8 * index)) & 0xFF) for index in range(4)).strip("\x00")


def make_base_canvas(width: int, height: int) -> np.ndarray:
    rng = np.random.default_rng(21)
    vertical = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    top = np.array((248, 246, 235), dtype=np.float32)
    bottom = np.array((242, 232, 249), dtype=np.float32)
    canvas = top + vertical * (bottom - top)
    canvas = np.broadcast_to(canvas, (height, width, 3)).copy()
    grain = rng.normal(0, 2.2, (height, width, 1)).astype(np.float32)
    canvas = np.clip(canvas + grain, 0, 255).astype(np.uint8)

    spacing = max(90, min(width, height) // 9)
    for index, x in enumerate(range(spacing, width, spacing)):
        color = (214, 198, 236) if index % 2 else (225, 218, 179)
        cv2.line(canvas, (x, 0), (x, height), color, 1, cv2.LINE_AA)
    for index, y in enumerate(range(spacing, height, spacing)):
        color = (225, 218, 179) if index % 2 else (214, 198, 236)
        cv2.line(canvas, (0, y), (width, y), color, 1, cv2.LINE_AA)

    x_values = np.arange(width, dtype=np.int32)
    for offset, color in ((0.28, (205, 225, 244)), (0.72, (235, 220, 194))):
        y_values = height * offset + np.sin(x_values / max(width, 1) * np.pi * 4) * height * 0.018
        curve = np.column_stack((x_values, y_values.astype(np.int32)))
        cv2.polylines(canvas, [curve], False, color, 2, cv2.LINE_AA)
    return canvas


class PaintBrush:
    def __init__(self, radius: int, alpha: float) -> None:
        self.radius = radius
        axis = np.arange(-radius, radius + 1, dtype=np.float32)
        xx, yy = np.meshgrid(axis, axis)
        sigma = max(radius * 0.52, 1.0)
        self.weight = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
        self.weight[xx * xx + yy * yy > radius * radius] = 0
        rng = np.random.default_rng(7)
        texture = cv2.GaussianBlur(
            rng.uniform(0.72, 1.08, self.weight.shape).astype(np.float32), (0, 0), 1.1
        )
        self.weight = (self.weight * texture * alpha)[..., None]
        self.rng = rng
        self.stroke_count = 0

    def apply(self, canvas: np.ndarray, point: np.ndarray, color: np.ndarray) -> None:
        x, y = np.rint(point).astype(int)
        height, width = canvas.shape[:2]
        x0, x1 = max(0, x - self.radius), min(width, x + self.radius + 1)
        y0, y1 = max(0, y - self.radius), min(height, y + self.radius + 1)
        if x0 >= x1 or y0 >= y1:
            return
        wx0, wy0 = x0 - (x - self.radius), y0 - (y - self.radius)
        weights = self.weight[wy0:wy0 + (y1 - y0), wx0:wx0 + (x1 - x0)]
        region = canvas[y0:y1, x0:x1].astype(np.float32)
        mixed = region * (1.0 - weights) + color.reshape(1, 1, 3) * weights
        canvas[y0:y1, x0:x1] = np.clip(mixed, 0, 255).astype(np.uint8)
        self.stroke_count += 1
        if self.stroke_count % 6 == 0:
            for _ in range(3):
                angle = self.rng.uniform(0, np.pi * 2)
                distance = self.rng.uniform(self.radius * 0.75, self.radius * 1.45)
                center = (
                    int(round(x + np.cos(angle) * distance)),
                    int(round(y + np.sin(angle) * distance)),
                )
                cv2.circle(
                    canvas,
                    center,
                    int(self.rng.integers(2, max(3, self.radius // 8))),
                    tuple(int(value) for value in color),
                    -1,
                    cv2.LINE_AA,
                )


def paint_color(point: np.ndarray, width: int, height: int) -> np.ndarray:
    palette = np.array(
        [
            (143, 93, 255),
            (76, 138, 255),
            (102, 209, 255),
            (160, 214, 6),
            (255, 194, 0),
            (238, 97, 67),
            (229, 93, 155),
            (181, 91, 241),
        ],
        dtype=np.float32,
    )
    position = np.clip(point[0] / max(width - 1, 1), 0, 1) * (len(palette) - 1)
    left = int(np.floor(position))
    right = min(left + 1, len(palette) - 1)
    color = palette[left] * (1.0 - (position - left)) + palette[right] * (position - left)
    brightness = 0.88 + 0.12 * (1.0 - np.clip(point[1] / max(height - 1, 1), 0, 1))
    return np.clip(color * brightness, 0, 255).astype(np.float32)


def draw_outlined_text(frame: np.ndarray, text: str, origin: tuple[int, int], scale: float = 0.65) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def draw_projector_targets(image: np.ndarray, targets: np.ndarray, selected: int) -> None:
    image[:] = (20, 20, 20)
    for index, target in enumerate(targets):
        center = tuple(np.rint(target).astype(int))
        active = index == selected
        color = (40, 220, 255) if active else (255, 255, 255)
        cv2.circle(image, center, 34 if active else 27, color, 5, cv2.LINE_AA)
        cv2.line(image, (center[0] - 48, center[1]), (center[0] + 48, center[1]), color, 3)
        cv2.line(image, (center[0], center[1] - 48), (center[0], center[1] + 48), color, 3)
        cv2.putText(image, str(index + 1), (center[0] + 42, center[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)


def draw_touch_target(image: np.ndarray, center: np.ndarray, progress: int, total: int) -> None:
    x, y = np.rint(center).astype(int)
    color = (40, 190, 255)
    cv2.circle(image, (x, y), 58, color, 6, cv2.LINE_AA)
    cv2.circle(image, (x, y), 8, color, -1, cv2.LINE_AA)
    cv2.line(image, (x - 80, y), (x + 80, y), color, 3)
    cv2.line(image, (x, y - 80), (x, y + 80), color, 3)
    if total:
        cv2.circle(image, (x, y), 74, (155, 160, 160), 8, cv2.LINE_AA)
        end_angle = int(360 * min(progress / total, 1.0))
        cv2.ellipse(image, (x, y), (74, 74), -90, 0, end_angle, (90, 245, 110), 8, cv2.LINE_AA)
    label = "TOUCH THIS TARGET"
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    label_origin = (x - label_size[0] // 2, y - 112)
    cv2.putText(
        image,
        label,
        label_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        5,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        label,
        label_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_wall_depth_calibration(image: np.ndarray, progress: int, total: int) -> None:
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    radius = max(42, min(width, height) // 18)
    cv2.circle(image, center, radius, (60, 210, 255), 5, cv2.LINE_AA)
    if total:
        end_angle = int(360 * min(progress / total, 1.0))
        cv2.ellipse(
            image,
            center,
            (radius + 14, radius + 14),
            -90,
            0,
            end_angle,
            (80, 245, 115),
            8,
            cv2.LINE_AA,
        )
    label = "KEEP WALL CLEAR"
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    origin = (center[0] - label_size[0] // 2, center[1] - radius - 38)
    cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)


def guided_depth_touch_targets(width: int, height: int) -> np.ndarray:
    inset_x = int(width * 0.18)
    inset_y = int(height * 0.18)
    return np.array(
        [
            [width * 0.5, height * 0.5],
            [inset_x, inset_y],
            [width - inset_x, inset_y],
            [width - inset_x, height - inset_y],
            [inset_x, height - inset_y],
        ],
        dtype=np.float32,
    )


def guided_corner_touch_targets(width: int, height: int) -> np.ndarray:
    inset_x = int(width * 0.18)
    inset_y = int(height * 0.18)
    return np.array(
        [
            [inset_x, inset_y],
            [width - inset_x, inset_y],
            [width - inset_x, height - inset_y],
            [inset_x, height - inset_y],
        ],
        dtype=np.float32,
    )


def camera_landmarks(
    result: object,
    frame_shape: tuple[int, ...],
    offset: tuple[int, int] = (0, 0),
) -> np.ndarray | None:
    if not result.hand_landmarks:
        return None
    height, width = frame_shape[:2]
    offset_x, offset_y = offset
    return np.array(
        [
            [landmark.x * width + offset_x, landmark.y * height + offset_y]
            for landmark in result.hand_landmarks[0]
        ],
        dtype=np.float32,
    )


def draw_hand(frame: np.ndarray, points: np.ndarray, active: bool) -> None:
    color = (70, 230, 90) if active else (0, 190, 255)
    integer_points = np.rint(points).astype(int)
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, tuple(integer_points[start]), tuple(integer_points[end]), color, 2, cv2.LINE_AA)
    for point in integer_points:
        cv2.circle(frame, tuple(point), 3, color, -1, cv2.LINE_AA)
    cv2.circle(frame, tuple(integer_points[8]), 11, color, 3, cv2.LINE_AA)


def save_calibration(
    path: Path,
    camera_identity: str,
    frame_size: tuple[int, int],
    output_size: tuple[int, int],
    camera_points: np.ndarray,
    output_points: np.ndarray,
    touch_reference_scale: float | None,
    wall_depth_model: WallDepthModel | None = None,
    sensor_mode: str = "rgb",
    wall_depth_reference: np.ndarray | None = None,
    wall_depth_noise: np.ndarray | None = None,
    depth_touch_profile: DepthTouchProfile | None = None,
    spatial_touch_calibration: SpatialTouchCalibration | None = None,
) -> None:
    reference_path = depth_reference_path(path)
    if wall_depth_reference is not None:
        noise = (
            np.zeros_like(wall_depth_reference, dtype=np.float32)
            if wall_depth_noise is None
            else np.asarray(wall_depth_noise, dtype=np.float32)
        )
        np.savez_compressed(
            reference_path,
            depth_mm=np.asarray(wall_depth_reference, dtype=np.float32),
            noise_mm=noise,
        )
    elif wall_depth_model is None:
        reference_path.unlink(missing_ok=True)
    data = {
        "version": 4,
        "sensor_mode": sensor_mode,
        "depth_touch_mode": DEPTH_TOUCH_MODE if sensor_mode.endswith("-depth") else None,
        "camera_identity": camera_identity,
        "camera_frame_size": list(frame_size),
        "projector_output_size": list(output_size),
        "camera_points": np.asarray(camera_points, dtype=float).tolist(),
        "output_points": np.asarray(output_points, dtype=float).tolist(),
        "corner_order": list(CORNER_NAMES),
        "touch_reference_scale": touch_reference_scale,
        "wall_depth_model": wall_depth_model.to_dict() if wall_depth_model else None,
        "depth_reference_file": reference_path.name if wall_depth_reference is not None else None,
        "depth_touch_profile": depth_touch_profile.to_dict() if depth_touch_profile else None,
        "spatial_touch_calibration": (
            spatial_touch_calibration.to_dict()
            if spatial_touch_calibration is not None
            else None
        ),
    }
    path.write_text(json.dumps(data, indent=2) + "\n")


def depth_reference_path(calibration_path: Path) -> Path:
    return calibration_path.with_name(f"{calibration_path.stem}.depth.npz")


def load_calibration(
    path: Path,
    camera_identity: str,
    frame_size: tuple[int, int],
    output_size: tuple[int, int],
    sensor_mode: str = "rgb",
) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if data.get("camera_identity") != camera_identity:
            return None
        if tuple(data.get("camera_frame_size", ())) != frame_size:
            return None
        if tuple(data.get("projector_output_size", ())) != output_size:
            return None
        if data.get("sensor_mode", "rgb") != sensor_mode:
            return None
        data["camera_points"] = np.array(data["camera_points"], dtype=np.float32)
        data["output_points"] = np.array(data["output_points"], dtype=np.float32)
        if data.get("wall_depth_model") is not None:
            data["wall_depth_model"] = WallDepthModel.from_dict(data["wall_depth_model"])
        data["depth_touch_profile"] = (
            DepthTouchProfile.from_dict(data["depth_touch_profile"])
            if data.get("depth_touch_profile") is not None
            else None
        )
        try:
            data["spatial_touch_calibration"] = (
                SpatialTouchCalibration.from_dict(data["spatial_touch_calibration"])
                if (
                    data.get("spatial_touch_calibration") is not None
                    and data.get("depth_touch_mode") == DEPTH_TOUCH_MODE
                )
                else None
            )
        except ValueError:
            # Keep valid projection/wall data when upgrading the touch model.
            data["spatial_touch_calibration"] = None
        data["wall_depth_reference"] = None
        data["wall_depth_noise"] = None
        reference_file = data.get("depth_reference_file")
        if reference_file:
            reference_path = path.parent / Path(reference_file).name
            with np.load(reference_path, allow_pickle=False) as stored:
                reference = np.asarray(stored["depth_mm"], dtype=np.float32)
                noise = (
                    np.asarray(stored["noise_mm"], dtype=np.float32)
                    if "noise_mm" in stored.files
                    else None
                )
            expected_shape = (frame_size[1], frame_size[0])
            if reference.shape != expected_shape:
                raise ValueError("Saved depth reference has the wrong shape")
            if noise is not None and noise.shape != expected_shape:
                raise ValueError("Saved depth noise map has the wrong shape")
            data["wall_depth_reference"] = reference
            data["wall_depth_noise"] = noise
        return data
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def create_landmarker(model_path: Path, confidence: float):
    if not model_path.exists():
        raise RuntimeError(
            f"Hand model missing: {model_path}\n"
            "Download the official MediaPipe hand_landmarker.task model before running."
        )
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=confidence,
        min_hand_presence_confidence=confidence,
        min_tracking_confidence=0.50,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def fingertip_wall_gap_mm(
    wall_depth_reference: np.ndarray,
    depth_mm: np.ndarray,
    camera_point: np.ndarray,
    radius: int = 7,
) -> float | None:
    """Wall-minus-fingertip depth (mm) at an aligned camera pixel.

    Positive means the fingertip is in front of the learned wall plane. The
    capture backend aligns depth to color, so a MediaPipe landmark pixel indexes
    the same location in ``depth_mm`` and ``wall_depth_reference``.
    """
    finger = fingertip_depth_mm(depth_mm, camera_point, radius)
    wall = sample_fingertip_depth(
        wall_depth_reference, camera_point, radius=radius, percentile=50.0
    )
    if finger is None or wall is None:
        return None
    return float(wall - finger)


def fingertip_depth_mm(
    depth_mm: np.ndarray,
    camera_point: np.ndarray,
    radius: int = 7,
) -> float | None:
    """Read aligned Z at the MediaPipe tip, expanding only to fill depth holes."""
    radii = list(dict.fromkeys((min(2, radius), min(4, radius), radius)))
    for sample_radius in radii:
        sampled = sample_fingertip_depth(
            depth_mm,
            camera_point,
            radius=max(1, sample_radius),
            percentile=40.0,
        )
        if sampled is not None:
            return sampled
    return None


def depth_foreground_roi(
    wall_depth_reference: np.ndarray,
    depth_mm: np.ndarray,
    base_roi: tuple[int, int, int, int],
    min_gap_mm: float = 20.0,
    max_gap_mm: float = 400.0,
    margin: float = 0.35,
    min_pixels: int = 60,
) -> tuple[int, int, int, int] | None:
    """Tight bounding box (in full-frame pixels) around the near-wall foreground.

    Uses depth to localize the hand/arm in front of the wall inside ``base_roi``.
    Depth detects the hand regardless of how small it is in the color image, so
    cropping MediaPipe to this box lets the hand fill the detector's input and
    extends usable range when the camera is far. Returns ``None`` if nothing is
    clearly in front of the wall (caller should fall back to ``base_roi``).
    """
    x0, y0, x1, y1 = base_roi
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    wall = wall_depth_reference[y0:y1, x0:x1]
    current = depth_mm[y0:y1, x0:x1]
    valid = (wall >= 100.0) & (current >= 100.0)
    gap = wall - current
    foreground = valid & (gap >= min_gap_mm) & (gap <= max_gap_mm)
    ys, xs = np.nonzero(foreground)
    if xs.size < min_pixels:
        return None
    # Percentile bounds reject isolated depth-noise specks that a raw min/max
    # would let inflate the box.
    bx0, bx1 = np.percentile(xs, 1.0), np.percentile(xs, 99.0)
    by0, by1 = np.percentile(ys, 1.0), np.percentile(ys, 99.0)
    pad_x = int((bx1 - bx0) * margin) + 12
    pad_y = int((by1 - by0) * margin) + 12
    fx0 = max(x0, x0 + int(bx0) - pad_x)
    fy0 = max(y0, y0 + int(by0) - pad_y)
    fx1 = min(x1, x0 + int(bx1) + pad_x)
    fy1 = min(y1, y0 + int(by1) + pad_y)
    if fx1 - fx0 < 2 or fy1 - fy0 < 2:
        return None
    return fx0, fy0, fx1, fy1


def main() -> None:
    args = parse_args()
    apply_close_bottom_preset(args)
    if args.touch_plane_tolerance_mm <= 0:
        raise ValueError("--touch-plane-tolerance-mm must be positive")
    if args.touch_dwell_ms < 0:
        raise ValueError("--touch-dwell-ms cannot be negative")
    if args.fingertip_depth_radius < 1:
        raise ValueError("--fingertip-depth-radius must be positive")
    depth_camera: RealSenseCamera | OrbbecCamera | None = None
    cap: cv2.VideoCapture | None = None
    if args.sensor in ("auto", "realsense") and realsense_device_count() > 0:
        depth_camera = RealSenseCamera(
            preferred_width=args.camera_width or 1280,
            preferred_height=args.camera_height or 720,
            fps=args.camera_fps or 30,
            depth_preset=args.realsense_preset,
        )
        identity = depth_camera.identity
        sensor_mode = (
            "realsense-close-bottom-depth"
            if args.close_bottom
            else "realsense-depth"
        )
        first_rgbd = depth_camera.read(timeout_ms=3000)
        frame = first_rgbd.color_bgr
        depth_mm: np.ndarray | None = first_rgbd.depth_mm
    elif args.sensor in ("auto", "orbbec") and orbbec_device_count() > 0:
        depth_camera = OrbbecCamera(
            preferred_width=args.camera_width or 1280,
            preferred_height=args.camera_height or 720,
            fps=args.camera_fps or 30,
        )
        identity = depth_camera.identity
        sensor_mode = "orbbec-depth"
        first_rgbd = depth_camera.read(timeout_ms=3000)
        frame = first_rgbd.color_bgr
        depth_mm: np.ndarray | None = first_rgbd.depth_mm
    else:
        if args.sensor == "realsense":
            raise RuntimeError(
                "No RealSense camera is available. Confirm it appears in lsusb, "
                "reconnect its USB 3 data cable, and install pyrealsense2."
            )
        if args.sensor == "orbbec":
            raise RuntimeError(
                "No Orbbec camera is available. The Gemini 336 must appear as "
                "2bc5:0803 in lsusb; reconnect its USB 3 data cable and install udev rules."
            )
        camera_path, identity = validate_camera(args.camera)
        pixel_format, camera_width, camera_height, camera_fps = camera_stream_profile(
            identity,
            args.camera_format,
            args.camera_width,
            args.camera_height,
            args.camera_fps,
        )
        cap = open_camera(camera_path, camera_width, camera_height, camera_fps, pixel_format)
        try:
            frame = read_camera_frame(cap)
        except RuntimeError:
            cap.release()
            raise
        depth_mm = None
        sensor_mode = "rgb"

    depth_enabled = depth_camera is not None
    depth_intrinsics = depth_camera.intrinsics if depth_camera is not None else None
    print(f"Wall Touch Demo v{APP_VERSION}")
    if args.close_bottom:
        print(
            "Placement preset: close-bottom "
            "(640x480 high-density depth, 60 wall frames, +/-12 mm touch)"
        )
    if depth_camera is not None:
        print(f"Depth camera: {identity}")
        print(f"Camera stream: {depth_camera.stream_description}")
    else:
        print(f"External camera: {identity} ({Path(camera_path).resolve()})")
        print(f"Camera request: {pixel_format} {camera_width}x{camera_height} at {camera_fps} FPS")
    print(
        f"Projector: {args.projector_width}x{args.projector_height} "
        f"at desktop ({args.projector_x},{args.projector_y})"
    )
    frame_height, frame_width = frame.shape[:2]
    frame_size = (frame_width, frame_height)
    if depth_intrinsics is not None and (
        depth_intrinsics.width != frame_width
        or depth_intrinsics.height != frame_height
    ):
        raise RuntimeError(
            "Color intrinsics do not match the aligned RGB-D frame: "
            f"{depth_intrinsics.width}x{depth_intrinsics.height} vs "
            f"{frame_width}x{frame_height}"
        )
    output_size = (args.projector_width, args.projector_height)
    if cap is not None:
        print(
            f"Camera stream: {capture_fourcc(cap) or 'unknown'} "
            f"{frame_width}x{frame_height} at {cap.get(cv2.CAP_PROP_FPS):.1f} FPS"
        )

    projector_window = "Wall Touch Paint - PROJECTOR"
    debug_window = f"Wall Touch Setup - {identity}"
    cv2.namedWindow(projector_window, cv2.WINDOW_NORMAL)
    cv2.moveWindow(projector_window, args.projector_x, args.projector_y)
    cv2.resizeWindow(projector_window, args.projector_width, args.projector_height)
    cv2.namedWindow(debug_window, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(debug_window, args.debug_x, args.debug_y)
    if not args.windowed:
        cv2.setWindowProperty(projector_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    output_points = projector_targets(*output_size)
    saved = None if args.fresh else load_calibration(
        args.calibration, identity, frame_size, output_size, sensor_mode
    )
    camera_points = saved["camera_points"] if saved else None
    output_points = saved["output_points"] if saved else output_points
    touch_reference = saved.get("touch_reference_scale") if saved and not depth_enabled else None
    wall_depth_model = saved.get("wall_depth_model") if saved and depth_enabled else None
    wall_depth_reference = (
        saved.get("wall_depth_reference") if saved and depth_enabled else None
    )
    wall_depth_noise = saved.get("wall_depth_noise") if saved and depth_enabled else None
    depth_touch_profile = (
        saved.get("depth_touch_profile") if saved and depth_enabled else None
    )
    spatial_touch_calibration = (
        saved.get("spatial_touch_calibration") if saved and depth_enabled else None
    )
    if (
        (depth_touch_profile is not None or spatial_touch_calibration is not None)
        and saved.get("depth_touch_mode") != DEPTH_TOUCH_MODE
    ):
        depth_touch_profile = None
        spatial_touch_calibration = None
        print(
            "The saved touch profile used an older contact model. Wall geometry "
            "is being kept; four-point touch calibration will be relearned."
        )
    if depth_touch_profile is not None:
        tightened_profile = depth_touch_profile.tightened(
            maximum_gap_mm=args.depth_calibration_max_gap_mm
        )
        if tightened_profile != depth_touch_profile:
            print(
                "Tightened saved touch profile: "
                f"maximum gap {tightened_profile.maximum_gap_mm:.0f} mm."
            )
        depth_touch_profile = tightened_profile
    if depth_enabled and args.recalibrate_depth and saved:
        wall_depth_model = None
        wall_depth_reference = None
        wall_depth_noise = None
        depth_touch_profile = None
        spatial_touch_calibration = None
        print("Keeping projection points; depth and guided touch will be recalibrated.")
    matrix = build_homography(camera_points, output_points) if camera_points is not None else None
    detection_roi = camera_detection_roi(camera_points, frame_size) if camera_points is not None else None
    if saved:
        print(f"Loaded calibration: {args.calibration}")
        print("If the camera or projector moved, press r and recalibrate.")

    clicks: list[list[float]] = []
    touch_samples: list[float] = []
    wall_depth_samples: list[WallDepthModel] = []
    wall_depth_frames: list[np.ndarray] = []
    guided_touch_samples: list[DepthContactObservation] = []
    current_target_samples: list[DepthContactObservation] = []
    fingertip_touch_points: list[np.ndarray] = []
    fingertip_touch_depths: list[float] = []
    current_fingertip_points: list[np.ndarray] = []
    current_fingertip_depths: list[float] = []
    depth_touch_target_index = 0
    depth_touch_targets = (
        guided_depth_touch_targets(*output_size)
        if args.legacy_depth_blob
        else guided_corner_touch_targets(*output_size)
    )
    collecting_wall_depth = (
        depth_enabled
        and matrix is not None
        and (
            wall_depth_model is None
            or wall_depth_reference is None
            or wall_depth_noise is None
        )
    )
    collecting_touch = not depth_enabled and matrix is not None and touch_reference is None
    depth_tracker = (
        DepthContactTracker(
            wall_depth_reference,
            camera_points,
            wall_noise_mm=wall_depth_noise,
            minimum_change_mm=args.depth_change_min_mm,
            minimum_component_area=args.depth_component_min_area,
            near_wall_limit_mm=args.depth_calibration_max_gap_mm,
            noise_multiplier=args.depth_noise_multiplier,
        )
        if (
            wall_depth_reference is not None
            and wall_depth_noise is not None
            and camera_points is not None
        )
        else None
    )
    collecting_depth_touch = bool(
        depth_enabled
        and matrix is not None
        and depth_tracker is not None
        and (
            (args.legacy_depth_blob and depth_touch_profile is None)
            or (not args.legacy_depth_blob and spatial_touch_calibration is None)
        )
        and not collecting_wall_depth
    )
    depth_lock = (
        DepthContactLock(depth_touch_profile)
        if depth_touch_profile is not None
        else None
    )
    if collecting_depth_touch:
        if args.legacy_depth_blob:
            print(
                "Guided hand calibration needed. Center an open hand on each target "
                "and press it against the wall."
            )
        else:
            print(
                "Four-point touch calibration needed. Touch each corner target "
                "with a straight index finger."
            )
    base_canvas = make_base_canvas(*output_size)
    canvas = base_canvas.copy()
    brush = PaintBrush(args.brush_radius, args.paint_alpha)
    spill = WatercolorPool(*output_size)
    ripple = WatercolorPool(
        *output_size,
        simulation_width=360,
        ripple_contrast=1.55,
        water_color=(20, 15, 105),
        reflection_color=(205, 226, 255),
        reflection_gain=2.35,
    )
    pulse = PulseGrid(*output_size)
    constellation = ConstellationField(*output_size)
    sand = MagneticSand(*output_size)
    tic_tac_toe = TicTacToe(*output_size)
    connect_four = PrismConnectFour(*output_size)
    orbit_keeper = OrbitKeeper(*output_size)
    reactive_effects = (
        spill,
        ripple,
        pulse,
        constellation,
        sand,
        tic_tac_toe,
        connect_four,
        orbit_keeper,
    )
    interaction_mode = args.mode
    gate = TouchGate(
        reference_scale=touch_reference,
        minimum_ratio=args.touch_scale_min,
        maximum_ratio=args.touch_scale_max,
        dwell_seconds=args.touch_dwell_ms / 1000.0,
    )
    depth_gate = DepthTouchGate(
        minimum_gap_mm=(
            depth_touch_profile.minimum_gap_mm
            if depth_touch_profile is not None
            else args.touch_min_gap_mm
            if spatial_touch_calibration is not None
            else args.touch_min_gap_mm
        ),
        maximum_gap_mm=(
            depth_touch_profile.maximum_gap_mm
            if depth_touch_profile is not None
            else min(
                args.touch_max_gap_mm,
                max(
                    args.touch_plane_tolerance_mm,
                    spatial_touch_calibration.tolerance_mm,
                ),
            )
            if spatial_touch_calibration is not None
            else args.touch_max_gap_mm
        ),
        dwell_seconds=args.touch_dwell_ms / 1000.0,
    )
    smoothed_tip: np.ndarray | None = None
    smoothed_scale: float | None = None
    smoothed_gap_mm: float | None = None
    last_timestamp_ms = 0
    last_timestamp_ms_alt = 0
    last_frame_time = time.monotonic()
    fps_history: deque[float] = deque(maxlen=30)
    fullscreen = not args.windowed
    geometry_message = ""
    depth_calibration_message = ""
    last_ripple_time = -1e9
    last_depth_touch_active = False
    last_depth_target_diagnostic = -1e9

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN and matrix is None and len(clicks) < 4:
            clicks.append([float(x), float(y)])
            print(f"Geometry {CORNER_NAMES[len(clicks) - 1]}: {x}, {y}")

    cv2.setMouseCallback(debug_window, on_mouse)
    landmarker = None if (depth_enabled and args.legacy_depth_blob) else create_landmarker(
        args.model, args.detection_confidence
    )
    detection_clahe = (
        cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if args.enhance_contrast
        else None
    )
    # Separate instance for the inverted-image fallback: MediaPipe VIDEO mode keeps
    # per-stream tracking state, so the negative pass needs its own landmarker and
    # its own monotonic timestamp counter.
    landmarker_alt = (
        create_landmarker(args.model, args.detection_confidence)
        if (landmarker is not None and args.dual_detect)
        else None
    )

    try:
        while True:
            if len(clicks) == 4 and matrix is None:
                candidate_points = np.array(clicks, dtype=np.float32)
                try:
                    validate_camera_quad(candidate_points, frame_size)
                    matrix = build_homography(candidate_points, output_points)
                except ValueError as error:
                    geometry_message = str(error)
                    clicks.clear()
                    print(f"Calibration rejected: {error}")
                else:
                    camera_points = candidate_points
                    detection_roi = camera_detection_roi(camera_points, frame_size)
                    touch_reference = None
                    wall_depth_model = None
                    wall_depth_reference = None
                    wall_depth_noise = None
                    depth_touch_profile = None
                    spatial_touch_calibration = None
                    depth_tracker = None
                    depth_lock = None
                    touch_samples.clear()
                    wall_depth_samples.clear()
                    wall_depth_frames.clear()
                    guided_touch_samples.clear()
                    current_target_samples.clear()
                    fingertip_touch_points.clear()
                    fingertip_touch_depths.clear()
                    current_fingertip_points.clear()
                    current_fingertip_depths.clear()
                    depth_touch_target_index = 0
                    collecting_wall_depth = depth_enabled
                    collecting_depth_touch = False
                    collecting_touch = not depth_enabled
                    smoothed_scale = None
                    smoothed_gap_mm = None
                    geometry_message = ""
                    depth_calibration_message = ""
                    gate.set_reference(None)
                    depth_gate.reset()
                    save_calibration(
                        args.calibration, identity, frame_size, output_size,
                        camera_points, output_points, touch_reference,
                        wall_depth_model, sensor_mode,
                    )
                    area_percent = 100 * abs(cv2.contourArea(camera_points)) / (frame_width * frame_height)
                    print(f"Geometry accepted ({area_percent:.2f}% of camera frame).")
                    if depth_enabled:
                        print("Keep the projected wall clear while depth calibration completes automatically.")
                    else:
                        print("Walk to the wall and touch the labeled center target; sampling starts automatically.")

            if depth_camera is not None:
                rgbd = depth_camera.read(timeout_ms=800)
                frame = rgbd.color_bgr
                depth_mm = rgbd.depth_mm
            else:
                frame = read_camera_frame(cap, timeout_seconds=0.8)
                depth_mm = None
            now = time.monotonic()
            delta = now - last_frame_time
            last_frame_time = now
            if delta > 0:
                fps_history.append(1.0 / delta)

            debug = frame.copy()
            if (
                collecting_wall_depth
                and depth_mm is not None
                and camera_points is not None
            ):
                try:
                    candidate_wall = fit_wall_depth_model(depth_mm, camera_points)
                    if candidate_wall.rmse_mm > 35.0:
                        raise ValueError(
                            f"Wall depth is noisy ({candidate_wall.rmse_mm:.1f} mm RMSE)"
                        )
                except ValueError as error:
                    depth_calibration_message = str(error)
                else:
                    wall_depth_samples.append(candidate_wall)
                    wall_depth_frames.append(depth_mm.copy())
                    depth_calibration_message = ""
                    if len(wall_depth_samples) >= args.depth_calibration_frames:
                        wall_depth_model = combine_wall_depth_models(wall_depth_samples)
                        depth_stack = np.stack(wall_depth_frames)
                        masked_depth = np.ma.masked_less(depth_stack, 100.0)
                        wall_depth_reference = np.ma.median(
                            masked_depth, axis=0
                        ).filled(0.0).astype(np.float32)
                        depth_deviation = np.ma.abs(
                            masked_depth - wall_depth_reference
                        )
                        wall_depth_noise = (
                            1.4826 * np.ma.median(depth_deviation, axis=0)
                        ).filled(args.touch_max_gap_mm).astype(np.float32)
                        depth_tracker = DepthContactTracker(
                            wall_depth_reference,
                            camera_points,
                            wall_noise_mm=wall_depth_noise,
                            minimum_change_mm=args.depth_change_min_mm,
                            minimum_component_area=args.depth_component_min_area,
                            near_wall_limit_mm=args.touch_max_gap_mm + 15.0,
                            noise_multiplier=args.depth_noise_multiplier,
                        )
                        collecting_wall_depth = False
                        collecting_depth_touch = True
                        guided_touch_samples.clear()
                        current_target_samples.clear()
                        fingertip_touch_points.clear()
                        fingertip_touch_depths.clear()
                        current_fingertip_points.clear()
                        current_fingertip_depths.clear()
                        depth_touch_target_index = 0
                        depth_gate.reset()
                        save_calibration(
                            args.calibration,
                            identity,
                            frame_size,
                            output_size,
                            camera_points,
                            output_points,
                            None,
                            wall_depth_model,
                            sensor_mode,
                            wall_depth_reference,
                            wall_depth_noise,
                        )
                        print(
                            "Wall depth learned: "
                            f"{wall_depth_model.rmse_mm:.1f} mm RMSE from "
                            f"{len(wall_depth_samples)} frames."
                        )
                        if args.legacy_depth_blob:
                            print(
                                "Guided hand calibration started. Center an open hand on each "
                                "target and press it against the wall."
                            )
                        else:
                            print(
                                "Wall depth learned. Touch the four corner targets with "
                                "a straight index finger to calibrate the angled plane."
                            )
            landmarks = None
            depth_camera_point = None
            depth_observation = None
            depth_observations: list[DepthContactObservation] = []
            if (
                depth_enabled
                and args.legacy_depth_blob
                and depth_tracker is not None
                and depth_mm is not None
            ):
                depth_observations = depth_tracker.observations(depth_mm)
                if collecting_depth_touch and matrix is not None:
                    target_output = depth_touch_targets[depth_touch_target_index]
                    target_camera = transform_points(
                        np.linalg.inv(matrix), target_output.reshape(1, 2)
                    )[0]
                    current_depth = depth_tracker.current_depth_mm
                    target_metrics = (
                        depth_target_foreground_metrics(
                            wall_depth_reference,
                            current_depth,
                            wall_depth_noise,
                            target_camera,
                        )
                        if (
                            current_depth is not None
                            and wall_depth_reference is not None
                            and wall_depth_noise is not None
                        )
                        else (False, 0.0, 0, 0)
                    )
                    target_present = target_metrics[0]
                    if now - last_depth_target_diagnostic >= 2.0:
                        nearest_distance = min(
                            (
                                float(np.linalg.norm(item.camera_point - target_camera))
                                for item in depth_observations
                            ),
                            default=float("nan"),
                        )
                        print(
                            f"Target {depth_touch_target_index + 1} depth: "
                            f"p90={target_metrics[1]:.0f} mm, "
                            f"strong={target_metrics[2]}, "
                            f"component-distance={nearest_distance:.0f} px"
                        )
                        last_depth_target_diagnostic = now
                    nearby = [
                        observation
                        for observation in depth_observations
                        if np.linalg.norm(observation.camera_point - target_camera) <= 140.0
                        and observation.gap_mm <= args.depth_calibration_max_gap_mm
                        and observation.contact_area >= depth_tracker.minimum_contact_area
                    ]
                    if target_present and nearby:
                        sample = min(
                            nearby,
                            key=lambda observation: float(
                                np.linalg.norm(observation.camera_point - target_camera)
                            ),
                        )
                        current_target_samples.append(sample)
                        if len(current_target_samples) == 1:
                            print(
                                f"Target {depth_touch_target_index + 1}: contact seen "
                                f"({sample.gap_mm:.0f} mm, hand contact area "
                                f"{sample.contact_area})."
                            )
                        depth_camera_point = sample.camera_point
                        if len(current_target_samples) >= args.depth_touch_samples:
                            guided_touch_samples.extend(current_target_samples)
                            current_target_samples.clear()
                            depth_touch_target_index += 1
                            if depth_touch_target_index >= len(depth_touch_targets):
                                depth_touch_profile = DepthTouchProfile.fit(
                                    guided_touch_samples,
                                    maximum_contact_gap_mm=args.depth_calibration_max_gap_mm,
                                )
                                depth_lock = DepthContactLock(depth_touch_profile)
                                collecting_depth_touch = False
                                depth_gate.minimum_gap_mm = depth_touch_profile.minimum_gap_mm
                                depth_gate.maximum_gap_mm = depth_touch_profile.maximum_gap_mm
                                depth_gate.reset()
                                save_calibration(
                                    args.calibration,
                                    identity,
                                    frame_size,
                                    output_size,
                                    camera_points,
                                    output_points,
                                    None,
                                    wall_depth_model,
                                    sensor_mode,
                                    wall_depth_reference,
                                    wall_depth_noise,
                                    depth_touch_profile,
                                )
                                print(
                                    "Guided hand contact learned: "
                                    f"gap {depth_touch_profile.minimum_gap_mm:.0f}-"
                                    f"{depth_touch_profile.maximum_gap_mm:.0f} mm, "
                                    f"area {depth_touch_profile.minimum_component_area}-"
                                    f"{depth_touch_profile.maximum_component_area}."
                                )
                            else:
                                print(
                                    f"Touch target {depth_touch_target_index}/"
                                    f"{len(depth_touch_targets)} captured. Move to the next target."
                                )
                elif depth_lock is not None:
                    depth_observation = depth_lock.update(depth_observations)
                if depth_observation is not None:
                    depth_camera_point = depth_observation.camera_point
            elif detection_roi is not None and landmarker is not None:
                # Prefer a tight, depth-localized crop: depth sees the hand no matter
                # how small it is in color, so cropping to it lets a far hand fill the
                # detector and greatly extends usable range.
                active_roi = detection_roi
                if (
                    args.depth_roi
                    and depth_enabled
                    and depth_mm is not None
                    and wall_depth_reference is not None
                ):
                    guided_roi = depth_foreground_roi(
                        wall_depth_reference, depth_mm, detection_roi
                    )
                    if guided_roi is not None:
                        active_roi = guided_roi
                x0, y0, x1, y1 = active_roi
                tracking_frame = frame[y0:y1, x0:x1]
                # Resize the crop so its long side matches --detection-size: upscale a
                # tight/far crop so the hand is big enough to detect, downscale a large
                # one to cut lag. Landmarks are normalized, so mapping back is unaffected.
                detect_image = tracking_frame
                long_side = max(tracking_frame.shape[0], tracking_frame.shape[1])
                if long_side > 0 and args.detection_size > 0:
                    det_scale = min(args.detection_size / long_side, 4.0)
                    if abs(det_scale - 1.0) > 0.02:
                        detect_image = cv2.resize(
                            tracking_frame,
                            None,
                            fx=det_scale,
                            fy=det_scale,
                            interpolation=cv2.INTER_AREA if det_scale < 1.0 else cv2.INTER_LINEAR,
                        )
                if detection_clahe is not None:
                    lab = cv2.cvtColor(detect_image, cv2.COLOR_BGR2LAB)
                    channels = list(cv2.split(lab))
                    channels[0] = detection_clahe.apply(channels[0])
                    detect_image = cv2.cvtColor(cv2.merge(channels), cv2.COLOR_LAB2BGR)
                rgb = cv2.cvtColor(detect_image, cv2.COLOR_BGR2RGB)
                timestamp_ms = max(last_timestamp_ms + 1, int(now * 1000))
                last_timestamp_ms = timestamp_ms
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)),
                    timestamp_ms,
                )
                # Landmarks are normalized, so map them back with the original ROI
                # shape regardless of the downscale used for inference.
                landmarks = camera_landmarks(result, tracking_frame.shape, (x0, y0))
                if landmarks is None and landmarker_alt is not None:
                    # Retry on the negative image: on a bright/washed wall the hand
                    # often only separates from the background once light and dark swap.
                    rgb_inv = np.ascontiguousarray(255 - rgb)
                    timestamp_ms_alt = max(last_timestamp_ms_alt + 1, int(now * 1000))
                    last_timestamp_ms_alt = timestamp_ms_alt
                    result_alt = landmarker_alt.detect_for_video(
                        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_inv),
                        timestamp_ms_alt,
                    )
                    landmarks = camera_landmarks(result_alt, tracking_frame.shape, (x0, y0))
                    if landmarks is not None:
                        cv2.putText(
                            debug, "NEG", (x0 + 4, y0 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 200, 255), 2, cv2.LINE_AA,
                        )
                cv2.rectangle(debug, (x0, y0), (x1 - 1, y1 - 1), (210, 150, 40), 1)
            mapped_landmarks = None
            mapped_tip = None
            scale = None
            gap_mm = None
            extended = False
            inside = False
            if (
                depth_enabled
                and not args.legacy_depth_blob
                and landmarks is not None
                and matrix is not None
            ):
                mapped_landmarks = transform_points(matrix, landmarks)
                raw_tip = mapped_landmarks[8]
                if smoothed_tip is None:
                    smoothed_tip = raw_tip.copy()
                else:
                    smoothed_tip = 0.15 * smoothed_tip + 0.85 * raw_tip
                mapped_tip = smoothed_tip.copy()
                finger_depth_mm = (
                    fingertip_depth_mm(
                        depth_mm,
                        landmarks[8],
                        radius=args.fingertip_depth_radius,
                    )
                    if depth_mm is not None
                    else None
                )
                raw_wall_gap_mm = (
                    fingertip_wall_gap_mm(
                        wall_depth_reference,
                        depth_mm,
                        landmarks[8],
                        radius=args.fingertip_depth_radius,
                    )
                    if (wall_depth_reference is not None and depth_mm is not None)
                    else None
                )
                extended = index_is_extended(landmarks)
                if (
                    collecting_depth_touch
                    and raw_wall_gap_mm is not None
                    and finger_depth_mm is not None
                    and extended
                    and args.touch_min_gap_mm <= raw_wall_gap_mm
                    <= args.depth_calibration_max_gap_mm
                ):
                    target = depth_touch_targets[depth_touch_target_index]
                    target_distance = float(np.linalg.norm(raw_tip - target))
                    if target_distance <= min(output_size) * 0.10:
                        current_fingertip_points.append(landmarks[8].copy())
                        current_fingertip_depths.append(float(finger_depth_mm))
                        if len(current_fingertip_points) > args.depth_touch_samples:
                            current_fingertip_points.pop(0)
                            current_fingertip_depths.pop(0)
                        if len(current_fingertip_points) == 1:
                            print(
                                f"Corner touch {depth_touch_target_index + 1}: "
                                f"contact seen at wall gap {raw_wall_gap_mm:.1f} mm."
                            )
                        point_window = np.stack(current_fingertip_points)
                        depth_window = np.asarray(current_fingertip_depths)
                        point_center = np.median(point_window, axis=0)
                        stable_contact = bool(
                            len(current_fingertip_points) >= args.depth_touch_samples
                            and np.max(
                                np.linalg.norm(point_window - point_center, axis=1)
                            )
                            <= 12.0
                            and np.percentile(depth_window, 90)
                            - np.percentile(depth_window, 10)
                            <= 16.0
                        )
                        if stable_contact:
                            fingertip_touch_points.append(
                                point_center.astype(np.float32)
                            )
                            fingertip_touch_depths.append(
                                float(np.median(current_fingertip_depths))
                            )
                            current_fingertip_points.clear()
                            current_fingertip_depths.clear()
                            depth_touch_target_index += 1
                            if depth_touch_target_index >= len(depth_touch_targets):
                                touched_camera_points = np.stack(fingertip_touch_points)
                                spatial_touch_calibration = SpatialTouchCalibration.fit(
                                    touched_camera_points,
                                    np.asarray(fingertip_touch_depths),
                                    depth_intrinsics,
                                )
                                matrix = build_homography(
                                    touched_camera_points,
                                    depth_touch_targets,
                                )
                                camera_points = transform_points(
                                    np.linalg.inv(matrix),
                                    output_points,
                                )
                                validate_camera_quad(camera_points, frame_size)
                                detection_roi = camera_detection_roi(
                                    camera_points, frame_size
                                )
                                collecting_depth_touch = False
                                depth_gate.minimum_gap_mm = args.touch_min_gap_mm
                                depth_gate.maximum_gap_mm = min(
                                    args.touch_max_gap_mm,
                                    max(
                                        args.touch_plane_tolerance_mm,
                                        spatial_touch_calibration.tolerance_mm,
                                    ),
                                )
                                depth_gate.reset()
                                save_calibration(
                                    args.calibration,
                                    identity,
                                    frame_size,
                                    output_size,
                                    camera_points,
                                    output_points,
                                    None,
                                    wall_depth_model,
                                    sensor_mode,
                                    wall_depth_reference,
                                    wall_depth_noise,
                                    None,
                                    spatial_touch_calibration,
                                )
                                print(
                                    "Four-point 3D touch plane learned: active at "
                                    f"{depth_gate.minimum_gap_mm:.1f} to "
                                    f"+{depth_gate.maximum_gap_mm:.1f} mm."
                                )
                            else:
                                print(
                                    f"Corner touch {depth_touch_target_index}/"
                                    f"{len(depth_touch_targets)} captured. "
                                    "Move to the next target."
                                )
                if (
                    finger_depth_mm is not None
                    and spatial_touch_calibration is not None
                    and depth_intrinsics is not None
                ):
                    raw_gap_mm = spatial_touch_calibration.signed_distance_mm(
                        landmarks[8],
                        finger_depth_mm,
                        depth_intrinsics,
                    )
                else:
                    raw_gap_mm = raw_wall_gap_mm
                if raw_gap_mm is None:
                    smoothed_gap_mm = None
                    gap_mm = None
                else:
                    if smoothed_gap_mm is None:
                        smoothed_gap_mm = raw_gap_mm
                    else:
                        smoothed_gap_mm = 0.25 * smoothed_gap_mm + 0.75 * raw_gap_mm
                    gap_mm = smoothed_gap_mm
                depth_camera_point = landmarks[8].copy()
                inside = point_in_output(mapped_tip, *output_size, margin=4)
            elif depth_observation is not None and matrix is not None:
                raw_tip = transform_points(
                    matrix, depth_observation.camera_point.reshape(1, 2)
                )[0]
                if smoothed_tip is None:
                    smoothed_tip = raw_tip.copy()
                else:
                    smoothed_tip = 0.15 * smoothed_tip + 0.85 * raw_tip
                mapped_tip = smoothed_tip.copy()
                raw_gap_mm = depth_observation.gap_mm
                if smoothed_gap_mm is None:
                    smoothed_gap_mm = raw_gap_mm
                else:
                    smoothed_gap_mm = 0.25 * smoothed_gap_mm + 0.75 * raw_gap_mm
                gap_mm = smoothed_gap_mm
                extended = True
                inside = point_in_output(mapped_tip, *output_size, margin=4)
            elif landmarks is not None and matrix is not None:
                mapped_landmarks = transform_points(matrix, landmarks)
                raw_tip = mapped_landmarks[8]
                if smoothed_tip is None:
                    smoothed_tip = raw_tip.copy()
                else:
                    smoothed_tip = 0.55 * smoothed_tip + 0.45 * raw_tip
                mapped_tip = smoothed_tip.copy()
                raw_scale = hand_plane_scale(mapped_landmarks)
                if smoothed_scale is None:
                    smoothed_scale = raw_scale
                else:
                    smoothed_scale = 0.72 * smoothed_scale + 0.28 * raw_scale
                scale = smoothed_scale
                extended = index_is_extended(landmarks)
                inside = point_in_output(mapped_tip, *output_size, margin=4)
            elif landmarks is None and depth_observation is None:
                smoothed_tip = None
                smoothed_gap_mm = None

            if matrix is None:
                if depth_enabled:
                    decision = depth_gate.update(
                        gap_mm=None,
                        point=None,
                        timestamp=now,
                        inside=False,
                        index_extended=False,
                        calibrated=False,
                    )
                else:
                    decision = gate.update(
                        scale=None, point=None, timestamp=now, inside=False, index_extended=False
                    )
            elif depth_enabled:
                if args.legacy_depth_blob:
                    depth_calibrated = (
                        depth_lock is not None
                        and not collecting_wall_depth
                        and not collecting_depth_touch
                    )
                    depth_index_extended = True
                else:
                    depth_calibrated = (
                        wall_depth_reference is not None
                        and spatial_touch_calibration is not None
                        and not collecting_wall_depth
                        and not collecting_depth_touch
                    )
                    depth_index_extended = extended or not args.require_index_extension
                decision = depth_gate.update(
                    gap_mm=gap_mm,
                    point=mapped_tip,
                    timestamp=now,
                    inside=inside,
                    index_extended=depth_index_extended,
                    calibrated=depth_calibrated,
                )
            else:
                decision = gate.update(
                    scale=scale,
                    point=mapped_tip,
                    timestamp=now,
                    inside=inside,
                    index_extended=extended or not args.require_index_extension,
                )

            if depth_enabled and decision.active != last_depth_touch_active:
                state = "started" if decision.active else "ended"
                gap_text = "--" if decision.distance_mm is None else f"{decision.distance_mm:.0f} mm"
                print(f"Depth touch {state}: wall gap {gap_text}")
                last_depth_touch_active = decision.active

            if collecting_touch and scale is not None and mapped_tip is not None:
                target = np.array([args.projector_width / 2, args.projector_height / 2], dtype=np.float32)
                if np.linalg.norm(mapped_tip - target) < min(output_size) * 0.16:
                    touch_samples.append(scale)
                if len(touch_samples) >= args.touch_samples:
                    touch_reference = float(np.median(touch_samples))
                    gate.set_reference(touch_reference)
                    collecting_touch = False
                    save_calibration(
                        args.calibration, identity, frame_size, output_size,
                        camera_points, output_points, touch_reference,
                        None, sensor_mode,
                    )
                    print(f"Touch plane learned: reference hand scale={touch_reference:.1f}")

            game_ready = bool(
                matrix is not None
                and not collecting_wall_depth
                and not collecting_depth_touch
                and not collecting_touch
            )
            if interaction_mode == "tic-tac-toe":
                tic_tac_toe.update(mapped_tip, decision.active, now)
            elif interaction_mode == "connect-four":
                connect_four.update(mapped_tip, decision.active, now)
            elif interaction_mode == "orbit-keeper":
                orbit_keeper.update(
                    mapped_tip if game_ready else None,
                    decision.active and game_ready,
                    now,
                    delta if game_ready else 0.0,
                )
            elif decision.active and mapped_tip is not None:
                color = paint_color(mapped_tip, *output_size)
                if interaction_mode == "spill":
                    spill.add_drop(mapped_tip, color, args.brush_radius)
                elif interaction_mode == "ripple":
                    if now - last_ripple_time >= 0.28:
                        ripple.add_ripple(mapped_tip, args.brush_radius + 18, strength=0.09)
                        last_ripple_time = now
                elif interaction_mode == "pulse":
                    pulse.add_pulse(mapped_tip, color, now)
                elif interaction_mode == "paint":
                    brush.apply(canvas, mapped_tip, color)
                elif interaction_mode == "constellation":
                    constellation.add(mapped_tip, color, now)
                elif interaction_mode == "sand":
                    sand.attract(mapped_tip)

            if interaction_mode == "spill":
                spill.step()
                art_frame = spill.render()
            elif interaction_mode == "ripple":
                ripple.step()
                art_frame = ripple.render()
            elif interaction_mode == "pulse":
                pulse.step(delta)
                art_frame = pulse.render()
            elif interaction_mode == "constellation":
                constellation.step(delta)
                art_frame = constellation.render()
            elif interaction_mode == "sand":
                sand.step(delta)
                art_frame = sand.render()
            elif interaction_mode == "tic-tac-toe":
                art_frame = tic_tac_toe.render(now)
            elif interaction_mode == "connect-four":
                art_frame = connect_four.render(now)
            elif interaction_mode == "orbit-keeper":
                art_frame = orbit_keeper.render(now)
            else:
                art_frame = canvas.copy()

            if matrix is None:
                projector_frame = np.empty_like(canvas)
                draw_projector_targets(projector_frame, output_points, min(len(clicks), 3))
            else:
                projector_frame = art_frame
                if collecting_wall_depth:
                    draw_wall_depth_calibration(
                        projector_frame,
                        len(wall_depth_samples),
                        args.depth_calibration_frames,
                    )
                elif collecting_depth_touch:
                    target = depth_touch_targets[depth_touch_target_index]
                    draw_touch_target(
                        projector_frame,
                        target,
                        (
                            len(current_target_samples)
                            if args.legacy_depth_blob
                            else len(current_fingertip_points)
                        ),
                        args.depth_touch_samples,
                    )
                elif not depth_enabled and (touch_reference is None or collecting_touch):
                    center = np.array([args.projector_width / 2, args.projector_height / 2])
                    draw_touch_target(
                        projector_frame, center,
                        len(touch_samples) if collecting_touch else 0,
                        args.touch_samples if collecting_touch else 0,
                    )
                elif mapped_tip is not None and inside:
                    point = tuple(np.rint(mapped_tip).astype(int))
                    cursor_color = (70, 220, 90) if decision.active else (30, 190, 255)
                    cursor_radius = (
                        28
                        if interaction_mode
                        in {"tic-tac-toe", "connect-four", "orbit-keeper"}
                        else args.brush_radius + 8
                    )
                    cv2.circle(projector_frame, point, cursor_radius, cursor_color, 5, cv2.LINE_AA)

            if camera_points is not None:
                cv2.polylines(debug, [np.rint(camera_points).astype(np.int32)], True, (80, 235, 100), 3)
                if projection_near_frame_edge(camera_points, frame_size):
                    draw_outlined_text(
                        debug,
                        "FRAMING WARNING: center the projection; the hand may leave the camera image",
                        (14, 88),
                        0.54,
                    )
            else:
                for index, point in enumerate(clicks):
                    cv2.circle(debug, tuple(np.rint(point).astype(int)), 8, (0, 220, 255), -1)
                    cv2.putText(debug, str(index + 1), tuple(np.rint(point).astype(int) + (12, -10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
            if landmarks is not None:
                draw_hand(debug, landmarks, decision.active)
            elif depth_camera_point is not None:
                camera_cursor = tuple(np.rint(depth_camera_point).astype(int))
                cursor_color = (70, 230, 90) if decision.active else (0, 190, 255)
                cv2.circle(debug, camera_cursor, 13, cursor_color, 3, cv2.LINE_AA)
                cv2.circle(debug, camera_cursor, 3, cursor_color, -1, cv2.LINE_AA)

            if matrix is None:
                next_corner = CORNER_NAMES[len(clicks)] if len(clicks) < 4 else "processing"
                status = f"GEOMETRY: click target {len(clicks) + 1}/4 ({next_corner})"
                detail = geometry_message or "Order: top-left, top-right, bottom-right, bottom-left"
            elif collecting_wall_depth:
                status = (
                    f"WALL DEPTH: {len(wall_depth_samples)}/"
                    f"{args.depth_calibration_frames}"
                )
                detail = depth_calibration_message or "Keep people and objects out of the projected area"
            elif collecting_depth_touch:
                status = (
                    f"{'GUIDED TOUCH' if args.legacy_depth_blob else 'CORNER TOUCH'}: "
                    f"{depth_touch_target_index + 1}/"
                    f"{len(depth_touch_targets)}"
                )
                touch_progress = (
                    len(current_target_samples)
                    if args.legacy_depth_blob
                    else len(current_fingertip_points)
                )
                if touch_progress:
                    detail = (
                        f"Hold contact: {touch_progress}/"
                        f"{args.depth_touch_samples} samples"
                    )
                elif not args.legacy_depth_blob and landmarks is None:
                    detail = "No hand detected: place your full hand inside the tracking box"
                elif not args.legacy_depth_blob and not extended:
                    detail = "Point with one straight index finger"
                elif not args.legacy_depth_blob and gap_mm is None:
                    detail = (
                        "No depth at fingertip: re-aim the camera so this target "
                        "has valid depth"
                    )
                else:
                    detail = (
                        "Center an open hand on the target and press it against the wall"
                        if args.legacy_depth_blob
                        else "Touch and hold the corner target with your index fingertip"
                    )
            elif collecting_touch:
                status = f"TOUCH CALIBRATION: {len(touch_samples)}/{args.touch_samples}"
                target = np.array([args.projector_width / 2, args.projector_height / 2], dtype=np.float32)
                if landmarks is None:
                    detail = "No hand detected: keep your full hand inside the blue tracking box"
                elif args.require_index_extension and not extended:
                    detail = "Point with one straight index finger"
                elif mapped_tip is None or np.linalg.norm(mapped_tip - target) >= min(output_size) * 0.16:
                    detail = "Move the fingertip onto the projected center target"
                else:
                    detail = "Hold still: samples are being collected automatically"
            elif not depth_enabled and touch_reference is None:
                status = "TOUCH CALIBRATION NEEDED"
                detail = "Touch and hold the projected center target"
            elif depth_enabled and depth_tracker is None:
                status = "WALL DEPTH CALIBRATION NEEDED"
                detail = "Press t and keep the projected wall clear"
            elif depth_enabled and not args.legacy_depth_blob and landmarks is None:
                status = "READY"
                detail = "No hand detected: point your index finger at the projected wall"
            else:
                status = decision.reason.upper()
                if depth_enabled:
                    gap_text = "--" if decision.distance_mm is None else f"{decision.distance_mm:.0f} mm"
                    detail = f"{interaction_mode} | wall gap {gap_text} | [ ]/m modes | c clear | r points | q quit"
                else:
                    ratio_text = "--" if decision.ratio is None else f"{decision.ratio:.2f}"
                    detail = f"{interaction_mode} | wall {ratio_text} | [ ]/m modes | c clear | r points | q quit"

            draw_outlined_text(debug, status, (14, 30), 0.76)
            draw_outlined_text(debug, detail, (14, 58), 0.56)
            fps = float(np.mean(fps_history)) if fps_history else 0.0
            draw_outlined_text(debug, f"{identity} | {fps:.1f} fps", (14, frame_height - 18), 0.52)

            cv2.imshow(projector_window, projector_frame)
            cv2.imshow(debug_window, debug)
            key = cv2.waitKeyEx(1)
            if key in (27, ord("q")):
                break
            if key == ord("c"):
                canvas = base_canvas.copy()
                for effect in reactive_effects:
                    effect.clear()
                print("Artwork cleared; calibration kept.")
            elif key in MODE_KEYS:
                interaction_mode = MODE_KEYS[key]
                print(f"Interaction mode: {interaction_mode}")
            elif key in (ord("m"), ord("]")):
                current_index = MODE_ORDER.index(interaction_mode)
                interaction_mode = MODE_ORDER[(current_index + 1) % len(MODE_ORDER)]
                print(f"Interaction mode: {interaction_mode}")
            elif key == ord("["):
                current_index = MODE_ORDER.index(interaction_mode)
                interaction_mode = MODE_ORDER[(current_index - 1) % len(MODE_ORDER)]
                print(f"Interaction mode: {interaction_mode}")
            elif key == ord("t") and matrix is not None:
                if depth_enabled and collecting_wall_depth:
                    print("Wall depth calibration is already running; keep the projection clear.")
                elif depth_enabled:
                    wall_depth_samples.clear()
                    wall_depth_frames.clear()
                    collecting_wall_depth = True
                    wall_depth_model = None
                    wall_depth_reference = None
                    wall_depth_noise = None
                    depth_touch_profile = None
                    spatial_touch_calibration = None
                    depth_tracker = None
                    depth_lock = None
                    collecting_depth_touch = False
                    guided_touch_samples.clear()
                    current_target_samples.clear()
                    fingertip_touch_points.clear()
                    fingertip_touch_depths.clear()
                    current_fingertip_points.clear()
                    current_fingertip_depths.clear()
                    depth_touch_target_index = 0
                    smoothed_gap_mm = None
                    depth_gate.reset()
                    save_calibration(
                        args.calibration,
                        identity,
                        frame_size,
                        output_size,
                        camera_points,
                        output_points,
                        None,
                        None,
                        sensor_mode,
                    )
                    print("Wall depth calibration started. Keep the projected area clear.")
                elif collecting_touch:
                    print("Touch calibration is already running; do not press t again.")
                else:
                    touch_samples.clear()
                    collecting_touch = True
                    touch_reference = None
                    smoothed_scale = None
                    gate.set_reference(None)
                    print("Touch calibration started. Press t only once, then hold the center target.")
            elif key == ord("r"):
                clicks.clear()
                camera_points = None
                matrix = None
                detection_roi = None
                touch_reference = None
                wall_depth_model = None
                wall_depth_reference = None
                wall_depth_noise = None
                depth_touch_profile = None
                spatial_touch_calibration = None
                depth_tracker = None
                depth_lock = None
                touch_samples.clear()
                wall_depth_samples.clear()
                wall_depth_frames.clear()
                guided_touch_samples.clear()
                current_target_samples.clear()
                fingertip_touch_points.clear()
                fingertip_touch_depths.clear()
                current_fingertip_points.clear()
                current_fingertip_depths.clear()
                depth_touch_target_index = 0
                collecting_touch = False
                collecting_wall_depth = False
                collecting_depth_touch = False
                smoothed_tip = None
                smoothed_scale = None
                smoothed_gap_mm = None
                gate.set_reference(None)
                depth_gate.reset()
                geometry_message = ""
                depth_calibration_message = ""
                canvas = base_canvas.copy()
                for effect in reactive_effects:
                    effect.clear()
                args.calibration.unlink(missing_ok=True)
                depth_reference_path(args.calibration).unlink(missing_ok=True)
                print("Artwork and calibration reset. Click the four projected targets.")
            elif key == ord("f"):
                fullscreen = not fullscreen
                mode = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(projector_window, cv2.WND_PROP_FULLSCREEN, mode)
                if not fullscreen:
                    cv2.resizeWindow(projector_window, args.projector_width, args.projector_height)
                    cv2.moveWindow(projector_window, args.projector_x, args.projector_y)
    finally:
        if landmarker is not None:
            landmarker.close()
        if landmarker_alt is not None:
            landmarker_alt.close()
        if depth_camera is not None:
            depth_camera.release()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
