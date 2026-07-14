from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import mediapipe as mp
import numpy as np

from wall_touch_ambient_effects import ConstellationField, MagneticSand
from wall_touch_core import (
    CORNER_NAMES,
    TouchGate,
    build_homography,
    camera_detection_roi,
    hand_plane_scale,
    index_is_extended,
    point_in_output,
    projection_near_frame_edge,
    projector_targets,
    transform_points,
    validate_camera_quad,
)
from wall_touch_effects import PulseGrid, WatercolorPool


ROOT = Path(__file__).resolve().parent
APP_VERSION = "2.7"
DEFAULT_CAMERA = "auto"
DEFAULT_MODEL = ROOT / "models/hand_landmarker.task"
DEFAULT_CALIBRATION = ROOT / "wall_touch_calibration.json"
BLOCKED_CAMERA_NAMES = ("usb2.0 fhd uvc webcam", "shinetech")
MODE_ORDER = (
    "paint",
    "spill",
    "ripple",
    "pulse",
    "constellation",
    "sand",
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
        description="Calibrated projector touch-paint demo using one external RGB camera."
    )
    parser.add_argument(
        "--camera",
        default=os.environ.get("WALL_TOUCH_CAMERA", DEFAULT_CAMERA),
        help="External V4L2 device path, or 'auto' to discover one (default).",
    )
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--projector-width", type=int, default=1920)
    parser.add_argument("--projector-height", type=int, default=1200)
    parser.add_argument("--projector-x", type=int, default=0)
    parser.add_argument("--projector-y", type=int, default=0)
    parser.add_argument("--debug-x", type=int, default=1980)
    parser.add_argument("--debug-y", type=int, default=60)
    parser.add_argument("--windowed", action="store_true", help="Do not fullscreen the projector output.")
    parser.add_argument("--fresh", action="store_true", help="Ignore saved geometry and touch calibration.")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--touch-samples", type=int, default=24)
    parser.add_argument("--touch-scale-min", type=float, default=0.50)
    parser.add_argument("--touch-scale-max", type=float, default=1.60)
    parser.add_argument("--touch-dwell-ms", type=int, default=50)
    parser.add_argument("--require-index-extension", action="store_true")
    parser.add_argument("--brush-radius", type=int, default=46)
    parser.add_argument("--paint-alpha", type=float, default=0.46)
    parser.add_argument("--mode", choices=MODE_ORDER, default="spill")
    parser.add_argument("--detection-confidence", type=float, default=0.40)
    return parser.parse_args()


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


def open_camera(path: str, width: int, height: int, fps: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open external camera {path}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
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
) -> None:
    data = {
        "version": 1,
        "camera_identity": camera_identity,
        "camera_frame_size": list(frame_size),
        "projector_output_size": list(output_size),
        "camera_points": np.asarray(camera_points, dtype=float).tolist(),
        "output_points": np.asarray(output_points, dtype=float).tolist(),
        "corner_order": list(CORNER_NAMES),
        "touch_reference_scale": touch_reference_scale,
    }
    path.write_text(json.dumps(data, indent=2) + "\n")


def load_calibration(
    path: Path,
    camera_identity: str,
    frame_size: tuple[int, int],
    output_size: tuple[int, int],
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
        data["camera_points"] = np.array(data["camera_points"], dtype=np.float32)
        data["output_points"] = np.array(data["output_points"], dtype=np.float32)
        return data
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
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


def main() -> None:
    args = parse_args()
    camera_path, identity = validate_camera(args.camera)
    print(f"Wall Touch Demo v{APP_VERSION}")
    print(f"External camera: {identity} ({Path(camera_path).resolve()})")
    print(
        f"Projector: {args.projector_width}x{args.projector_height} "
        f"at desktop ({args.projector_x},{args.projector_y})"
    )

    cap = open_camera(camera_path, args.camera_width, args.camera_height, args.camera_fps)
    try:
        frame = read_camera_frame(cap)
    except RuntimeError:
        cap.release()
        raise
    frame_height, frame_width = frame.shape[:2]
    frame_size = (frame_width, frame_height)
    output_size = (args.projector_width, args.projector_height)
    print(f"Camera stream: {frame_width}x{frame_height}")

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
        args.calibration, identity, frame_size, output_size
    )
    camera_points = saved["camera_points"] if saved else None
    output_points = saved["output_points"] if saved else output_points
    touch_reference = saved.get("touch_reference_scale") if saved else None
    matrix = build_homography(camera_points, output_points) if camera_points is not None else None
    detection_roi = camera_detection_roi(camera_points, frame_size) if camera_points is not None else None
    if saved:
        print(f"Loaded calibration: {args.calibration}")
        print("If the camera or projector moved, press r and recalibrate.")

    clicks: list[list[float]] = []
    touch_samples: list[float] = []
    collecting_touch = matrix is not None and touch_reference is None
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
    reactive_effects = (
        spill,
        ripple,
        pulse,
        constellation,
        sand,
    )
    interaction_mode = args.mode
    gate = TouchGate(
        reference_scale=touch_reference,
        minimum_ratio=args.touch_scale_min,
        maximum_ratio=args.touch_scale_max,
        dwell_seconds=args.touch_dwell_ms / 1000.0,
    )
    smoothed_tip: np.ndarray | None = None
    smoothed_scale: float | None = None
    last_timestamp_ms = 0
    last_frame_time = time.monotonic()
    fps_history: deque[float] = deque(maxlen=30)
    fullscreen = not args.windowed
    geometry_message = ""
    last_ripple_time = -1e9

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN and matrix is None and len(clicks) < 4:
            clicks.append([float(x), float(y)])
            print(f"Geometry {CORNER_NAMES[len(clicks) - 1]}: {x}, {y}")

    cv2.setMouseCallback(debug_window, on_mouse)
    landmarker = create_landmarker(args.model, args.detection_confidence)

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
                    touch_samples.clear()
                    collecting_touch = True
                    smoothed_scale = None
                    geometry_message = ""
                    gate.set_reference(None)
                    save_calibration(
                        args.calibration, identity, frame_size, output_size,
                        camera_points, output_points, touch_reference,
                    )
                    area_percent = 100 * abs(cv2.contourArea(camera_points)) / (frame_width * frame_height)
                    print(f"Geometry accepted ({area_percent:.2f}% of camera frame).")
                    print("Walk to the wall and touch the labeled center target; sampling starts automatically.")

            frame = read_camera_frame(cap, timeout_seconds=0.8)
            now = time.monotonic()
            delta = now - last_frame_time
            last_frame_time = now
            if delta > 0:
                fps_history.append(1.0 / delta)

            debug = frame.copy()
            landmarks = None
            if detection_roi is not None:
                x0, y0, x1, y1 = detection_roi
                tracking_frame = frame[y0:y1, x0:x1]
                rgb = cv2.cvtColor(tracking_frame, cv2.COLOR_BGR2RGB)
                timestamp_ms = max(last_timestamp_ms + 1, int(now * 1000))
                last_timestamp_ms = timestamp_ms
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)),
                    timestamp_ms,
                )
                landmarks = camera_landmarks(result, tracking_frame.shape, (x0, y0))
                cv2.rectangle(debug, (x0, y0), (x1 - 1, y1 - 1), (210, 150, 40), 1)
            mapped_landmarks = None
            mapped_tip = None
            scale = None
            extended = False
            inside = False
            if landmarks is not None and matrix is not None:
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
            elif landmarks is None:
                smoothed_tip = None

            if matrix is None:
                decision = gate.update(
                    scale=None, point=None, timestamp=now, inside=False, index_extended=False
                )
            else:
                decision = gate.update(
                    scale=scale,
                    point=mapped_tip,
                    timestamp=now,
                    inside=inside,
                    index_extended=extended or not args.require_index_extension,
                )

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
                    )
                    print(f"Touch plane learned: reference hand scale={touch_reference:.1f}")

            if decision.active and mapped_tip is not None:
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
            else:
                art_frame = canvas.copy()

            if matrix is None:
                projector_frame = np.empty_like(canvas)
                draw_projector_targets(projector_frame, output_points, min(len(clicks), 3))
            else:
                projector_frame = art_frame
                if touch_reference is None or collecting_touch:
                    center = np.array([args.projector_width / 2, args.projector_height / 2])
                    draw_touch_target(
                        projector_frame, center,
                        len(touch_samples) if collecting_touch else 0,
                        args.touch_samples if collecting_touch else 0,
                    )
                elif mapped_tip is not None and inside:
                    point = tuple(np.rint(mapped_tip).astype(int))
                    cursor_color = (70, 220, 90) if decision.active else (30, 190, 255)
                    cv2.circle(projector_frame, point, args.brush_radius + 8, cursor_color, 5, cv2.LINE_AA)

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

            if matrix is None:
                next_corner = CORNER_NAMES[len(clicks)] if len(clicks) < 4 else "processing"
                status = f"GEOMETRY: click target {len(clicks) + 1}/4 ({next_corner})"
                detail = geometry_message or "Order: top-left, top-right, bottom-right, bottom-left"
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
            elif touch_reference is None:
                status = "TOUCH CALIBRATION NEEDED"
                detail = "Touch and hold the projected center target"
            else:
                status = decision.reason.upper()
                ratio_text = "--" if decision.ratio is None else f"{decision.ratio:.2f}"
                coordinate_text = "--" if mapped_tip is None else f"{mapped_tip[0]:.0f},{mapped_tip[1]:.0f}"
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
                if collecting_touch:
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
                touch_samples.clear()
                collecting_touch = False
                smoothed_tip = None
                smoothed_scale = None
                gate.set_reference(None)
                geometry_message = ""
                canvas = base_canvas.copy()
                for effect in reactive_effects:
                    effect.clear()
                args.calibration.unlink(missing_ok=True)
                print("Artwork and calibration reset. Click the four projected targets.")
            elif key == ord("f"):
                fullscreen = not fullscreen
                mode = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(projector_window, cv2.WND_PROP_FULLSCREEN, mode)
                if not fullscreen:
                    cv2.resizeWindow(projector_window, args.projector_width, args.projector_height)
                    cv2.moveWindow(projector_window, args.projector_x, args.projector_y)
    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
