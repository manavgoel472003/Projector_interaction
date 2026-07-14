from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


CORNER_NAMES = ("top-left", "top-right", "bottom-right", "bottom-left")


def projector_targets(width: int, height: int, inset_ratio: float = 0.065) -> np.ndarray:
    inset = max(48, int(min(width, height) * inset_ratio))
    return np.array(
        [
            [inset, inset],
            [width - 1 - inset, inset],
            [width - 1 - inset, height - 1 - inset],
            [inset, height - 1 - inset],
        ],
        dtype=np.float32,
    )


def build_homography(camera_points: np.ndarray, output_points: np.ndarray) -> np.ndarray:
    camera_points = np.asarray(camera_points, dtype=np.float32)
    output_points = np.asarray(output_points, dtype=np.float32)
    if camera_points.shape != (4, 2) or output_points.shape != (4, 2):
        raise ValueError("A homography requires four 2D camera and output points")
    matrix = cv2.getPerspectiveTransform(camera_points, output_points)
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-10:
        raise ValueError("Calibration points produced an invalid homography")
    return matrix


def validate_camera_quad(
    camera_points: np.ndarray,
    frame_size: tuple[int, int],
    minimum_area_ratio: float = 0.0125,
) -> None:
    points = np.asarray(camera_points, dtype=np.float32)
    if points.shape != (4, 2):
        raise ValueError("Click all four projected targets")
    contour = np.rint(points).astype(np.int32)
    if not cv2.isContourConvex(contour):
        raise ValueError("Points cross or are out of order; use top-left, top-right, bottom-right, bottom-left")
    width, height = frame_size
    area_ratio = abs(cv2.contourArea(points)) / max(width * height, 1)
    if area_ratio < minimum_area_ratio:
        raise ValueError("Selected projection is too small in the camera view; aim/zoom the camera closer")
    if np.any(points[:, 0] < 0) or np.any(points[:, 0] >= width):
        raise ValueError("A calibration point is outside the camera frame")
    if np.any(points[:, 1] < 0) or np.any(points[:, 1] >= height):
        raise ValueError("A calibration point is outside the camera frame")


def camera_detection_roi(
    camera_points: np.ndarray,
    frame_size: tuple[int, int],
    margin_ratio: float = 1.15,
) -> tuple[int, int, int, int]:
    points = np.asarray(camera_points, dtype=np.float32)
    width, height = frame_size
    span_x = float(np.ptp(points[:, 0]))
    span_y = float(np.ptp(points[:, 1]))
    margin = max(64, int(max(span_x, span_y) * margin_ratio))
    x0 = max(0, int(np.floor(points[:, 0].min())) - margin)
    y0 = max(0, int(np.floor(points[:, 1].min())) - margin)
    x1 = min(width, int(np.ceil(points[:, 0].max())) + margin + 1)
    y1 = min(height, int(np.ceil(points[:, 1].max())) + margin + 1)
    if x1 - x0 < 32 or y1 - y0 < 32:
        raise ValueError("Projection tracking region is too small")
    return x0, y0, x1, y1


def projection_near_frame_edge(
    camera_points: np.ndarray,
    frame_size: tuple[int, int],
    minimum_margin_ratio: float = 0.07,
) -> bool:
    points = np.asarray(camera_points, dtype=np.float32)
    width, height = frame_size
    margin_x = width * minimum_margin_ratio
    margin_y = height * minimum_margin_ratio
    return bool(
        points[:, 0].min() < margin_x
        or points[:, 0].max() > width - margin_x
        or points[:, 1].min() < margin_y
        or points[:, 1].max() > height - margin_y
    )


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Points must have shape (N, 2)")
    return cv2.perspectiveTransform(points.reshape(1, -1, 2), matrix)[0]


def point_in_output(point: np.ndarray, width: int, height: int, margin: int = 0) -> bool:
    x, y = np.asarray(point, dtype=float)
    return margin <= x < width - margin and margin <= y < height - margin


def hand_plane_scale(mapped_landmarks: np.ndarray) -> float:
    """Return a stable apparent hand size after mapping into projector coordinates."""
    points = np.asarray(mapped_landmarks, dtype=np.float32)
    if points.shape != (21, 2):
        raise ValueError("Expected 21 mapped hand landmarks")

    palm_width = np.linalg.norm(points[5] - points[17])
    palm_length = np.linalg.norm(points[0] - points[9])
    index_base = np.linalg.norm(points[5] - points[9])
    return float(np.median((palm_width, palm_length * 1.15, index_base * 2.1)))


def index_is_extended(camera_landmarks: np.ndarray) -> bool:
    points = np.asarray(camera_landmarks, dtype=np.float32)
    if points.shape != (21, 2):
        return False

    wrist = points[0]
    mcp = points[5]
    pip = points[6]
    tip = points[8]
    tip_distance = np.linalg.norm(tip - wrist)
    pip_distance = np.linalg.norm(pip - wrist)
    mcp_distance = np.linalg.norm(mcp - wrist)
    return bool(
        tip_distance > max(1.16 * pip_distance, 1.42 * mcp_distance)
        and np.linalg.norm(tip - pip) > 0.32 * max(pip_distance, 1.0)
    )


@dataclass(frozen=True)
class TouchDecision:
    active: bool
    candidate: bool
    ratio: float | None
    reason: str


class TouchGate:
    def __init__(
        self,
        reference_scale: float | None = None,
        minimum_ratio: float = 0.50,
        maximum_ratio: float = 1.60,
        dwell_seconds: float = 0.05,
        maximum_dwell_motion: float = 70.0,
        tracking_grace_seconds: float = 0.22,
    ) -> None:
        self.reference_scale = reference_scale
        self.minimum_ratio = minimum_ratio
        self.maximum_ratio = maximum_ratio
        self.dwell_seconds = dwell_seconds
        self.maximum_dwell_motion = maximum_dwell_motion
        self.tracking_grace_seconds = tracking_grace_seconds
        self._candidate_since: float | None = None
        self._candidate_origin: np.ndarray | None = None
        self._last_valid_time: float | None = None
        self._active = False

    def reset(self) -> None:
        self._candidate_since = None
        self._candidate_origin = None
        self._last_valid_time = None
        self._active = False

    def set_reference(self, scale: float | None) -> None:
        self.reference_scale = scale
        self.reset()

    def update(
        self,
        *,
        scale: float | None,
        point: np.ndarray | None,
        timestamp: float,
        inside: bool,
        index_extended: bool,
    ) -> TouchDecision:
        if self.reference_scale is None or self.reference_scale <= 0:
            self.reset()
            return TouchDecision(False, False, None, "calibrate touch")
        if scale is None or point is None:
            return self._temporary_loss(timestamp, None, "brief tracking loss")

        ratio = scale / self.reference_scale
        if not inside:
            self.reset()
            return TouchDecision(False, False, ratio, "outside projection")
        if not index_extended:
            return self._temporary_loss(timestamp, ratio, "finger pose uncertain")
        if ratio > self.maximum_ratio:
            return self._temporary_loss(timestamp, ratio, "hand too close to camera")
        if ratio < self.minimum_ratio:
            return self._temporary_loss(timestamp, ratio, "hand beyond wall estimate")

        point = np.asarray(point, dtype=np.float32)
        self._last_valid_time = timestamp
        if self._active:
            return TouchDecision(True, True, ratio, "touch")

        if self._candidate_since is None:
            self._candidate_since = timestamp
            self._candidate_origin = point.copy()
            return TouchDecision(False, True, ratio, "hold briefly")

        motion = float(np.linalg.norm(point - self._candidate_origin))
        if motion > self.maximum_dwell_motion:
            self._candidate_since = timestamp
            self._candidate_origin = point.copy()
            return TouchDecision(False, True, ratio, "steady fingertip")

        if timestamp - self._candidate_since >= self.dwell_seconds:
            self._active = True
            return TouchDecision(True, True, ratio, "touch")
        return TouchDecision(False, True, ratio, "hold briefly")

    def _temporary_loss(
        self,
        timestamp: float,
        ratio: float | None,
        reason: str,
    ) -> TouchDecision:
        has_progress = self._active or self._candidate_since is not None
        within_grace = (
            self._last_valid_time is not None
            and timestamp - self._last_valid_time <= self.tracking_grace_seconds
        )
        if has_progress and within_grace:
            return TouchDecision(False, True, ratio, reason)
        self.reset()
        return TouchDecision(False, False, ratio, reason)
