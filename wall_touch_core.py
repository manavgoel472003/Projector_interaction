from __future__ import annotations

from collections import deque
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
    distance_mm: float | None = None


@dataclass(frozen=True)
class WallDepthModel:
    coefficients: np.ndarray
    rmse_mm: float
    sample_count: int

    def expected_depth(self, point: np.ndarray, frame_size: tuple[int, int]) -> float:
        width, height = frame_size
        x, y = np.asarray(point, dtype=np.float64)
        normalized = np.array(
            [(x - (width - 1) * 0.5) / width, (y - (height - 1) * 0.5) / height, 1.0]
        )
        inverse_depth = float(normalized @ np.asarray(self.coefficients, dtype=np.float64))
        return 1.0 / inverse_depth if inverse_depth > 0 else float("nan")

    def to_dict(self) -> dict[str, object]:
        return {
            "coefficients": np.asarray(self.coefficients, dtype=float).tolist(),
            "rmse_mm": float(self.rmse_mm),
            "sample_count": int(self.sample_count),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "WallDepthModel":
        coefficients = np.asarray(data["coefficients"], dtype=np.float64)
        if coefficients.shape != (3,) or not np.isfinite(coefficients).all():
            raise ValueError("Invalid wall depth coefficients")
        return cls(coefficients, float(data["rmse_mm"]), int(data["sample_count"]))


@dataclass(frozen=True)
class DepthContactObservation:
    camera_point: np.ndarray
    gap_mm: float
    component_area: int


@dataclass(frozen=True)
class DepthTouchProfile:
    minimum_gap_mm: float
    maximum_gap_mm: float
    minimum_component_area: int
    maximum_component_area: int
    sample_count: int

    @classmethod
    def fit(cls, samples: list[DepthContactObservation]) -> "DepthTouchProfile":
        if len(samples) < 12:
            raise ValueError("Not enough guided touch samples")
        gaps = np.asarray([sample.gap_mm for sample in samples], dtype=np.float32)
        areas = np.asarray([sample.component_area for sample in samples], dtype=np.float32)
        return cls(
            minimum_gap_mm=max(5.0, float(np.percentile(gaps, 5)) - 10.0),
            maximum_gap_mm=float(np.percentile(gaps, 95)) + 10.0,
            minimum_component_area=max(60, int(np.percentile(areas, 10) * 0.45)),
            maximum_component_area=max(200, int(np.percentile(areas, 90) * 2.5)),
            sample_count=len(samples),
        )

    def accepts(self, observation: DepthContactObservation) -> bool:
        return bool(
            self.minimum_gap_mm <= observation.gap_mm <= self.maximum_gap_mm
            and self.minimum_component_area
            <= observation.component_area
            <= self.maximum_component_area
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "minimum_gap_mm": self.minimum_gap_mm,
            "maximum_gap_mm": self.maximum_gap_mm,
            "minimum_component_area": self.minimum_component_area,
            "maximum_component_area": self.maximum_component_area,
            "sample_count": self.sample_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "DepthTouchProfile":
        profile = cls(
            minimum_gap_mm=float(data["minimum_gap_mm"]),
            maximum_gap_mm=float(data["maximum_gap_mm"]),
            minimum_component_area=int(data["minimum_component_area"]),
            maximum_component_area=int(data["maximum_component_area"]),
            sample_count=int(data["sample_count"]),
        )
        if (
            profile.minimum_gap_mm < 0
            or profile.maximum_gap_mm <= profile.minimum_gap_mm
            or profile.minimum_component_area <= 0
            or profile.maximum_component_area <= profile.minimum_component_area
        ):
            raise ValueError("Invalid depth touch profile")
        return profile


class DepthContactLock:
    def __init__(
        self,
        profile: DepthTouchProfile,
        acquisition_frames: int = 3,
        maximum_motion_pixels: float = 18.0,
        loss_grace_frames: int = 2,
    ) -> None:
        self.profile = profile
        self.acquisition_frames = max(1, int(acquisition_frames))
        self.maximum_motion_pixels = float(maximum_motion_pixels)
        self.loss_grace_frames = max(0, int(loss_grace_frames))
        self.reset()

    def reset(self) -> None:
        self._point: np.ndarray | None = None
        self._consecutive = 0
        self._missing = 0

    def update(
        self, observations: list[DepthContactObservation]
    ) -> DepthContactObservation | None:
        accepted = [item for item in observations if self.profile.accepts(item)]
        if not accepted:
            self._missing += 1
            if self._missing > self.loss_grace_frames:
                self.reset()
            return None

        if self._point is None:
            selected = max(accepted, key=lambda item: item.component_area)
            self._point = selected.camera_point.copy()
            self._consecutive = 1
        else:
            selected = min(
                accepted,
                key=lambda item: float(np.linalg.norm(item.camera_point - self._point)),
            )
            motion = float(np.linalg.norm(selected.camera_point - self._point))
            if motion > self.maximum_motion_pixels:
                self._point = selected.camera_point.copy()
                self._consecutive = 1
            else:
                self._point = 0.45 * self._point + 0.55 * selected.camera_point
                self._consecutive += 1
        self._missing = 0
        if self._consecutive < self.acquisition_frames:
            return None
        return DepthContactObservation(
            self._point.copy(), selected.gap_mm, selected.component_area
        )


def depth_target_has_foreground(
    wall_depth_mm: np.ndarray,
    current_depth_mm: np.ndarray,
    wall_noise_mm: np.ndarray,
    target: np.ndarray,
    radius: int = 28,
) -> bool:
    present, _, _, _ = depth_target_foreground_metrics(
        wall_depth_mm, current_depth_mm, wall_noise_mm, target, radius
    )
    return present


def depth_target_foreground_metrics(
    wall_depth_mm: np.ndarray,
    current_depth_mm: np.ndarray,
    wall_noise_mm: np.ndarray,
    target: np.ndarray,
    radius: int = 28,
) -> tuple[bool, float, int, int]:
    reference = np.asarray(wall_depth_mm, dtype=np.float32)
    current = np.asarray(current_depth_mm, dtype=np.float32)
    noise = np.asarray(wall_noise_mm, dtype=np.float32)
    if reference.shape != current.shape or reference.shape != noise.shape:
        return False, 0.0, 0, 0
    x, y = np.rint(np.asarray(target, dtype=np.float32)).astype(int)
    height, width = reference.shape
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    circle = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
    ref_patch = reference[y0:y1, x0:x1]
    current_patch = current[y0:y1, x0:x1]
    noise_patch = noise[y0:y1, x0:x1]
    valid = circle & (ref_patch >= 100) & (current_patch >= 100)
    valid_count = int(np.count_nonzero(valid))
    if valid_count < 40:
        return False, 0.0, 0, valid_count
    gap = ref_patch - current_patch
    strong_threshold = np.maximum(55.0, 1.35 * noise_patch + 12.0)
    strong_count = int(np.count_nonzero(valid & (gap >= strong_threshold)))
    gap_p90 = float(np.percentile(gap[valid], 90))
    return strong_count >= 18, gap_p90, strong_count, valid_count


class DepthContactTracker:
    """Locate the part of a foreground depth component nearest the wall."""

    def __init__(
        self,
        wall_depth_mm: np.ndarray,
        camera_points: np.ndarray,
        wall_noise_mm: np.ndarray | None = None,
        minimum_change_mm: float = 15.0,
        maximum_change_mm: float = 800.0,
        minimum_component_area: int = 80,
        near_wall_limit_mm: float = 60.0,
        noise_multiplier: float = 0.75,
        temporal_frames: int = 3,
    ) -> None:
        reference = np.asarray(wall_depth_mm, dtype=np.float32)
        if reference.ndim != 2:
            raise ValueError("Wall depth reference must be a 2D array")
        points = np.asarray(camera_points, dtype=np.float32)
        if points.shape != (4, 2):
            raise ValueError("Expected four projection corners")
        if minimum_change_mm <= 0 or maximum_change_mm <= minimum_change_mm:
            raise ValueError("Invalid depth-change range")

        self.wall_depth_mm = reference.copy()
        if wall_noise_mm is None:
            self.wall_noise_mm = np.zeros_like(reference)
        else:
            noise = np.asarray(wall_noise_mm, dtype=np.float32)
            if noise.shape != reference.shape:
                raise ValueError("Wall noise map must match the depth reference")
            self.wall_noise_mm = np.nan_to_num(
                noise, nan=maximum_change_mm, posinf=maximum_change_mm
            )
        self.minimum_change_mm = float(minimum_change_mm)
        self.maximum_change_mm = float(maximum_change_mm)
        self.minimum_component_area = int(minimum_component_area)
        self.near_wall_limit_mm = float(near_wall_limit_mm)
        self.noise_multiplier = float(noise_multiplier)
        self._history: deque[np.ndarray] = deque(maxlen=max(1, int(temporal_frames)))
        self.projection_mask = np.zeros(reference.shape, dtype=np.uint8)
        cv2.fillConvexPoly(
            self.projection_mask,
            np.rint(points).astype(np.int32),
            255,
        )
        self._open_kernel = np.ones((3, 3), dtype=np.uint8)
        self._close_kernel = np.ones((5, 5), dtype=np.uint8)
        self.current_depth_mm: np.ndarray | None = None

    def observations(self, depth_mm: np.ndarray) -> list[DepthContactObservation]:
        depth = np.asarray(depth_mm, dtype=np.float32)
        if depth.shape != self.wall_depth_mm.shape:
            return []

        self._history.append(depth.copy())
        history = np.stack(self._history)
        current = np.ma.median(
            np.ma.masked_less(history, 100.0), axis=0
        ).filled(0.0).astype(np.float32)
        self.current_depth_mm = current

        reference = self.wall_depth_mm
        valid = (
            (self.projection_mask > 0)
            & np.isfinite(reference)
            & (reference >= 100.0)
            & np.isfinite(current)
            & (current >= 100.0)
        )
        gap = reference - current
        change_threshold = np.maximum(
            self.minimum_change_mm,
            self.noise_multiplier * self.wall_noise_mm + 8.0,
        )
        foreground = (
            valid
            & (gap >= change_threshold)
            & (gap <= self.maximum_change_mm)
        ).astype(np.uint8) * 255
        foreground = cv2.morphologyEx(
            foreground, cv2.MORPH_OPEN, self._open_kernel
        )
        foreground = cv2.morphologyEx(
            foreground, cv2.MORPH_CLOSE, self._close_kernel
        )

        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            foreground, connectivity=8
        )
        observations: list[DepthContactObservation] = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.minimum_component_area:
                continue
            ys, xs = np.nonzero(labels == label)
            values = gap[ys, xs]
            near_percentile = float(np.percentile(values, 10.0))
            location_limit = min(
                float(np.percentile(values, 20.0)),
                near_percentile + 8.0,
            )
            near = values <= location_limit
            if np.count_nonzero(near) < 4:
                continue
            weights = np.maximum(location_limit - values[near] + 1.0, 1.0)
            point = np.array(
                [
                    np.average(xs[near], weights=weights),
                    np.average(ys[near], weights=weights),
                ],
                dtype=np.float32,
            )
            observation = DepthContactObservation(point, near_percentile, area)
            observations.append(observation)

        return observations

    def detect(self, depth_mm: np.ndarray) -> DepthContactObservation | None:
        observations = self.observations(depth_mm)
        if not observations:
            return None
        return min(
            observations,
            key=lambda item: (
                item.gap_mm > self.near_wall_limit_mm,
                -item.component_area,
                item.gap_mm,
            ),
        )


def fit_wall_depth_model(
    depth_mm: np.ndarray,
    camera_points: np.ndarray,
    minimum_depth_mm: float = 150.0,
    maximum_depth_mm: float = 10_000.0,
    sample_step: int = 4,
) -> WallDepthModel:
    depth = np.asarray(depth_mm, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("Depth frame must be a 2D array")
    points = np.asarray(camera_points, dtype=np.float32)
    if points.shape != (4, 2):
        raise ValueError("Expected four projection corners")

    height, width = depth.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.rint(points).astype(np.int32), 1)
    yy, xx = np.mgrid[0:height:sample_step, 0:width:sample_step]
    values = depth[::sample_step, ::sample_step]
    valid = (
        (mask[::sample_step, ::sample_step] > 0)
        & np.isfinite(values)
        & (values >= minimum_depth_mm)
        & (values <= maximum_depth_mm)
    )
    if np.count_nonzero(valid) < 120:
        raise ValueError("Not enough valid wall depth pixels")

    x = (xx[valid].astype(np.float64) - (width - 1) * 0.5) / width
    y = (yy[valid].astype(np.float64) - (height - 1) * 0.5) / height
    measured = values[valid].astype(np.float64)
    design = np.column_stack((x, y, np.ones_like(x)))
    inliers = np.ones(len(measured), dtype=bool)
    coefficients = np.zeros(3, dtype=np.float64)
    for _ in range(4):
        coefficients = np.linalg.lstsq(
            design[inliers], 1.0 / measured[inliers], rcond=None
        )[0]
        inverse = design @ coefficients
        predicted = np.divide(
            1.0,
            inverse,
            out=np.full_like(inverse, np.nan),
            where=inverse > 0,
        )
        residual = measured - predicted
        finite = np.isfinite(residual)
        median = float(np.median(residual[finite]))
        mad = float(np.median(np.abs(residual[finite] - median)))
        threshold = min(60.0, max(12.0, 3.5 * 1.4826 * mad))
        next_inliers = finite & (np.abs(residual - median) <= threshold)
        if np.count_nonzero(next_inliers) < 120:
            break
        inliers = next_inliers

    coefficients = np.linalg.lstsq(
        design[inliers], 1.0 / measured[inliers], rcond=None
    )[0]
    predicted = 1.0 / (design[inliers] @ coefficients)
    rmse = float(np.sqrt(np.mean((measured[inliers] - predicted) ** 2)))
    return WallDepthModel(coefficients, rmse, int(np.count_nonzero(inliers)))


def combine_wall_depth_models(models: list[WallDepthModel]) -> WallDepthModel:
    if not models:
        raise ValueError("No wall depth models were collected")
    coefficients = np.median(
        np.stack([np.asarray(model.coefficients) for model in models]), axis=0
    )
    return WallDepthModel(
        coefficients=coefficients,
        rmse_mm=float(np.median([model.rmse_mm for model in models])),
        sample_count=int(sum(model.sample_count for model in models)),
    )


def sample_fingertip_depth(
    depth_mm: np.ndarray,
    point: np.ndarray,
    radius: int = 7,
    percentile: float = 25.0,
) -> float | None:
    depth = np.asarray(depth_mm, dtype=np.float32)
    if depth.ndim != 2:
        return None
    x, y = np.rint(np.asarray(point, dtype=float)).astype(int)
    height, width = depth.shape
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    values = depth[y0:y1, x0:x1]
    valid = values[np.isfinite(values) & (values >= 100) & (values <= 10_000)]
    if valid.size < max(4, radius):
        return None
    return float(np.percentile(valid, percentile))


class DepthTouchGate:
    def __init__(
        self,
        minimum_gap_mm: float = -15.0,
        maximum_gap_mm: float = 45.0,
        dwell_seconds: float = 0.05,
        maximum_dwell_motion: float = 70.0,
        tracking_grace_seconds: float = 0.22,
    ) -> None:
        self.minimum_gap_mm = minimum_gap_mm
        self.maximum_gap_mm = maximum_gap_mm
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

    def update(
        self,
        *,
        gap_mm: float | None,
        point: np.ndarray | None,
        timestamp: float,
        inside: bool,
        index_extended: bool,
        calibrated: bool = True,
    ) -> TouchDecision:
        if not calibrated:
            self.reset()
            return TouchDecision(False, False, None, "calibrate wall depth")
        if gap_mm is None or point is None:
            return self._temporary_loss(timestamp, gap_mm, "depth unavailable")
        if not inside:
            self.reset()
            return TouchDecision(False, False, None, "outside projection", gap_mm)
        if not index_extended:
            return self._temporary_loss(timestamp, gap_mm, "finger pose uncertain")
        if gap_mm > self.maximum_gap_mm:
            return self._temporary_loss(timestamp, gap_mm, "finger above wall")
        if gap_mm < self.minimum_gap_mm:
            return self._temporary_loss(timestamp, gap_mm, "depth behind wall")

        point = np.asarray(point, dtype=np.float32)
        self._last_valid_time = timestamp
        if self._active:
            return TouchDecision(True, True, None, "touch", gap_mm)
        if self._candidate_since is None:
            self._candidate_since = timestamp
            self._candidate_origin = point.copy()
            return TouchDecision(False, True, None, "hold briefly", gap_mm)
        if float(np.linalg.norm(point - self._candidate_origin)) > self.maximum_dwell_motion:
            self._candidate_since = timestamp
            self._candidate_origin = point.copy()
            return TouchDecision(False, True, None, "steady fingertip", gap_mm)
        if timestamp - self._candidate_since >= self.dwell_seconds:
            self._active = True
            return TouchDecision(True, True, None, "touch", gap_mm)
        return TouchDecision(False, True, None, "hold briefly", gap_mm)

    def _temporary_loss(
        self,
        timestamp: float,
        gap_mm: float | None,
        reason: str,
    ) -> TouchDecision:
        has_progress = self._active or self._candidate_since is not None
        within_grace = (
            self._last_valid_time is not None
            and timestamp - self._last_valid_time <= self.tracking_grace_seconds
        )
        if has_progress and within_grace:
            return TouchDecision(False, True, None, reason, gap_mm)
        self.reset()
        return TouchDecision(False, False, None, reason, gap_mm)


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
