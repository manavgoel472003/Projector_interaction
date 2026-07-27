from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError as error:  # pragma: no cover - exercised without the optional SDK
    rs = None
    SDK_IMPORT_ERROR: ImportError | None = error
else:
    SDK_IMPORT_ERROR = None


@dataclass(frozen=True)
class RGBDFrame:
    color_bgr: np.ndarray
    depth_mm: np.ndarray


def realsense_device_count() -> int:
    if rs is None:
        return 0
    try:
        return len(rs.context().query_devices())
    except Exception:
        return 0


def depth_values_to_mm(values: np.ndarray, depth_scale_m: float) -> np.ndarray:
    return np.asarray(values, dtype=np.float32) * float(depth_scale_m) * 1000.0


class RealSenseCamera:
    """Synchronized RealSense color and depth aligned in color-pixel coordinates."""

    def __init__(
        self,
        preferred_width: int = 1280,
        preferred_height: int = 720,
        fps: int = 30,
        depth_preset: str = "high-accuracy",
    ) -> None:
        if rs is None:
            raise RuntimeError(
                "RealSense SDK is not installed. Run ./install.sh or install pyrealsense2."
            ) from SDK_IMPORT_ERROR

        self.pipeline, self.profile, self.width, self.height, self.fps = self._start_streams(
            preferred_width, preferred_height, fps
        )
        self.closed = False
        device = self.profile.get_device()
        self.identity = " ".join(
            value
            for value in (
                self._device_info(device, rs.camera_info.name),
                self._device_info(device, rs.camera_info.serial_number),
            )
            if value
        )
        self.usb_type = self._device_info(device, rs.camera_info.usb_type_descriptor) or "USB"
        self.depth_sensor = device.first_depth_sensor()
        self.depth_scale_m = float(self.depth_sensor.get_depth_scale())
        self.accuracy_settings = self._configure_depth_sensor(
            self.depth_sensor, depth_preset
        )

        self.align = rs.align(rs.stream.color)
        self.to_disparity = rs.disparity_transform(True)
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.to_depth = rs.disparity_transform(False)
        self._configure_filters()

        # Discard startup frames while stereo auto-exposure converges. Tolerate
        # slow first frames -- when a second camera shares USB bandwidth the first
        # frames can lag well past 1.5 s, which should not abort startup.
        warmed = 0
        for _ in range(40):
            try:
                self.pipeline.wait_for_frames(2500)
            except RuntimeError:
                continue
            warmed += 1
            if warmed >= 10:
                break
        if warmed == 0:
            raise RuntimeError(
                "RealSense did not deliver any startup frames. If a second camera "
                "is attached, lower its resolution or move it to a separate USB "
                "controller to free bandwidth."
            )

    @staticmethod
    def _device_info(device: object, field: object) -> str:
        try:
            return str(device.get_info(field)) if device.supports(field) else ""
        except Exception:
            return ""

    @staticmethod
    def _candidate_profiles(
        preferred_width: int,
        preferred_height: int,
        preferred_fps: int,
    ) -> list[tuple[int, int, int]]:
        candidates = [
            (preferred_width, preferred_height, preferred_fps),
            (1280, 720, min(preferred_fps, 30)),
            (848, 480, min(preferred_fps, 30)),
            (640, 480, min(preferred_fps, 30)),
        ]
        return list(dict.fromkeys(candidates))

    def _start_streams(
        self,
        preferred_width: int,
        preferred_height: int,
        preferred_fps: int,
    ) -> tuple[object, object, int, int, int]:
        errors: list[str] = []
        for width, height, fps in self._candidate_profiles(
            preferred_width, preferred_height, preferred_fps
        ):
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            try:
                profile = pipeline.start(config)
            except Exception as error:
                errors.append(f"{width}x{height}@{fps}: {error}")
                continue
            return pipeline, profile, width, height, fps
        detail = "; ".join(errors)
        raise RuntimeError(f"RealSense has no matching synchronized color/depth profile ({detail})")

    def _set_option(self, sensor: object, option: object, value: float) -> bool:
        try:
            if not sensor.supports(option):
                return False
            option_range = sensor.get_option_range(option)
            clamped = min(max(float(value), option_range.min), option_range.max)
            sensor.set_option(option, clamped)
            return True
        except Exception:
            return False

    def _configure_depth_sensor(self, sensor: object, preset: str) -> list[str]:
        settings: list[str] = []
        preset_attributes = {
            "high-accuracy": "high_accuracy",
            "high-density": "high_density",
        }
        if preset not in preset_attributes:
            raise ValueError(f"Unsupported RealSense depth preset: {preset}")
        try:
            preset_value = float(
                getattr(rs.rs400_visual_preset, preset_attributes[preset])
            )
        except (AttributeError, TypeError, ValueError):
            preset_value = 3.0 if preset == "high-accuracy" else 4.0
        if self._set_option(sensor, rs.option.visual_preset, preset_value):
            settings.append(f"{preset} preset")
        if self._set_option(sensor, rs.option.emitter_enabled, 1.0):
            settings.append("IR emitter")
        try:
            laser_range = sensor.get_option_range(rs.option.laser_power)
            laser_power = laser_range.max
        except Exception:
            laser_power = None
        if laser_power is not None and self._set_option(sensor, rs.option.laser_power, laser_power):
            settings.append(f"laser {laser_power:g}")
        return settings

    def _configure_filters(self) -> None:
        self._set_option(self.spatial, rs.option.filter_magnitude, 2.0)
        self._set_option(self.spatial, rs.option.filter_smooth_alpha, 0.5)
        self._set_option(self.spatial, rs.option.filter_smooth_delta, 20.0)
        self._set_option(self.spatial, rs.option.holes_fill, 0.0)
        self._set_option(self.temporal, rs.option.filter_smooth_alpha, 0.4)
        self._set_option(self.temporal, rs.option.filter_smooth_delta, 20.0)

    @property
    def stream_description(self) -> str:
        settings = ", ".join(self.accuracy_settings) or "default depth controls"
        return (
            f"{self.usb_type} depth-to-color aligned {self.width}x{self.height} "
            f"at {self.fps} FPS; {settings}; scale {self.depth_scale_m * 1000:.4g} mm/unit"
        )

    def read(self, timeout_ms: int = 1000) -> RGBDFrame:
        deadline = time.monotonic() + max(timeout_ms, 1) / 1000.0
        frames = None
        while time.monotonic() < deadline:
            try:
                frames = self.pipeline.wait_for_frames(
                    max(1, int((deadline - time.monotonic()) * 1000))
                )
                break
            except RuntimeError:
                pass
        if frames is None:
            raise RuntimeError("RealSense camera did not return synchronized frames")

        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("RealSense frame set is missing aligned color or depth")

        filtered = self.to_disparity.process(depth_frame)
        filtered = self.spatial.process(filtered)
        filtered = self.temporal.process(filtered)
        filtered = self.to_depth.process(filtered)

        color = np.asanyarray(color_frame.get_data()).copy()
        depth = depth_values_to_mm(np.asanyarray(filtered.get_data()), self.depth_scale_m)
        if depth.shape != color.shape[:2]:
            raise RuntimeError(
                f"RealSense alignment shape mismatch: color {color.shape[:2]}, depth {depth.shape}"
            )
        return RGBDFrame(color, depth)

    def release(self) -> None:
        if not self.closed:
            self.pipeline.stop()
            self.closed = True
