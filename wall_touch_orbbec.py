from __future__ import annotations

from dataclasses import dataclass
import time

import cv2
import numpy as np

try:
    from pyorbbecsdk import (
        Config,
        Context,
        OBAlignMode,
        OBFormat,
        OBFrameAggregateOutputMode,
        OBSensorType,
        Pipeline,
    )
except ImportError as error:  # pragma: no cover - exercised on installations without the optional SDK
    Config = Context = Pipeline = None
    OBAlignMode = OBFormat = OBFrameAggregateOutputMode = OBSensorType = None
    SDK_IMPORT_ERROR: ImportError | None = error
else:
    SDK_IMPORT_ERROR = None


@dataclass(frozen=True)
class RGBDFrame:
    color_bgr: np.ndarray
    depth_mm: np.ndarray


def orbbec_device_count() -> int:
    if Context is None:
        return 0
    try:
        return int(Context().query_devices().get_count())
    except Exception:
        return 0


def color_frame_to_bgr(frame: object) -> np.ndarray:
    width = int(frame.get_width())
    height = int(frame.get_height())
    pixel_format = frame.get_format()
    data = np.frombuffer(frame.get_data(), dtype=np.uint8)
    if pixel_format == OBFormat.RGB:
        return cv2.cvtColor(data.reshape(height, width, 3), cv2.COLOR_RGB2BGR)
    if pixel_format == OBFormat.BGR:
        return data.reshape(height, width, 3).copy()
    if pixel_format in (OBFormat.YUYV, OBFormat.YUY2):
        return cv2.cvtColor(data.reshape(height, width, 2), cv2.COLOR_YUV2BGR_YUY2)
    if pixel_format == OBFormat.UYVY:
        return cv2.cvtColor(data.reshape(height, width, 2), cv2.COLOR_YUV2BGR_UYVY)
    if pixel_format == OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is not None:
            return image
    raise RuntimeError(f"Unsupported Orbbec color format: {pixel_format}")


def depth_frame_to_mm(frame: object) -> np.ndarray:
    width = int(frame.get_width())
    height = int(frame.get_height())
    values = np.frombuffer(frame.get_data(), dtype=np.uint16).reshape(height, width)
    return values.astype(np.float32) * float(frame.get_depth_scale())


class OrbbecCamera:
    def __init__(self, preferred_width: int = 1280, preferred_height: int = 720, fps: int = 30) -> None:
        if Pipeline is None:
            raise RuntimeError(
                "Orbbec SDK is not installed. Run ./install.sh or install pyorbbecsdk2."
            ) from SDK_IMPORT_ERROR

        self.pipeline = Pipeline()
        device_info = self.pipeline.get_device().get_device_info()
        self.connection_type = str(device_info.get_connection_type())
        self.identity = (
            f"{device_info.get_name()} {device_info.get_serial_number()}"
        ).strip()
        if "USB2" in self.connection_type.upper():
            preferred_width, preferred_height, fps = 640, 480, min(fps, 15)
        self.config, self.color_profile, self.depth_profile = self._hardware_d2c_config(
            preferred_width, preferred_height, fps
        )
        try:
            self.pipeline.enable_frame_sync()
        except Exception as error:
            print(f"Orbbec frame-sync warning: {error}")
        self.pipeline.start(self.config)
        self.closed = False
        self.has_received_frame = False

    def _hardware_d2c_config(
        self,
        preferred_width: int,
        preferred_height: int,
        preferred_fps: int,
    ) -> tuple[object, object, object]:
        profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        choices: list[tuple[float, object, object]] = []
        for index in range(len(profiles)):
            color_profile = profiles[index]
            if color_profile.get_format() != OBFormat.RGB:
                continue
            depth_profiles = self.pipeline.get_d2c_depth_profile_list(
                color_profile, OBAlignMode.HW_MODE
            )
            if len(depth_profiles) == 0:
                continue
            depth_profile = min(
                (depth_profiles[depth_index] for depth_index in range(len(depth_profiles))),
                key=lambda depth: (
                    1000 * abs(depth.get_fps() - color_profile.get_fps())
                    + abs(depth.get_width() - color_profile.get_width())
                    + abs(depth.get_height() - color_profile.get_height())
                ),
            )
            score = (
                abs(color_profile.get_width() - preferred_width)
                + abs(color_profile.get_height() - preferred_height)
                + 25 * abs(color_profile.get_fps() - preferred_fps)
                + 1000 * abs(depth_profile.get_fps() - color_profile.get_fps())
            )
            choices.append((score, color_profile, depth_profile))
        if not choices:
            raise RuntimeError("Orbbec camera does not expose hardware depth-to-color alignment")

        _, color_profile, depth_profile = min(choices, key=lambda choice: choice[0])
        config = Config()
        config.enable_stream(depth_profile)
        config.enable_stream(color_profile)
        config.set_align_mode(OBAlignMode.HW_MODE)
        config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        return config, color_profile, depth_profile

    @property
    def stream_description(self) -> str:
        profile = self.color_profile
        depth = self.depth_profile
        return (
            f"{self.connection_type} HW-D2C color {profile.get_width()}x{profile.get_height()} "
            f"+ depth {depth.get_width()}x{depth.get_height()} at {profile.get_fps()} FPS"
        )

    def read(self, timeout_ms: int = 1000) -> RGBDFrame:
        frames = None
        for _ in range(2):
            frames = self.pipeline.wait_for_frames(timeout_ms)
            if frames is not None:
                break
        if frames is None:
            if not self.has_received_frame:
                self.pipeline.stop()
                time.sleep(3.0)
                self.pipeline.start(self.config)
                try:
                    self.pipeline.enable_frame_sync()
                except Exception:
                    pass
                for _ in range(2):
                    frames = self.pipeline.wait_for_frames(timeout_ms)
                    if frames is not None:
                        break
            if frames is None:
                raise RuntimeError("Orbbec camera did not return synchronized frames")
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if color_frame is None or depth_frame is None:
            raise RuntimeError("Orbbec frame set is missing color or depth")
        color = color_frame_to_bgr(color_frame)
        depth = depth_frame_to_mm(depth_frame)
        if depth.shape != color.shape[:2]:
            depth = cv2.resize(
                depth,
                (color.shape[1], color.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        self.has_received_frame = True
        return RGBDFrame(color, depth)

    def release(self) -> None:
        if not self.closed:
            self.pipeline.stop()
            self.closed = True
