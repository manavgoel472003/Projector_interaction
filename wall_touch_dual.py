"""Two-camera projector wall touch.

Splits the job across two RGB-D cameras so each does what it is good at:

* a **tracker** camera, mounted roughly frontal, runs MediaPipe to locate the
  index fingertip (MediaPipe needs a frontal-ish hand and fails on a steep
  grazing view), and
* a **touch** camera, mounted low / grazing to the wall, senses contact from
  depth (a grazing view turns the tiny finger-to-wall gap into a large, easily
  measured offset, which a single frontal depth camera cannot resolve).

The two never need to be stereo-calibrated to each other: their common frame is
the projector/wall plane, and each is tied to it by its own four-corner
homography. A touch fires when the tracker has a fingertip *and* the touch
camera reports near-wall contact at the same projector location.

Either physical camera can play either role -- both backends expose the same
``read() -> RGBDFrame`` interface -- so ``--tracker`` and ``--touch`` may be any
two distinct sensors.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from wall_touch_core import (
    DepthContactTracker,
    build_homography,
    camera_detection_roi,
    index_is_extended,
    point_in_output,
    projector_targets,
    transform_points,
    validate_camera_quad,
)
from wall_touch_paint import (
    PaintBrush,
    camera_landmarks,
    create_landmarker,
    draw_outlined_text,
    make_base_canvas,
    paint_color,
)
from wall_touch_effects import WatercolorPool
from wall_touch_orbbec import OrbbecCamera, orbbec_device_count
from wall_touch_realsense import RealSenseCamera, realsense_device_count

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models/hand_landmarker.task"
CORNER_NAMES = ("top-left", "top-right", "bottom-right", "bottom-left")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-camera projector wall touch: frontal tracker + grazing touch camera.",
    )
    parser.add_argument("--tracker", choices=("orbbec", "realsense"), default="orbbec",
                        help="Frontal camera used for MediaPipe hand tracking.")
    parser.add_argument("--touch", choices=("orbbec", "realsense"), default="realsense",
                        help="Grazing/low camera used for depth contact sensing.")
    parser.add_argument("--realsense-preset", choices=("high-accuracy", "high-density"),
                        default="high-density")
    parser.add_argument("--projector-width", type=int, default=1920)
    parser.add_argument("--projector-height", type=int, default=1200)
    parser.add_argument("--projector-x", type=int, default=0)
    parser.add_argument("--projector-y", type=int, default=0)
    parser.add_argument("--debug-x", type=int, default=1980)
    parser.add_argument("--debug-y", type=int, default=60)
    parser.add_argument("--debug-width", type=int, default=960,
                        help="Width of each camera preview window.")
    parser.add_argument("--debug-height", type=int, default=600,
                        help="Height of each camera preview window.")
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--detection-confidence", type=float, default=0.40)
    parser.add_argument("--detection-size", type=int, default=384,
                        help="Long-side px the tracker ROI is resized to before MediaPipe.")
    parser.add_argument("--enhance-contrast", action="store_true",
                        help="CLAHE-normalize the tracker ROI for bright/washed walls.")
    parser.add_argument("--require-index-extension", action="store_true")
    parser.add_argument("--mode", choices=("spill", "paint"), default="spill")
    parser.add_argument("--brush-radius", type=int, default=46)
    parser.add_argument("--paint-alpha", type=float, default=0.46)
    parser.add_argument("--touch-wall-frames", type=int, default=45,
                        help="Frames of clear wall averaged to learn the touch camera plane.")
    parser.add_argument("--touch-gap-mm", type=float, default=25.0,
                        help="Max near-wall gap counted as contact on the touch camera.")
    parser.add_argument("--touch-min-contact-area", type=int, default=6)
    parser.add_argument("--touch-match-px", type=float, default=140.0,
                        help="Max projector-space distance between fingertip and a contact.")
    args = parser.parse_args()
    if args.tracker == args.touch:
        parser.error("--tracker and --touch must be two different cameras")
    return args


def open_camera(kind: str, realsense_preset: str):
    if kind == "realsense":
        if realsense_device_count() <= 0:
            raise RuntimeError("No RealSense camera detected (needed for this role).")
        return RealSenseCamera(depth_preset=realsense_preset)
    if orbbec_device_count() <= 0:
        raise RuntimeError("No Orbbec camera detected (needed for this role).")
    return OrbbecCamera()


def prepare_detection_rgb(
    crop: np.ndarray, detection_size: int, clahe: object | None
) -> np.ndarray:
    """Resize a crop to the detection target size and optionally boost contrast."""
    image = crop
    long_side = max(crop.shape[0], crop.shape[1])
    if long_side > 0 and detection_size > 0:
        scale = min(detection_size / long_side, 4.0)
        if abs(scale - 1.0) > 0.02:
            image = cv2.resize(
                crop, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
            )
    if clahe is not None:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        channels = list(cv2.split(lab))
        channels[0] = clahe.apply(channels[0])
        image = cv2.cvtColor(cv2.merge(channels), cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def learn_wall_reference(frames: list[np.ndarray]) -> np.ndarray:
    """Per-pixel wall depth from a stack of clear-wall frames (invalid -> 0)."""
    stack = np.stack(frames)
    masked = np.ma.masked_less(stack, 100.0)
    return np.ma.median(masked, axis=0).filled(0.0).astype(np.float32)


def touch_contacts_in_projector(
    tracker: DepthContactTracker,
    depth_mm: np.ndarray,
    homography: np.ndarray,
    gap_mm: float,
    min_contact_area: int,
    output_size: tuple[int, int],
) -> list[np.ndarray]:
    """Near-wall contacts from the touch camera, mapped to projector coordinates."""
    contacts: list[np.ndarray] = []
    for observation in tracker.observations(depth_mm):
        if observation.contact_area < min_contact_area or observation.gap_mm > gap_mm:
            continue
        projector_point = transform_points(
            homography, observation.camera_point.reshape(1, 2)
        )[0]
        if point_in_output(projector_point, *output_size, margin=8):
            contacts.append(projector_point)
    return contacts


def main() -> None:
    args = parse_args()
    output_size = (args.projector_width, args.projector_height)
    output_points = projector_targets(*output_size)

    tracker_cam = open_camera(args.tracker, args.realsense_preset)
    touch_cam = open_camera(args.touch, args.realsense_preset)
    print(f"Tracker [{args.tracker}]: {tracker_cam.identity}")
    print(f"Touch   [{args.touch}]: {touch_cam.identity}")

    landmarker = create_landmarker(args.model, args.detection_confidence)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if args.enhance_contrast else None

    tracker_window = "Wall Touch Dual - TRACKER"
    touch_window = "Wall Touch Dual - TOUCH"
    projector_window = "Wall Touch Dual - PROJECTOR"
    for name, offset in ((tracker_window, 0), (touch_window, 1)):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(name, args.debug_x, args.debug_y + offset * (args.debug_height + 40))
        cv2.resizeWindow(name, args.debug_width, args.debug_height)
    cv2.namedWindow(projector_window, cv2.WINDOW_NORMAL)
    cv2.moveWindow(projector_window, args.projector_x, args.projector_y)
    cv2.resizeWindow(projector_window, args.projector_width, args.projector_height)
    fullscreen = not args.windowed
    if fullscreen:
        cv2.setWindowProperty(projector_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    tracker_clicks: list[list[float]] = []
    touch_clicks: list[list[float]] = []

    def make_click_handler(clicks: list[list[float]], label: str):
        def handler(event: int, x: int, y: int, flags: int, param: object) -> None:
            del flags, param
            if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
                clicks.append([float(x), float(y)])
                print(f"{label} corner {CORNER_NAMES[len(clicks) - 1]}: {x}, {y}")
        return handler

    cv2.setMouseCallback(tracker_window, make_click_handler(tracker_clicks, "Tracker"))
    cv2.setMouseCallback(touch_window, make_click_handler(touch_clicks, "Touch"))

    tracker_matrix: np.ndarray | None = None
    tracker_camera_points: np.ndarray | None = None
    tracker_roi: tuple[int, int, int, int] | None = None
    touch_matrix: np.ndarray | None = None
    touch_camera_points: np.ndarray | None = None
    touch_wall_frames: list[np.ndarray] = []
    touch_tracker: DepthContactTracker | None = None

    canvas = make_base_canvas(*output_size)
    brush = PaintBrush(args.brush_radius, args.paint_alpha)
    spill = WatercolorPool(*output_size)
    smoothed_tip: np.ndarray | None = None
    active_frames = 0
    last_timestamp_ms = 0

    print("Calibrate: click the four projected targets in the TRACKER window, then in the "
          "TOUCH window, then keep the wall clear while the touch plane is learned.")

    try:
        while True:
            try:
                tracker_rgbd = tracker_cam.read(timeout_ms=800)
                touch_rgbd = touch_cam.read(timeout_ms=800)
            except RuntimeError as error:
                print(f"Camera read warning: {error}")
                continue
            tracker_frame = tracker_rgbd.color_bgr
            touch_frame = touch_rgbd.color_bgr
            touch_depth = touch_rgbd.depth_mm
            tracker_size = (tracker_frame.shape[1], tracker_frame.shape[0])
            touch_size = (touch_frame.shape[1], touch_frame.shape[0])
            tracker_debug = tracker_frame.copy()
            touch_debug = touch_frame.copy()

            # --- resolve calibration state ---------------------------------
            if tracker_matrix is None and len(tracker_clicks) == 4:
                pts = np.array(tracker_clicks, dtype=np.float32)
                try:
                    validate_camera_quad(pts, tracker_size)
                    tracker_matrix = build_homography(pts, output_points)
                    tracker_camera_points = pts
                    tracker_roi = camera_detection_roi(pts, tracker_size)
                    print("Tracker geometry accepted.")
                except ValueError as error:
                    print(f"Tracker geometry rejected: {error}")
                    tracker_clicks.clear()
            if touch_matrix is None and len(touch_clicks) == 4:
                pts = np.array(touch_clicks, dtype=np.float32)
                try:
                    validate_camera_quad(pts, touch_size)
                    touch_matrix = build_homography(pts, output_points)
                    touch_camera_points = pts
                    print("Touch geometry accepted. Keep the wall clear to learn its depth.")
                except ValueError as error:
                    print(f"Touch geometry rejected: {error}")
                    touch_clicks.clear()

            learning_wall = (
                touch_matrix is not None
                and touch_tracker is None
                and touch_depth is not None
            )
            if learning_wall:
                touch_wall_frames.append(touch_depth.copy())
                if len(touch_wall_frames) >= args.touch_wall_frames:
                    wall_reference = learn_wall_reference(touch_wall_frames)
                    touch_tracker = DepthContactTracker(
                        wall_reference,
                        touch_camera_points,
                        near_wall_limit_mm=args.touch_gap_mm + 15.0,
                        minimum_contact_area=args.touch_min_contact_area,
                    )
                    print("Touch plane learned. Point at the wall to paint.")

            calibrated = tracker_matrix is not None and touch_tracker is not None

            # --- tracker: MediaPipe fingertip ------------------------------
            fingertip = None
            extended = False
            landmarks = None
            if tracker_matrix is not None and tracker_roi is not None:
                x0, y0, x1, y1 = tracker_roi
                crop = tracker_frame[y0:y1, x0:x1]
                rgb = prepare_detection_rgb(crop, args.detection_size, clahe)
                last_timestamp_ms += 1
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)),
                    last_timestamp_ms,
                )
                landmarks = camera_landmarks(result, crop.shape, (x0, y0))
                cv2.rectangle(tracker_debug, (x0, y0), (x1 - 1, y1 - 1), (210, 150, 40), 1)
                if landmarks is not None:
                    extended = index_is_extended(landmarks)
                    raw_tip = transform_points(tracker_matrix, landmarks[8].reshape(1, 2))[0]
                    smoothed_tip = raw_tip if smoothed_tip is None else 0.4 * smoothed_tip + 0.6 * raw_tip
                    fingertip = smoothed_tip.copy()
                    tip_px = tuple(np.rint(landmarks[8]).astype(int))
                    cv2.circle(tracker_debug, tip_px, 9, (70, 230, 90), 2, cv2.LINE_AA)
                else:
                    smoothed_tip = None

            # --- touch: near-wall contacts ---------------------------------
            contacts: list[np.ndarray] = []
            if touch_tracker is not None and touch_depth is not None and touch_matrix is not None:
                contacts = touch_contacts_in_projector(
                    touch_tracker, touch_depth, touch_matrix,
                    args.touch_gap_mm, args.touch_min_contact_area, output_size,
                )

            # --- fuse: fingertip present AND a contact at the same place ----
            inside = fingertip is not None and point_in_output(fingertip, *output_size, margin=4)
            pose_ok = extended or not args.require_index_extension
            match = False
            if calibrated and inside and pose_ok and contacts:
                nearest = min(float(np.linalg.norm(c - fingertip)) for c in contacts)
                match = nearest <= args.touch_match_px
            # brief grace so a one-frame contact dropout does not break a stroke
            active_frames = min(active_frames + 1, 6) if match else max(active_frames - 3, 0)
            touch_active = active_frames > 0

            if touch_active and fingertip is not None:
                color = paint_color(fingertip, *output_size)
                if args.mode == "spill":
                    spill.add_drop(fingertip, color, args.brush_radius)
                else:
                    brush.apply(canvas, fingertip, color)

            # --- render -----------------------------------------------------
            if args.mode == "spill":
                spill.step()
                art = spill.render()
            else:
                art = canvas.copy()
            projector_frame = art
            if not calibrated:
                projector_frame = art.copy()
                for index, target in enumerate(output_points):
                    center = tuple(np.rint(target).astype(int))
                    cv2.circle(projector_frame, center, 34, (60, 210, 255), 5, cv2.LINE_AA)
                    cv2.circle(projector_frame, center, 6, (60, 210, 255), -1, cv2.LINE_AA)
            elif fingertip is not None and inside:
                color = (70, 220, 90) if touch_active else (30, 190, 255)
                cv2.circle(projector_frame, tuple(np.rint(fingertip).astype(int)),
                           args.brush_radius + 8, color, 5, cv2.LINE_AA)

            for target, clicks, done_pts in (
                (tracker_debug, tracker_clicks, tracker_camera_points),
                (touch_debug, touch_clicks, touch_camera_points),
            ):
                if done_pts is not None:
                    cv2.polylines(target, [np.rint(done_pts).astype(np.int32)], True,
                                  (80, 235, 100), 3)
                for index, click in enumerate(clicks):
                    center = tuple(np.rint(np.asarray(click)).astype(int))
                    cv2.circle(target, center, 13, (0, 220, 255), 3, cv2.LINE_AA)
                    cv2.circle(target, center, 4, (0, 220, 255), -1, cv2.LINE_AA)
                    cv2.putText(target, str(index + 1), (center[0] + 15, center[1] - 11),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2, cv2.LINE_AA)
            for contact in contacts:
                if touch_camera_points is not None and touch_matrix is not None:
                    cam_pt = transform_points(np.linalg.inv(touch_matrix),
                                              contact.reshape(1, 2))[0]
                    cv2.circle(touch_debug, tuple(np.rint(cam_pt).astype(int)), 8,
                               (0, 220, 255), 2, cv2.LINE_AA)

            if not calibrated:
                if tracker_matrix is None:
                    msg = f"TRACKER: click target {len(tracker_clicks) + 1}/4"
                elif touch_matrix is None:
                    msg = f"TOUCH: click target {len(touch_clicks) + 1}/4"
                else:
                    msg = f"LEARNING WALL: {len(touch_wall_frames)}/{args.touch_wall_frames} (keep clear)"
            else:
                msg = f"TOUCH {'ON' if touch_active else 'off'} | contacts: {len(contacts)}"
            draw_outlined_text(tracker_debug, msg, (12, 28), 0.6)

            cv2.imshow(tracker_window, tracker_debug)
            cv2.imshow(touch_window, touch_debug)
            cv2.imshow(projector_window, projector_frame)

            key = cv2.waitKeyEx(1)
            if key in (27, ord("q")):
                break
            if key == ord("c"):
                canvas = make_base_canvas(*output_size)
                spill.clear()
                print("Artwork cleared.")
            elif key == ord("r"):
                tracker_clicks.clear()
                touch_clicks.clear()
                tracker_matrix = touch_matrix = touch_tracker = None
                tracker_camera_points = touch_camera_points = tracker_roi = None
                touch_wall_frames.clear()
                smoothed_tip = None
                active_frames = 0
                print("Calibration reset. Click the four targets in each window again.")
            elif key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(
                    projector_window, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
                )
    finally:
        landmarker.close()
        tracker_cam.release()
        touch_cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
