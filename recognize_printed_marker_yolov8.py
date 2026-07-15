import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_MODEL = Path("runs_yolov8_cls") / "zoumaguangbei_cls_camstyle" / "weights" / "best.pt"
PAGE_WIDTH_CM = 12.0
IMAGE_HEIGHT_CM = 12.0
RED_HEIGHT_CM = 5.0
PAGE_HEIGHT_CM = IMAGE_HEIGHT_CM + RED_HEIGHT_CM
YOLO_IMGSZ_MULTIPLE = 32
NEAR_GEOMETRY_FIX_DEFAULT = 0.15
DIRECTION_STABLE_FRAMES_DEFAULT = 4
DIRECTION_RESET_DEG_DEFAULT = 35.0

# Ground-plane calibration for the 320x240 runtime camera. Distances are
# measured from the rear axle; inverse distance is nearly linear near horizon.
CALIBRATION_WIDTH = 320.0
CALIBRATION_HEIGHT = 240.0
CALIBRATION_CENTER_X = CALIBRATION_WIDTH / 2.0
CALIBRATION_HORIZON_Y = 34.0
CALIBRATION_ROWS = np.array(
    [CALIBRATION_HORIZON_Y, 44.0, 46.0, 50.0, 55.0, 66.0, 80.0, 85.0, 92.0, 100.0, 113.0, 132.0, 158.0, 190.0],
    dtype=np.float64,
)
CALIBRATION_DISTANCES_CM = np.array(
    [np.inf, 360.0, 300.0, 240.0, 180.0, 120.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0],
    dtype=np.float64,
)
CALIBRATION_INV_DISTANCES = np.where(
    np.isfinite(CALIBRATION_DISTANCES_CM),
    1.0 / CALIBRATION_DISTANCES_CM,
    0.0,
)
CAMERA_HEIGHT_CM = 28.0
CAMERA_FORWARD_FROM_REAR_AXLE_CM = 15.0
CAMERA_PITCH_RAD = np.deg2rad(45.0)
CAMERA_HORIZONTAL_FOV_RAD = np.deg2rad(120.0)
CAMERA_FOCAL_X = CALIBRATION_WIDTH / (2.0 * np.tan(CAMERA_HORIZONTAL_FOV_RAD / 2.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect a 12x5cm red marker, rectify 12x17cm print, and classify the top image.")
    parser.add_argument("--mode", choices=("image", "webcam", "usbcam"), default="image", help="Run on one image, the default webcam path, or an explicit USB camera path.")
    parser.add_argument("--source", type=Path, help="Input image path for image mode.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index for webcam/usbcam mode.")
    parser.add_argument("--camera-width", type=int, default=1280, help="Requested camera width.")
    parser.add_argument("--camera-height", type=int, default=720, help="Requested camera height.")
    parser.add_argument("--camera-backend", choices=("auto", "any", "dshow", "msmf"), default="auto", help="OpenCV camera backend. auto prefers USB-friendly backends for usbcam mode.")
    parser.add_argument("--camera-brightness", type=float, help="Requested camera brightness.")
    parser.add_argument("--camera-exposure", type=float, help="Requested camera exposure.")
    parser.add_argument("--camera-gain", type=float, help="Requested camera gain.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="YOLOv8 classification model path.")
    parser.add_argument("--device", default="cpu", help="YOLO device: cpu, 0, 0,1, etc.")
    parser.add_argument("--imgsz", type=int, default=128, help="YOLO classification image size.")
    parser.add_argument("--output-width", type=int, default=720, help="Rectified 12x17 image width in pixels.")
    parser.add_argument("--save-dir", type=Path, default=Path("runs_printed_marker"), help="Output directory.")
    parser.add_argument("--json-name", default="result.json", help="Image-mode result JSON name.")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Low confidence threshold.")
    parser.add_argument("--first-aid-conf-thres", type=float, default=0.80, help="Minimum confidence required to accept class C-急救包.")
    parser.add_argument("--red-h-low1", type=int, default=0, help="First red hue lower bound.")
    parser.add_argument("--red-h-high1", type=int, default=12, help="First red hue upper bound.")
    parser.add_argument("--red-h-low2", type=int, default=168, help="Second red hue lower bound.")
    parser.add_argument("--red-h-high2", type=int, default=180, help="Second red hue upper bound.")
    parser.add_argument("--red-s-min", type=int, default=80, help="Minimum red saturation.")
    parser.add_argument("--red-v-min", type=int, default=50, help="Minimum red value.")
    parser.add_argument("--min-red-area", type=float, default=80.0, help="Minimum red contour area in pixels.")
    parser.add_argument("--red-aspect-min", type=float, default=1.0, help="Minimum red rotated-rectangle aspect ratio.")
    parser.add_argument("--red-aspect-max", type=float, default=6.0, help="Maximum red rotated-rectangle aspect ratio.")
    parser.add_argument("--morph-kernel", type=int, default=3, help="Morphology kernel size.")
    parser.add_argument("--frame-skip", type=int, default=1, help="Classify every N webcam frames.")
    parser.add_argument("--smooth-confirm-frames", type=int, default=2, help="Consecutive frames required before switching to a new recognized class in realtime modes.")
    parser.add_argument("--smooth-unknown-hold-frames", type=int, default=4, help="Consecutive unknown frames required before dropping a stable realtime result to unknown.")
    parser.add_argument("--direction-stable-frames", type=int, default=DIRECTION_STABLE_FRAMES_DEFAULT, help="Recent frame count used to stabilize marker direction in realtime modes.")
    parser.add_argument("--direction-reset-deg", type=float, default=DIRECTION_RESET_DEG_DEFAULT, help="Reset direction history when heading jumps more than this many degrees.")
    parser.add_argument("--save-debug", action="store_true", help="Save mask, overlay, rectified image, and crop.")
    parser.add_argument("--show", action="store_true", help="Show OpenCV windows.")
    parser.add_argument("--no-save", action="store_true", help="Do not save outputs.")
    parser.add_argument("--crop-scale", type=float, default=1.0, help="Postprocess selected image scale before classification.")
    parser.add_argument("--crop-exposure", type=float, default=1.0, help="Postprocess selected image exposure before classification.")
    parser.add_argument("--crop-contrast", type=float, default=1.0, help="Postprocess selected image contrast before classification.")
    parser.add_argument("--crop-blur", type=float, default=0.0, help="Postprocess selected image Gaussian blur radius before classification.")
    parser.add_argument(
        "--near-geometry-fix",
        type=float,
        default=NEAR_GEOMETRY_FIX_DEFAULT,
        help="Near-field geometry correction. Positive narrows width and stretches height; 0 disables it.",
    )
    parser.add_argument("--roi-top", type=float, default=0.2, help="ROI top ratio (0-1, 0=top).")
    parser.add_argument("--roi-bottom", type=float, default=0.78, help="ROI bottom ratio (0-1, 1=bottom).")
    parser.add_argument("--roi-left", type=float, default=0.1, help="ROI left ratio (0-1, 0=left edge).")
    parser.add_argument("--roi-right", type=float, default=0.9, help="ROI right ratio (0-1, 1=right edge).")
    return parser.parse_args()


def imread_unicode(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return image


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".jpg"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise OSError(f"Unable to encode image: {path}")
    encoded.tofile(str(path))


def load_model(model_path: Path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Missing dependency: ultralytics. Install it with: python -m pip install ultralytics") from exc

    resolved = model_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Model not found: {resolved}")
    return YOLO(str(resolved))


def camera_backend_candidates(mode: str, backend: str) -> list[int | None]:
    if backend == "any":
        return [None]
    if backend == "dshow":
        return [cv2.CAP_DSHOW]
    if backend == "msmf":
        return [cv2.CAP_MSMF]
    if mode == "usbcam":
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]
    return [None, cv2.CAP_DSHOW, cv2.CAP_MSMF]


def open_camera(args: argparse.Namespace) -> cv2.VideoCapture:
    tried: list[str] = []
    for backend in camera_backend_candidates(args.mode, args.camera_backend):
        if backend is None:
            capture = cv2.VideoCapture(args.camera)
            backend_name = "default"
        else:
            capture = cv2.VideoCapture(args.camera, backend)
            backend_name = "dshow" if backend == cv2.CAP_DSHOW else "msmf"
        tried.append(backend_name)
        if capture.isOpened():
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
            if args.camera_brightness is not None:
                capture.set(cv2.CAP_PROP_BRIGHTNESS, args.camera_brightness)
            if args.camera_exposure is not None:
                capture.set(cv2.CAP_PROP_EXPOSURE, args.camera_exposure)
            if args.camera_gain is not None:
                capture.set(cv2.CAP_PROP_GAIN, args.camera_gain)
            return capture
        capture.release()
    tried_text = ", ".join(tried)
    raise RuntimeError(f"Unable to open camera {args.camera} with backends: {tried_text}")


def roi_bounds(frame: np.ndarray, args: argparse.Namespace) -> tuple[int, int, int, int]:
    height, width = frame.shape[:2]
    y0 = int(round(float(args.roi_top) * height))
    y1 = int(round(float(args.roi_bottom) * height))
    x0 = int(round(float(args.roi_left) * width))
    x1 = int(round(float(args.roi_right) * width))
    y0 = max(0, min(height, y0))
    y1 = max(y0, min(height, y1))
    x0 = max(0, min(width, x0))
    x1 = max(x0, min(width, x1))
    return y0, y1, x0, x1


def make_red_mask(frame: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower1 = np.array([args.red_h_low1, args.red_s_min, args.red_v_min], dtype=np.uint8)
    upper1 = np.array([args.red_h_high1, 255, 255], dtype=np.uint8)
    lower2 = np.array([args.red_h_low2, args.red_s_min, args.red_v_min], dtype=np.uint8)
    upper2 = np.array([args.red_h_high2, 255, 255], dtype=np.uint8)
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))

    # ROI：仅保留 [roi_left:roi_right, roi_top:roi_bottom] 区域内的红块
    if args.roi_top > 0.0 or args.roi_bottom < 1.0 or args.roi_left > 0.0 or args.roi_right < 1.0:
        roi_mask = np.zeros_like(mask)
        y0, y1, x0, x1 = roi_bounds(frame, args)
        roi_mask[y0:y1, x0:x1] = 255
        mask = cv2.bitwise_and(mask, roi_mask)

    kernel_size = max(1, args.morph_kernel)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def order_quad_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = pts[:, 0] - pts[:, 1]
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmax(diffs)]
    ordered[3] = pts[np.argmin(diffs)]
    return ordered


def contour_quad(contour: np.ndarray) -> np.ndarray:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
    if len(approx) == 4:
        return order_quad_points(approx.reshape(4, 2))
    rect = cv2.minAreaRect(contour)
    return order_quad_points(cv2.boxPoints(rect))


def detect_red_patch(frame: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    mask = make_red_mask(frame, args)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_height, _ = frame.shape[:2]
    _, roi_y1, _, _ = roi_bounds(frame, args)
    clipped_by_bottom = False
    best: dict[str, Any] | None = None

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < args.min_red_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        bottom = y + h - 1
        if (roi_y1 < frame_height and bottom >= roi_y1 - 2) or bottom >= frame_height - 2:
            clipped_by_bottom = True
            continue

        rect = cv2.minAreaRect(contour)
        (cx, cy), (rw, rh), _ = rect
        if rw <= 1 or rh <= 1:
            continue
        aspect = max(rw, rh) / min(rw, rh)
        if aspect < args.red_aspect_min or aspect > args.red_aspect_max:
            continue

        rect_area = float(rw * rh)
        fill_ratio = area / rect_area if rect_area else 0.0
        if fill_ratio < 0.25:
            continue

        contour_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
        mean_hsv = cv2.mean(hsv, mask=contour_mask)
        y_weight = cy / max(1, frame.shape[0])
        score = area * max(fill_ratio, 0.01) * (1.0 + mean_hsv[1] / 255.0) * (1.0 + 0.25 * y_weight)
        quad = contour_quad(contour)
        candidate = {
            "area": area,
            "aspect": float(aspect),
            "fill_ratio": float(fill_ratio),
            "score": float(score),
            "center": [float(cx), float(cy)],
            "red_quad": quad,
            "red_contour": contour.copy(),
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate

    if best is None:
        error = "red_patch_clipped_by_bottom" if clipped_by_bottom else "red_patch_not_found"
        return {"success": False, "error": error, "mask": mask}

    best["success"] = True
    best["mask"] = mask
    return best


def row_to_distance_cm(rows: np.ndarray, frame_height: int) -> np.ndarray:
    calibration_rows = np.asarray(rows, dtype=np.float64) * CALIBRATION_HEIGHT / max(1, frame_height)
    inverse_distance = np.interp(calibration_rows, CALIBRATION_ROWS, CALIBRATION_INV_DISTANCES)
    distances = np.full(inverse_distance.shape, np.inf, dtype=np.float64)
    valid = inverse_distance > 0.0
    distances[valid] = 1.0 / inverse_distance[valid]
    return distances


def distance_to_row_px(distances_cm: np.ndarray, frame_height: int) -> np.ndarray:
    distances = np.asarray(distances_cm, dtype=np.float64)
    inverse_distance = np.zeros(distances.shape, dtype=np.float64)
    valid = np.isfinite(distances) & (distances > 0.0)
    inverse_distance[valid] = 1.0 / distances[valid]
    calibration_rows = np.interp(inverse_distance, CALIBRATION_INV_DISTANCES, CALIBRATION_ROWS)
    return calibration_rows * max(1, frame_height) / CALIBRATION_HEIGHT


def camera_depth_cm(distances_from_rear_axle_cm: np.ndarray) -> np.ndarray:
    forward_from_camera = np.asarray(distances_from_rear_axle_cm, dtype=np.float64) - CAMERA_FORWARD_FROM_REAR_AXLE_CM
    return (
        forward_from_camera * np.cos(CAMERA_PITCH_RAD)
        + CAMERA_HEIGHT_CM * np.sin(CAMERA_PITCH_RAD)
    )


def image_points_to_ground(points: np.ndarray, frame_shape: tuple[int, ...]) -> np.ndarray:
    height, width = frame_shape[:2]
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    calibration_x = pts[:, 0] * CALIBRATION_WIDTH / max(1, width)
    distances = row_to_distance_cm(pts[:, 1], height)
    depth = camera_depth_cm(distances)
    lateral = (calibration_x - CALIBRATION_CENTER_X) * depth / CAMERA_FOCAL_X
    return np.column_stack((lateral, distances))


def ground_points_to_image(points: np.ndarray, frame_shape: tuple[int, ...]) -> np.ndarray:
    height, width = frame_shape[:2]
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    lateral = pts[:, 0]
    distances = pts[:, 1]
    depth = camera_depth_cm(distances)
    calibration_x = CALIBRATION_CENTER_X + CAMERA_FOCAL_X * lateral / depth
    x = calibration_x * max(1, width) / CALIBRATION_WIDTH
    y = distance_to_row_px(distances, height)
    return np.column_stack((x, y)).astype(np.float32)


def fit_image_width_edge(contour_points: np.ndarray, image_box: np.ndarray) -> np.ndarray:
    points = np.asarray(contour_points, dtype=np.float64).reshape(-1, 2)
    edge_a = image_box[1] - image_box[0]
    edge_b = image_box[2] - image_box[1]
    length_a = float(np.linalg.norm(edge_a))
    length_b = float(np.linalg.norm(edge_b))
    if min(length_a, length_b) <= 1e-6:
        raise ValueError("red_patch_ground_fit_failed")

    fallback = image_box[[0, 1]] if length_a >= length_b else image_box[[1, 2]]
    if len(points) < 4:
        return fallback

    center = points.mean(axis=0)
    centered = points - center
    try:
        covariance = np.cov(centered.T)
        values, vectors = np.linalg.eigh(covariance)
    except np.linalg.LinAlgError:
        return fallback

    axis = vectors[:, int(np.argmax(values))]
    axis_length = float(np.linalg.norm(axis))
    if axis_length <= 1e-6:
        return fallback
    axis /= axis_length
    if axis[0] < 0.0:
        axis = -axis

    half_length = max(length_a, length_b) * 0.5
    return np.array([center - axis * half_length, center + axis * half_length], dtype=np.float64)


def estimate_track_forward_axis(frame: np.ndarray | None, center_image: np.ndarray, frame_shape: tuple[int, ...]) -> np.ndarray | None:
    if frame is None:
        return None

    height, width = frame_shape[:2]
    cx, cy = np.asarray(center_image, dtype=np.float64).reshape(2)
    if not (0.0 <= cx < width and 0.0 <= cy < height):
        return None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white_mask = ((hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 115)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

    y0 = max(0, int(round(cy - height * 0.16)))
    y1 = min(height - 1, int(round(cy + height * 0.12)))
    step = max(2, height // 120)
    min_run_width = max(14, int(round(width * 0.04)))
    max_center_jump = max(45.0, width * 0.18)
    rows: list[float] = []
    centers: list[float] = []

    for y in range(y0, y1 + 1, step):
        row = white_mask[y]
        runs: list[tuple[int, int]] = []
        start: int | None = None
        for x, value in enumerate(row):
            if value and start is None:
                start = x
            elif not value and start is not None:
                if x - start >= min_run_width:
                    runs.append((start, x - 1))
                start = None
        if start is not None and width - start >= min_run_width:
            runs.append((start, width - 1))
        if not runs:
            continue

        left_runs = [run for run in runs if run[1] < cx]
        right_runs = [run for run in runs if run[0] > cx]
        if left_runs and right_runs:
            left_run = max(left_runs, key=lambda run: run[1])
            right_run = min(right_runs, key=lambda run: run[0])
            if right_run[0] - left_run[1] <= max(24.0, width * 0.18):
                runs.append((left_run[0], right_run[1]))

        def run_distance(run: tuple[int, int]) -> float:
            left, right = run
            if left <= cx <= right:
                return 0.0
            return min(abs(cx - left), abs(cx - right), abs(cx - (left + right) * 0.5))

        left, right = min(runs, key=run_distance)
        center = (left + right) * 0.5
        if run_distance((left, right)) > max_center_jump:
            continue
        rows.append(float(y))
        centers.append(float(center))

    if len(rows) < 8 or max(rows) - min(rows) < height * 0.08:
        return None

    fit = np.polyfit(np.asarray(rows), np.asarray(centers), 1)
    predicted = np.polyval(fit, np.asarray(rows))
    residual = float(np.sqrt(np.mean((np.asarray(centers) - predicted) ** 2)))
    if residual > width * 0.05:
        return None

    slope, intercept = float(fit[0]), float(fit[1])
    mid_y = float(np.median(rows))
    delta_y = max(18.0, (max(rows) - min(rows)) * 0.35)
    near_y = min(height - 1.0, mid_y + delta_y)
    far_y = max(0.0, mid_y - delta_y)
    line_points = np.array(
        [
            [slope * near_y + intercept, near_y],
            [slope * far_y + intercept, far_y],
        ],
        dtype=np.float64,
    )
    if not np.all((line_points[:, 0] >= 0.0) & (line_points[:, 0] < width)):
        return None

    ground = image_points_to_ground(line_points, frame_shape)
    if not np.all(np.isfinite(ground)):
        return None
    axis = ground[1] - ground[0]
    axis_length = float(np.linalg.norm(axis))
    if axis_length <= 1e-6:
        return None
    axis /= axis_length
    if axis[1] < 0.0:
        axis = -axis
    return axis


def near_geometry_scales(red_distance_cm: float, args: argparse.Namespace | None) -> tuple[float, float, float]:
    correction = float(getattr(args, "near_geometry_fix", NEAR_GEOMETRY_FIX_DEFAULT))
    correction = float(np.clip(correction, -0.5, 0.6))
    weight = float(np.clip((90.0 - red_distance_cm) / 70.0, 0.0, 1.0))
    applied = correction * weight
    width_scale = float(np.clip(1.0 - 0.5 * applied, 0.65, 1.35))
    height_scale = float(np.clip(1.0 + applied, 0.65, 1.65))
    return width_scale, height_scale, applied


def perpendicular_width_axis(forward_axis: np.ndarray) -> np.ndarray:
    axis = np.array([forward_axis[1], -forward_axis[0]], dtype=np.float64)
    length = float(np.linalg.norm(axis))
    if length <= 1e-6:
        return np.array([1.0, 0.0], dtype=np.float64)
    axis /= length
    if axis[0] < 0.0:
        axis = -axis
    return axis


def enforce_image_above_red(
    forward_axis: np.ndarray,
    red_center: np.ndarray,
    red_center_image: np.ndarray,
    red_height_cm: float,
    image_height_cm: float,
    frame_shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, bool, float]:
    red_y = float(np.asarray(red_center_image, dtype=np.float64).reshape(2)[1])
    image_center_offset = red_height_cm * 0.5 + image_height_cm * 0.5

    def projected_image_center_y(axis: np.ndarray) -> float:
        image_center_ground = red_center + axis * image_center_offset
        image_center = ground_points_to_image(image_center_ground.reshape(1, 2), frame_shape)[0]
        return float(image_center[1])

    current_y = projected_image_center_y(forward_axis)
    flipped_axis = -forward_axis
    flipped_y = projected_image_center_y(flipped_axis)
    if np.isfinite(flipped_y) and (not np.isfinite(current_y) or (current_y >= red_y - 0.5 and flipped_y < current_y)):
        return flipped_axis, perpendicular_width_axis(flipped_axis), True, flipped_y
    return forward_axis, perpendicular_width_axis(forward_axis), False, current_y


class DirectionStabilizer:
    def __init__(
        self,
        max_frames: int = DIRECTION_STABLE_FRAMES_DEFAULT,
        reset_degrees: float = DIRECTION_RESET_DEG_DEFAULT,
    ) -> None:
        self.max_frames = max(1, int(max_frames))
        self.reset_degrees = max(1.0, float(reset_degrees))
        self.history: list[np.ndarray] = []

    def reset(self) -> None:
        self.history.clear()

    def apply(self, forward_axis: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        axis = self._normalize_axis(forward_axis)
        previous = self._weighted_mean() if self.history else None
        delta_degrees = 0.0
        reset = False

        if previous is not None:
            delta_degrees = self._angle_between_degrees(previous, axis)
            if delta_degrees > self.reset_degrees:
                self.history = [axis]
                reset = True
                return axis, {
                    "direction_stabilized": False,
                    "direction_history": len(self.history),
                    "direction_delta_deg": delta_degrees,
                    "direction_reset": reset,
                }

        self.history.append(axis)
        if len(self.history) > self.max_frames:
            self.history = self.history[-self.max_frames:]

        stabilized = self._weighted_mean()
        return stabilized, {
            "direction_stabilized": len(self.history) > 1,
            "direction_history": len(self.history),
            "direction_delta_deg": delta_degrees,
            "direction_reset": reset,
        }

    @staticmethod
    def _normalize_axis(axis: np.ndarray) -> np.ndarray:
        normalized = np.asarray(axis, dtype=np.float64).reshape(2)
        length = float(np.linalg.norm(normalized))
        if length <= 1e-6:
            normalized = np.array([0.0, 1.0], dtype=np.float64)
        else:
            normalized = normalized / length
        if normalized[1] < 0.0:
            normalized = -normalized
        return normalized

    @staticmethod
    def _angle_between_degrees(a: np.ndarray, b: np.ndarray) -> float:
        dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
        return float(np.degrees(np.arccos(dot)))

    def _weighted_mean(self) -> np.ndarray:
        if not self.history:
            return np.array([0.0, 1.0], dtype=np.float64)
        weights = np.arange(1, len(self.history) + 1, dtype=np.float64)
        vector = np.sum(np.vstack(self.history) * weights[:, None], axis=0)
        return self._normalize_axis(vector)


def estimate_marker_geometry(
    red_contour: np.ndarray,
    frame_shape: tuple[int, ...],
    args: argparse.Namespace | None = None,
    frame: np.ndarray | None = None,
) -> dict[str, Any]:
    contour_points = np.asarray(red_contour, dtype=np.float32).reshape(-1, 2)
    ground_points = image_points_to_ground(contour_points, frame_shape)
    ground_points = ground_points[np.all(np.isfinite(ground_points), axis=1)]
    if len(ground_points) < 4:
        raise ValueError("red_patch_outside_ground_calibration")

    image_rect = cv2.minAreaRect(contour_points.reshape(-1, 1, 2))
    image_box = cv2.boxPoints(image_rect).astype(np.float64)
    width_edge_image = fit_image_width_edge(contour_points, image_box)
    width_edge_ground = image_points_to_ground(width_edge_image, frame_shape)
    if not np.all(np.isfinite(width_edge_ground)):
        raise ValueError("red_patch_outside_ground_calibration")
    width_axis = width_edge_ground[1] - width_edge_ground[0]
    width_axis_length = float(np.linalg.norm(width_axis))
    if width_axis_length <= 1e-6:
        raise ValueError("red_patch_ground_fit_failed")
    width_axis /= width_axis_length
    if width_axis[0] < 0.0:
        width_axis = -width_axis
    forward_axis = np.array([-width_axis[1], width_axis[0]], dtype=np.float64)
    if forward_axis[1] < 0.0:
        forward_axis = -forward_axis

    moments = cv2.moments(contour_points.reshape(-1, 1, 2))
    if abs(moments["m00"]) > 1e-6:
        center_image = np.array(
            [[moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]],
            dtype=np.float64,
        )
    else:
        center_image = np.array([image_rect[0]], dtype=np.float64)
    red_center = image_points_to_ground(center_image, frame_shape)[0]
    if not np.all(np.isfinite(red_center)):
        raise ValueError("red_patch_outside_ground_calibration")

    width_scale, height_scale, applied_near_fix = near_geometry_scales(float(red_center[1]), args)
    red_height_cm = RED_HEIGHT_CM * height_scale
    image_height_cm = IMAGE_HEIGHT_CM * height_scale
    half_width = PAGE_WIDTH_CM * 0.5 * width_scale

    route_axis_used = False
    route_axis_anchor = "disabled"
    # 赛道切线方向修正先保留代码但关闭入口，避免近处框选被不稳定拟合带偏。
    # page_center_ground = red_center + forward_axis * (image_height_cm * 0.5)
    # page_center_image = ground_points_to_image(page_center_ground.reshape(1, 2), frame_shape)[0]
    # route_forward_axis = estimate_track_forward_axis(frame, page_center_image, frame_shape)
    # if route_forward_axis is not None and page_center_ground[1] <= 120.0:
    #     alignment = abs(float(np.dot(route_forward_axis, forward_axis)))
    #     if alignment >= 0.45:
    #         forward_axis = route_forward_axis
    #         width_axis = np.array([forward_axis[1], -forward_axis[0]], dtype=np.float64)
    #         width_axis /= max(float(np.linalg.norm(width_axis)), 1e-6)
    #         if width_axis[0] < 0.0:
    #             width_axis = -width_axis
    #         route_axis_used = True
    #         route_axis_anchor = "page_center"

    direction_info: dict[str, Any] = {
        "direction_stabilized": False,
        "direction_history": 0,
        "direction_delta_deg": 0.0,
        "direction_reset": False,
    }
    direction_stabilizer = getattr(args, "direction_stabilizer", None)
    if direction_stabilizer is not None:
        forward_axis, direction_info = direction_stabilizer.apply(forward_axis)
        width_axis = perpendicular_width_axis(forward_axis)

    forward_axis, width_axis, image_above_red_forced, image_center_y = enforce_image_above_red(
        forward_axis,
        red_center,
        center_image[0],
        red_height_cm,
        image_height_cm,
        frame_shape,
    )

    forward_projection = ground_points @ forward_axis
    width_projection = ground_points @ width_axis
    observed_width = float(np.percentile(width_projection, 95.0) - np.percentile(width_projection, 5.0))
    observed_height = float(np.percentile(forward_projection, 95.0) - np.percentile(forward_projection, 5.0))
    red_near_center = red_center - forward_axis * (red_height_cm * 0.5)
    red_far_center = red_center + forward_axis * (red_height_cm * 0.5)
    image_far_center = red_far_center + forward_axis * image_height_cm

    red_ground = np.array(
        [
            red_far_center - width_axis * half_width,
            red_far_center + width_axis * half_width,
            red_near_center + width_axis * half_width,
            red_near_center - width_axis * half_width,
        ],
        dtype=np.float64,
    )
    image_ground = np.array(
        [
            image_far_center - width_axis * half_width,
            image_far_center + width_axis * half_width,
            red_far_center + width_axis * half_width,
            red_far_center - width_axis * half_width,
        ],
        dtype=np.float64,
    )
    page_ground = np.array(
        [
            image_far_center - width_axis * half_width,
            image_far_center + width_axis * half_width,
            red_near_center + width_axis * half_width,
            red_near_center - width_axis * half_width,
        ],
        dtype=np.float64,
    )

    return {
        "red_quad": ground_points_to_image(red_ground, frame_shape),
        "image_quad": ground_points_to_image(image_ground, frame_shape),
        "page_quad": ground_points_to_image(page_ground, frame_shape),
        "near_distance_cm": float(red_near_center[1]),
        "heading_deg": float(np.degrees(np.arctan2(forward_axis[0], forward_axis[1]))),
        "observed_width_cm": observed_width,
        "observed_height_cm": observed_height,
        "near_geometry_fix_applied": applied_near_fix,
        "geometry_width_scale": width_scale,
        "geometry_height_scale": height_scale,
        "route_axis_used": route_axis_used,
        "route_axis_anchor": route_axis_anchor,
        "image_above_red_forced": image_above_red_forced,
        "image_center_y": image_center_y,
        "red_center_y": float(center_image[0][1]),
        **direction_info,
    }


def warp_page(frame: np.ndarray, page_quad: np.ndarray, output_width: int) -> np.ndarray:
    output_height = round(output_width * PAGE_HEIGHT_CM / PAGE_WIDTH_CM)
    dst = np.array(
        [[0, 0], [output_width - 1, 0], [output_width - 1, output_height - 1], [0, output_height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(page_quad.astype(np.float32), dst)
    return cv2.warpPerspective(frame, matrix, (output_width, output_height))


def warp_image_area(frame: np.ndarray, image_quad: np.ndarray, output_width: int) -> np.ndarray:
    dst = np.array(
        [[0, 0], [output_width - 1, 0], [output_width - 1, output_width - 1], [0, output_width - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(image_quad.astype(np.float32), dst)
    return cv2.warpPerspective(frame, matrix, (output_width, output_width))


def crop_image_area(rectified: np.ndarray, output_width: int) -> np.ndarray:
    return rectified[:output_width, :output_width].copy()


def center_scale_image(image: np.ndarray, scale: float) -> np.ndarray:
    scale = max(0.05, float(scale))
    if abs(scale - 1.0) < 1e-6:
        return image

    height, width = image.shape[:2]
    scaled_width = max(1, int(round(width * scale)))
    scaled_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)

    if scale > 1.0:
        left = max(0, (scaled_width - width) // 2)
        top = max(0, (scaled_height - height) // 2)
        return resized[top:top + height, left:left + width].copy()

    pad_left = max(0, (width - scaled_width) // 2)
    pad_right = max(0, width - scaled_width - pad_left)
    pad_top = max(0, (height - scaled_height) // 2)
    pad_bottom = max(0, height - scaled_height - pad_top)
    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_REPLICATE,
    )
    return padded[:height, :width].copy()


def apply_crop_postprocess(crop: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    scale = float(getattr(args, "crop_scale", 1.0))
    exposure = float(getattr(args, "crop_exposure", 1.0))
    contrast = float(getattr(args, "crop_contrast", 1.0))
    blur = float(getattr(args, "crop_blur", 0.0))

    processed = center_scale_image(crop, scale)
    if abs(exposure - 1.0) > 1e-6:
        processed = cv2.convertScaleAbs(processed, alpha=max(0.0, exposure), beta=0)
    if abs(contrast - 1.0) > 1e-6:
        data = processed.astype(np.float32)
        processed = np.clip((data - 127.5) * max(0.0, contrast) + 127.5, 0, 255).astype(np.uint8)
    if blur > 0:
        processed = cv2.GaussianBlur(processed, (0, 0), blur)
    return processed


def normalize_yolo_imgsz(imgsz: int | float) -> int:
    target_size = max(YOLO_IMGSZ_MULTIPLE, int(round(imgsz)))
    return ((target_size + YOLO_IMGSZ_MULTIPLE - 1) // YOLO_IMGSZ_MULTIPLE) * YOLO_IMGSZ_MULTIPLE


def resize_for_yolo_input(crop: np.ndarray, imgsz: int) -> np.ndarray:
    target_size = normalize_yolo_imgsz(imgsz)
    height, width = crop.shape[:2]
    if width == target_size and height == target_size:
        return crop
    interpolation = cv2.INTER_AREA if target_size < max(width, height) else cv2.INTER_LINEAR
    return cv2.resize(crop, (target_size, target_size), interpolation=interpolation)


def classify_crop(model, crop: np.ndarray, imgsz: int, device: str, conf_thres: float, first_aid_conf_thres: float) -> dict[str, Any]:
    result = model.predict(source=crop, imgsz=imgsz, device=device, verbose=False)[0]
    top1 = int(result.probs.top1)
    confidence = float(result.probs.top1conf)
    class_name = result.names[top1]
    low_confidence = confidence < conf_thres
    rejected_first_aid = class_name == "C-急救包" and confidence < first_aid_conf_thres
    final_class_id = -1 if rejected_first_aid else top1
    final_class_name = "unknown" if rejected_first_aid else class_name
    final_low_confidence = True if rejected_first_aid else low_confidence
    return {
        "class_id": final_class_id,
        "class_name": final_class_name,
        "confidence": confidence,
        "low_confidence": final_low_confidence,
        "rejected_first_aid": rejected_first_aid,
        "top1_class_id": top1,
        "top1_class_name": class_name,
        "top1_confidence": confidence,
    }


def quad_to_list(quad: np.ndarray) -> list[list[float]]:
    return [[round(float(x), 2), round(float(y), 2)] for x, y in quad]


def page_warning(page_quad: np.ndarray, frame_shape: tuple[int, int, int]) -> str | None:
    height, width = frame_shape[:2]
    inside = 0
    for x, y in page_quad:
        if -width <= x <= width * 2 and -height <= y <= height * 2:
            inside += 1
    if inside < 4:
        return "page_quad_partly_outside_frame"
    return None


def draw_polyline(image: np.ndarray, quad: np.ndarray, color: tuple[int, int, int], label: str) -> None:
    pts = np.round(quad).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [pts], True, color, 2)
    x, y = pts.reshape(-1, 2)[0]
    cv2.putText(image, label, (int(x), int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_overlay(frame: np.ndarray, result: dict[str, Any]) -> np.ndarray:
    overlay = frame.copy()
    # 画 ROI 范围框（黄色虚线效果：实线+半透明底色方便辨识）
    if "roi" in result:
        roi = result["roi"]
        y0, y1, x0, x1 = roi["y0"], roi["y1"], roi["x0"], roi["x1"]
        if x0 > 0 or y0 > 0 or x1 < overlay.shape[1] - 1 or y1 < overlay.shape[0] - 1:
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 255), 2)
            cv2.putText(overlay, "ROI", (x0 + 4, y0 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    if result.get("red_quad") is not None:
        draw_polyline(overlay, np.array(result["red_quad"], dtype=np.float32), (0, 255, 0), "red")
    if result.get("image_quad") is not None:
        draw_polyline(overlay, np.array(result["image_quad"], dtype=np.float32), (255, 0, 0), "image")
    elif result.get("page_quad") is not None:
        draw_polyline(overlay, np.array(result["page_quad"], dtype=np.float32), (255, 0, 0), "page")

    if result.get("success"):
        final_text = f"final: {result['class_name']} {result['confidence']:.2f}"
        top1_name = str(result.get("top1_class_name", result["class_name"]))
        top1_conf = float(result.get("top1_confidence", result["confidence"]))
        top1_text = f"top1: {top1_name} {top1_conf:.2f}"
        color = (0, 255, 255) if result.get("low_confidence") else (0, 255, 0)
        cv2.putText(overlay, final_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(overlay, top1_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    else:
        text = result.get("error", "failed")
        color = (0, 0, 255)
        cv2.putText(overlay, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    return overlay


def recognize_frame(frame: np.ndarray, model, args: argparse.Namespace) -> tuple[dict[str, Any], np.ndarray]:
    result = process_frame(frame, model, args)
    overlay = draw_overlay(frame, result)
    return result, overlay


def _make_roi_dict(frame: np.ndarray, args: argparse.Namespace) -> dict[str, int]:
    y0, y1, x0, x1 = roi_bounds(frame, args)
    return {
        "y0": y0,
        "y1": y1,
        "x0": x0,
        "x1": x1,
    }


def process_frame(frame: np.ndarray, model, args: argparse.Namespace) -> dict[str, Any]:
    detected = detect_red_patch(frame, args)
    if not detected["success"]:
        direction_stabilizer = getattr(args, "direction_stabilizer", None)
        if direction_stabilizer is not None:
            direction_stabilizer.reset()
        return {"success": False, "error": detected["error"], "mask": detected["mask"], "roi": _make_roi_dict(frame, args)}

    raw_imgsz = int(round(args.imgsz))
    yolo_imgsz = normalize_yolo_imgsz(raw_imgsz)
    if yolo_imgsz != raw_imgsz:
        print(f"YOLO imgsz adjusted: {raw_imgsz} -> {yolo_imgsz} ({YOLO_IMGSZ_MULTIPLE}x multiple)")
        args.imgsz = yolo_imgsz

    try:
        geometry = estimate_marker_geometry(detected["red_contour"], frame.shape, args, frame=frame)
    except ValueError as exc:
        return {
            "success": False,
            "error": str(exc),
            "mask": detected["mask"],
            "red_quad": quad_to_list(detected["red_quad"]),
            "roi": _make_roi_dict(frame, args),
        }

    red_quad = geometry["red_quad"]
    image_quad = geometry["image_quad"]
    page_quad = geometry["page_quad"]
    processed_crop = apply_crop_postprocess(warp_image_area(frame, image_quad, args.output_width), args)
    crop = resize_for_yolo_input(processed_crop, yolo_imgsz)
    rectified = warp_page(frame, page_quad, args.output_width)
    cls = classify_crop(model, crop, yolo_imgsz, args.device, args.conf_thres, args.first_aid_conf_thres)
    warning = page_warning(image_quad, frame.shape)

    result: dict[str, Any] = {
        "success": True,
        "class_id": cls["class_id"],
        "class_name": cls["class_name"],
        "confidence": cls["confidence"],
        "low_confidence": cls["low_confidence"],
        "rejected_first_aid": cls["rejected_first_aid"],
        "top1_class_id": cls["top1_class_id"],
        "top1_class_name": cls["top1_class_name"],
        "top1_confidence": cls["top1_confidence"],
        "red_quad": quad_to_list(red_quad),
        "image_quad": quad_to_list(image_quad),
        "page_quad": quad_to_list(page_quad),
        "red_area": detected["area"],
        "red_aspect": detected["aspect"],
        "red_fill_ratio": detected["fill_ratio"],
        "geometry_mode": "ground_calibrated",
        "marker_near_distance_cm": round(geometry["near_distance_cm"], 2),
        "marker_heading_deg": round(geometry["heading_deg"], 2),
        "observed_red_width_cm": round(geometry["observed_width_cm"], 2),
        "observed_red_height_cm": round(geometry["observed_height_cm"], 2),
        "near_geometry_fix_applied": round(geometry["near_geometry_fix_applied"], 3),
        "geometry_width_scale": round(geometry["geometry_width_scale"], 3),
        "geometry_height_scale": round(geometry["geometry_height_scale"], 3),
        "route_axis_used": bool(geometry["route_axis_used"]),
        "route_axis_anchor": geometry["route_axis_anchor"],
        "direction_stabilized": bool(geometry["direction_stabilized"]),
        "direction_history": int(geometry["direction_history"]),
        "direction_delta_deg": round(geometry["direction_delta_deg"], 2),
        "direction_reset": bool(geometry["direction_reset"]),
        "image_above_red_forced": bool(geometry["image_above_red_forced"]),
        "image_center_y": round(geometry["image_center_y"], 2),
        "red_center_y": round(geometry["red_center_y"], 2),
        "rectified_size": [int(rectified.shape[1]), int(rectified.shape[0])],
        "processed_crop_size": [int(processed_crop.shape[1]), int(processed_crop.shape[0])],
        "yolo_imgsz": yolo_imgsz,
        "crop_size": [int(crop.shape[1]), int(crop.shape[0])],
        "model": str(args.model),
        "mask": detected["mask"],
        "rectified": rectified,
        "crop": crop,
        "roi": _make_roi_dict(frame, args),
    }
    if warning:
        result["warning"] = warning
    return result


def json_safe_result(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key not in {"mask", "rectified", "crop", "overlay"}}


class TemporalResultSmoother:
    def __init__(self, confirm_frames: int, unknown_hold_frames: int) -> None:
        self.confirm_frames = max(1, confirm_frames)
        self.unknown_hold_frames = max(1, unknown_hold_frames)
        self.stable_result: dict[str, Any] | None = None
        self.candidate_class_name: str | None = None
        self.candidate_count = 0
        self.unknown_count = 0

    def apply(self, result: dict[str, Any]) -> dict[str, Any]:
        if not result.get("success"):
            self.stable_result = None
            self.candidate_class_name = None
            self.candidate_count = 0
            self.unknown_count = 0
            return result

        class_name = str(result.get("class_name", "unknown"))
        if self.stable_result is None:
            if class_name != "unknown":
                self.stable_result = result.copy()
            return result

        stable_class_name = str(self.stable_result.get("class_name", "unknown"))
        if class_name == stable_class_name:
            self.stable_result = result.copy()
            self.candidate_class_name = None
            self.candidate_count = 0
            self.unknown_count = 0
            smoothed = result.copy()
            smoothed["smoothed_from_history"] = False
            return smoothed

        if class_name == "unknown":
            self.unknown_count += 1
            self.candidate_class_name = None
            self.candidate_count = 0
            if self.unknown_count < self.unknown_hold_frames:
                smoothed = self.stable_result.copy()
                smoothed["confidence"] = result.get("confidence", smoothed.get("confidence", 0.0))
                smoothed["low_confidence"] = True
                smoothed["smoothed_from_history"] = True
                smoothed["raw_class_name"] = class_name
                return smoothed
            self.stable_result = result.copy()
            self.unknown_count = 0
            smoothed = result.copy()
            smoothed["smoothed_from_history"] = False
            return smoothed

        self.unknown_count = 0
        if self.candidate_class_name == class_name:
            self.candidate_count += 1
        else:
            self.candidate_class_name = class_name
            self.candidate_count = 1

        if self.candidate_count >= self.confirm_frames:
            self.stable_result = result.copy()
            self.candidate_class_name = None
            self.candidate_count = 0
            smoothed = result.copy()
            smoothed["smoothed_from_history"] = False
            return smoothed

        smoothed = self.stable_result.copy()
        smoothed["confidence"] = result.get("confidence", smoothed.get("confidence", 0.0))
        smoothed["low_confidence"] = result.get("low_confidence", smoothed.get("low_confidence", False))
        smoothed["smoothed_from_history"] = True
        smoothed["raw_class_name"] = class_name
        return smoothed


def highgui_unavailable(exc: cv2.error) -> bool:
    message = str(exc).lower()
    return "the function is not implemented" in message or "cvshowimage" in message


def safe_imshow(window_name: str, image: np.ndarray) -> None:
    try:
        cv2.imshow(window_name, image)
    except cv2.error as exc:
        if highgui_unavailable(exc):
            raise RuntimeError(
                "OpenCV GUI is unavailable in this Python environment. Use recognize_printed_marker_tkinter.py for realtime preview, or rerun without --show."
            ) from exc
        raise


def safe_destroy_all_windows() -> None:
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        return


def save_image_outputs(out_dir: Path, frame: np.ndarray, result: dict[str, Any], args: argparse.Namespace) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay = draw_overlay(frame, result)
    imwrite_unicode(out_dir / "input_with_overlay.jpg", overlay)
    if result.get("mask") is not None:
        imwrite_unicode(out_dir / "red_mask.jpg", result["mask"])
    if result.get("rectified") is not None:
        imwrite_unicode(out_dir / "rectified_12x17.jpg", result["rectified"])
    if result.get("crop") is not None:
        imwrite_unicode(out_dir / "crop_12x12.jpg", result["crop"])
    (out_dir / args.json_name).write_text(json.dumps(json_safe_result(result), ensure_ascii=False, indent=2), encoding="utf-8")


def run_image(args: argparse.Namespace) -> None:
    if args.source is None:
        raise ValueError("--source is required in image mode.")
    source = args.source.expanduser().resolve()
    frame = imread_unicode(source)
    model = load_model(args.model)
    result = process_frame(frame, model, args)
    out_dir = args.save_dir.expanduser().resolve() / source.stem

    if not args.no_save:
        save_image_outputs(out_dir, frame, result, args)

    print(json.dumps(json_safe_result(result), ensure_ascii=False, indent=2))
    if args.show:
        safe_imshow("input_with_overlay", draw_overlay(frame, result))
        if result.get("rectified") is not None:
            safe_imshow("rectified_12x17", result["rectified"])
        if result.get("crop") is not None:
            safe_imshow("crop_12x12", result["crop"])
        cv2.waitKey(0)
        safe_destroy_all_windows()


def save_webcam_snapshot(snapshot_dir: Path, frame_index: int, frame: np.ndarray, result: dict[str, Any], args: argparse.Namespace) -> None:
    stem = f"frame_{frame_index:06d}"
    frame_dir = snapshot_dir / stem
    save_image_outputs(frame_dir, frame, result, args)


def run_webcam(args: argparse.Namespace) -> None:
    model = load_model(args.model)
    capture = open_camera(args)
    smoother = TemporalResultSmoother(args.smooth_confirm_frames, args.smooth_unknown_hold_frames)
    args.direction_stabilizer = DirectionStabilizer(args.direction_stable_frames, args.direction_reset_deg)

    run_dir = args.save_dir.expanduser().resolve() / f"{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    snapshot_dir = run_dir / "snapshots"
    results_path = run_dir / "results.jsonl"
    if not args.no_save:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        results_path.parent.mkdir(parents=True, exist_ok=True)

    frame_index = 0
    last_result: dict[str, Any] | None = None
    last_time = time.time()

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_index += 1

        if last_result is None or frame_index % max(1, args.frame_skip) == 0:
            raw_result, _ = recognize_frame(frame, model, args)
            result = smoother.apply(raw_result)
            overlay = draw_overlay(frame, result)
            last_result = result
            if not args.no_save:
                with results_path.open("a", encoding="utf-8") as f:
                    record = {"frame": frame_index, "time": datetime.now().isoformat(), **json_safe_result(result)}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            result = last_result
            overlay = draw_overlay(frame, result)

        now = time.time()
        fps = 1.0 / max(now - last_time, 1e-6)
        last_time = now
        cv2.putText(overlay, f"FPS {fps:.1f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        if args.show:
            try:
                safe_imshow("printed_marker_webcam", overlay)
            except RuntimeError:
                capture.release()
                safe_destroy_all_windows()
                raise
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s") and not args.no_save:
                save_webcam_snapshot(snapshot_dir, frame_index, frame, result, args)
        else:
            print(json.dumps(json_safe_result(result), ensure_ascii=False))

    capture.release()
    safe_destroy_all_windows()


def main() -> None:
    args = parse_args()
    if args.mode == "image":
        run_image(args)
    else:
        run_webcam(args)


if __name__ == "__main__":
    main()
