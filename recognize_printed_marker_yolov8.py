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
EXTEND_RATIO = IMAGE_HEIGHT_CM / RED_HEIGHT_CM
YOLO_IMGSZ_MULTIPLE = 32


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
    parser.add_argument("--morph-kernel", type=int, default=5, help="Morphology kernel size.")
    parser.add_argument("--frame-skip", type=int, default=1, help="Classify every N webcam frames.")
    parser.add_argument("--smooth-confirm-frames", type=int, default=2, help="Consecutive frames required before switching to a new recognized class in realtime modes.")
    parser.add_argument("--smooth-unknown-hold-frames", type=int, default=4, help="Consecutive unknown frames required before dropping a stable realtime result to unknown.")
    parser.add_argument("--save-debug", action="store_true", help="Save mask, overlay, rectified image, and crop.")
    parser.add_argument("--show", action="store_true", help="Show OpenCV windows.")
    parser.add_argument("--no-save", action="store_true", help="Do not save outputs.")
    parser.add_argument("--crop-scale", type=float, default=1.0, help="Postprocess selected image scale before classification.")
    parser.add_argument("--crop-exposure", type=float, default=1.0, help="Postprocess selected image exposure before classification.")
    parser.add_argument("--crop-contrast", type=float, default=1.0, help="Postprocess selected image contrast before classification.")
    parser.add_argument("--crop-blur", type=float, default=0.0, help="Postprocess selected image Gaussian blur radius before classification.")
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


def make_red_mask(frame: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower1 = np.array([args.red_h_low1, args.red_s_min, args.red_v_min], dtype=np.uint8)
    upper1 = np.array([args.red_h_high1, 255, 255], dtype=np.uint8)
    lower2 = np.array([args.red_h_low2, args.red_s_min, args.red_v_min], dtype=np.uint8)
    upper2 = np.array([args.red_h_high2, 255, 255], dtype=np.uint8)
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))

    # ROI：仅保留 [roi_left:roi_right, roi_top:roi_bottom] 区域内的红块
    H, W = frame.shape[:2]
    if args.roi_top > 0.0 or args.roi_bottom < 1.0 or args.roi_left > 0.0 or args.roi_right < 1.0:
        roi_mask = np.zeros_like(mask)
        y0 = int(round(args.roi_top * H))
        y1 = int(round(args.roi_bottom * H))
        x0 = int(round(args.roi_left * W))
        x1 = int(round(args.roi_right * W))
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
    best: dict[str, Any] | None = None

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < args.min_red_area:
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
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate

    if best is None:
        return {"success": False, "error": "red_patch_not_found", "mask": mask}

    best["success"] = True
    best["mask"] = mask
    return best


def estimate_page_quad_from_red(red_quad: np.ndarray) -> np.ndarray:
    red_tl, red_tr, red_br, red_bl = red_quad.astype(np.float32)
    page_tl = red_tl + (red_tl - red_bl) * EXTEND_RATIO
    page_tr = red_tr + (red_tr - red_br) * EXTEND_RATIO
    return np.array([page_tl, page_tr, red_br, red_bl], dtype=np.float32)


def estimate_image_quad_from_red(red_quad: np.ndarray) -> np.ndarray:
    red_tl, red_tr, red_br, red_bl = red_quad.astype(np.float32)
    left_extend = (red_tl - red_bl) * EXTEND_RATIO
    right_extend = (red_tr - red_br) * EXTEND_RATIO
    image_tl = red_tl + left_extend
    image_tr = red_tr + right_extend
    return np.array([image_tl, image_tr, red_tr, red_tl], dtype=np.float32)


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
    H, W = frame.shape[:2]
    return {
        "y0": int(round(args.roi_top * H)),
        "y1": int(round(args.roi_bottom * H)),
        "x0": int(round(args.roi_left * W)),
        "x1": int(round(args.roi_right * W)),
    }


def process_frame(frame: np.ndarray, model, args: argparse.Namespace) -> dict[str, Any]:
    detected = detect_red_patch(frame, args)
    if not detected["success"]:
        return {"success": False, "error": detected["error"], "mask": detected["mask"], "roi": _make_roi_dict(frame, args)}

    raw_imgsz = int(round(args.imgsz))
    yolo_imgsz = normalize_yolo_imgsz(raw_imgsz)
    if yolo_imgsz != raw_imgsz:
        print(f"YOLO imgsz adjusted: {raw_imgsz} -> {yolo_imgsz} ({YOLO_IMGSZ_MULTIPLE}x multiple)")
        args.imgsz = yolo_imgsz

    red_quad = detected["red_quad"]
    image_quad = estimate_image_quad_from_red(red_quad)
    page_quad = estimate_page_quad_from_red(red_quad)
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
