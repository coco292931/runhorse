import argparse
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from recognize_printed_marker_yolov8 import DEFAULT_MODEL, TemporalResultSmoother, draw_overlay, load_model, open_camera, recognize_frame


CAMERA_CONTROL_PROPS = {
    "camera_brightness": cv2.CAP_PROP_BRIGHTNESS,
    "camera_exposure": cv2.CAP_PROP_EXPOSURE,
    "camera_gain": cv2.CAP_PROP_GAIN,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tkinter GUI for real-time printed marker recognition.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--camera-width", type=int, default=1280, help="Requested camera width.")
    parser.add_argument("--camera-height", type=int, default=720, help="Requested camera height.")
    parser.add_argument("--camera-backend", choices=("auto", "any", "dshow", "msmf"), default="auto", help="OpenCV camera backend.")
    parser.add_argument("--camera-brightness", type=float, help="Requested camera brightness.")
    parser.add_argument("--camera-exposure", type=float, help="Requested camera exposure.")
    parser.add_argument("--camera-gain", type=float, help="Requested camera gain.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="YOLOv8 classification model path.")
    parser.add_argument("--device", default="cpu", help="YOLO device: cpu, 0, 0,1, etc.")
    parser.add_argument("--imgsz", type=int, default=100, help="YOLO classification image size.")
    parser.add_argument("--output-width", type=int, default=720, help="Rectified 12x17 image width in pixels.")
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
    parser.add_argument("--frame-skip", type=int, default=1, help="Classify every N frames.")
    parser.add_argument("--smooth-confirm-frames", type=int, default=2, help="Consecutive frames required before switching to a new recognized class.")
    parser.add_argument("--smooth-unknown-hold-frames", type=int, default=4, help="Consecutive unknown frames required before dropping a stable result to unknown.")
    parser.add_argument("--refresh-ms", type=int, default=30, help="GUI refresh interval in milliseconds.")
    parser.add_argument("--preview-width", type=int, default=640, help="Preview width in the GUI.")
    parser.add_argument("--mode", choices=("webcam", "usbcam"), default="usbcam", help="Camera opening mode.")
    return parser.parse_args()


class RealtimeRecognizerApp:
    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root = root
        self.args = args
        self.model = load_model(args.model)
        self.capture: cv2.VideoCapture | None = None
        self.overlay_image: ImageTk.PhotoImage | None = None
        self.mask_image: ImageTk.PhotoImage | None = None
        self.last_result: dict | None = None
        self.smoother = TemporalResultSmoother(args.smooth_confirm_frames, args.smooth_unknown_hold_frames)
        self.last_tick = time.time()
        self.last_fps = 0.0
        self.frame_index = 0
        self.control_vars: dict[str, tk.DoubleVar] = {}
        self.control_value_labels: dict[str, ttk.Label] = {}

        self.status_var = tk.StringVar(value="Ready")
        self.class_var = tk.StringVar(value="-")
        self.conf_var = tk.StringVar(value="-")
        self.top1_var = tk.StringVar(value="-")
        self.top1_conf_var = tk.StringVar(value="-")
        self.fps_var = tk.StringVar(value="FPS -")

        self.root.title("Printed Marker Recognizer")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main = ttk.Frame(self.root, padding=12)
        main.grid(sticky="nsew")
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)

        preview_frame = ttk.Frame(main)
        preview_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.columnconfigure(1, weight=1)
        preview_frame.rowconfigure(1, weight=1)

        ttk.Label(preview_frame, text="Overlay").grid(row=0, column=0, sticky="w")
        ttk.Label(preview_frame, text="Red mask").grid(row=0, column=1, sticky="w")
        self.preview_label = ttk.Label(preview_frame)
        self.preview_label.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        self.mask_label = ttk.Label(preview_frame)
        self.mask_label.grid(row=1, column=1, sticky="nsew")

        info_frame = ttk.Frame(preview_frame)
        info_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        info_frame.columnconfigure(0, weight=1)
        info_frame.columnconfigure(1, weight=1)
        info_frame.columnconfigure(2, weight=1)
        info_frame.columnconfigure(3, weight=1)
        info_frame.columnconfigure(4, weight=1)
        ttk.Label(info_frame, text="Status").grid(row=0, column=0, sticky="w")
        ttk.Label(info_frame, text="Final").grid(row=0, column=1, sticky="w")
        ttk.Label(info_frame, text="Final conf").grid(row=0, column=2, sticky="w")
        ttk.Label(info_frame, text="Top1").grid(row=0, column=3, sticky="w")
        ttk.Label(info_frame, text="Top1 conf").grid(row=0, column=4, sticky="w")
        ttk.Label(info_frame, textvariable=self.status_var).grid(row=1, column=0, sticky="w")
        ttk.Label(info_frame, textvariable=self.class_var).grid(row=1, column=1, sticky="w")
        ttk.Label(info_frame, textvariable=self.conf_var).grid(row=1, column=2, sticky="w")
        ttk.Label(info_frame, textvariable=self.top1_var).grid(row=1, column=3, sticky="w")
        ttk.Label(info_frame, textvariable=self.top1_conf_var).grid(row=1, column=4, sticky="w")
        ttk.Label(info_frame, textvariable=self.fps_var).grid(row=2, column=0, sticky="w", pady=(8, 0))

        controls_frame = ttk.Frame(main)
        controls_frame.grid(row=0, column=1, sticky="nsew")
        controls_frame.columnconfigure(0, weight=1)

        button_row = ttk.Frame(controls_frame)
        button_row.grid(row=0, column=0, sticky="w")
        self.start_button = ttk.Button(button_row, text="Start", command=self.start_camera)
        self.start_button.grid(row=0, column=0, padx=(0, 8))
        self.stop_button = ttk.Button(button_row, text="Stop", command=self.stop_camera)
        self.stop_button.grid(row=0, column=1, padx=(0, 8))
        ttk.Button(button_row, text="Exit", command=self.on_close).grid(row=0, column=2)

        tuning_frame = ttk.LabelFrame(controls_frame, text="Realtime tuning", padding=10)
        tuning_frame.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        tuning_frame.columnconfigure(1, weight=1)

        control_specs = [
            ("camera_brightness", "Brightness", 0.0, 255.0, self.args.camera_brightness if self.args.camera_brightness is not None else 80.0, False),
            ("camera_exposure", "Exposure", -13.0, 0.0, self.args.camera_exposure if self.args.camera_exposure is not None else -6.0, False),
            ("camera_gain", "Gain", 0.0, 255.0, self.args.camera_gain if self.args.camera_gain is not None else 32.0, False),
            ("red_h_low1", "Red H low1", 0.0, 30.0, float(self.args.red_h_low1), True),
            ("red_h_high1", "Red H high1", 0.0, 30.0, float(self.args.red_h_high1), True),
            ("red_h_low2", "Red H low2", 150.0, 180.0, float(self.args.red_h_low2), True),
            ("red_h_high2", "Red H high2", 150.0, 180.0, float(self.args.red_h_high2), True),
            ("red_s_min", "Red S min", 0.0, 255.0, float(self.args.red_s_min), True),
            ("red_v_min", "Red V min", 0.0, 255.0, float(self.args.red_v_min), True),
            ("min_red_area", "Min red area", 1.0, 1000.0, float(self.args.min_red_area), False),
            ("morph_kernel", "Morph kernel", 1.0, 15.0, float(self.args.morph_kernel), True),
        ]

        for row, (name, label, start, end, value, integer_only) in enumerate(control_specs):
            self.add_slider(tuning_frame, row, name, label, start, end, value, integer_only)

    def add_slider(
        self,
        parent: ttk.Widget,
        row: int,
        name: str,
        label: str,
        start: float,
        end: float,
        value: float,
        integer_only: bool,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        var = tk.DoubleVar(value=value)
        self.control_vars[name] = var
        scale = ttk.Scale(parent, from_=start, to=end, variable=var)
        scale.grid(row=row, column=1, sticky="ew", padx=8, pady=2)
        value_label = ttk.Label(parent, width=8)
        value_label.grid(row=row, column=2, sticky="e")
        self.control_value_labels[name] = value_label
        self.update_control_label(name, value, integer_only)
        scale.configure(command=lambda raw, n=name, i=integer_only: self.on_slider_change(n, raw, i))

    def on_slider_change(self, name: str, raw_value: str, integer_only: bool) -> None:
        value = float(raw_value)
        if integer_only:
            value = round(value)
            self.control_vars[name].set(value)
        self.update_arg_value(name, value, integer_only)
        self.update_control_label(name, value, integer_only)
        if name in CAMERA_CONTROL_PROPS:
            self.apply_camera_control(name)

    def update_arg_value(self, name: str, value: float, integer_only: bool) -> None:
        if integer_only:
            setattr(self.args, name, int(round(value)))
        else:
            setattr(self.args, name, float(value))

    def update_control_label(self, name: str, value: float, integer_only: bool) -> None:
        text = f"{int(round(value))}" if integer_only else f"{value:.1f}"
        self.control_value_labels[name].configure(text=text)

    def apply_camera_control(self, name: str) -> None:
        if self.capture is None:
            return
        value = getattr(self.args, name)
        if value is None:
            return
        self.capture.set(CAMERA_CONTROL_PROPS[name], float(value))

    def start_camera(self) -> None:
        if self.capture is not None:
            return
        try:
            self.capture = open_camera(self.args)
        except Exception as exc:
            self.status_var.set(str(exc))
            return
        self.status_var.set("Camera started")
        self.smoother = TemporalResultSmoother(self.args.smooth_confirm_frames, self.args.smooth_unknown_hold_frames)
        self.last_tick = time.time()
        self.frame_index = 0
        self.update_frame()

    def stop_camera(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.status_var.set("Camera stopped")

    def update_frame(self) -> None:
        if self.capture is None:
            return

        ok, frame = self.capture.read()
        if not ok:
            self.status_var.set("Failed to read camera frame")
            self.stop_camera()
            return

        self.frame_index += 1
        if self.last_result is None or self.frame_index % max(1, self.args.frame_skip) == 0:
            raw_result, _ = recognize_frame(frame, self.model, self.args)
            result = self.smoother.apply(raw_result)
            overlay = draw_overlay(frame, result)
            self.last_result = result
        else:
            result = self.last_result
            overlay = draw_overlay(frame, result)

        now = time.time()
        self.last_fps = 1.0 / max(now - self.last_tick, 1e-6)
        self.last_tick = now
        self.fps_var.set(f"FPS {self.last_fps:.1f}")
        self.update_result_text(result)
        self.update_preview(self.preview_label, overlay, is_mask=False)
        self.update_preview(self.mask_label, result.get("mask"), is_mask=True)
        self.root.after(max(1, self.args.refresh_ms), self.update_frame)

    def update_result_text(self, result: dict) -> None:
        if result.get("success"):
            self.status_var.set(result.get("warning", "OK"))
            self.class_var.set(str(result.get("class_name", "-")))
            self.conf_var.set(f"{result.get('confidence', 0.0):.3f}")
            self.top1_var.set(str(result.get("top1_class_name", "-")))
            self.top1_conf_var.set(f"{result.get('top1_confidence', 0.0):.3f}")
        else:
            self.status_var.set(str(result.get("error", "failed")))
            self.class_var.set("-")
            self.conf_var.set("-")
            self.top1_var.set("-")
            self.top1_conf_var.set("-")

    def update_preview(self, label: ttk.Label, frame, is_mask: bool) -> None:
        if frame is None:
            return
        if is_mask:
            image = Image.fromarray(frame)
            image = image.convert("L").convert("RGB")
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
        preview_width = max(1, self.args.preview_width)
        if image.width > preview_width:
            preview_height = round(image.height * preview_width / image.width)
            image = image.resize((preview_width, preview_height))
        photo = ImageTk.PhotoImage(image=image)
        label.configure(image=photo)
        if is_mask:
            self.mask_image = photo
        else:
            self.overlay_image = photo

    def on_close(self) -> None:
        self.stop_camera()
        self.root.destroy()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    app = RealtimeRecognizerApp(root, args)
    app.start_camera()
    root.mainloop()


if __name__ == "__main__":
    main()
