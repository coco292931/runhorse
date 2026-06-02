import argparse
import subprocess
import sys
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLOv5 classification model.")
    parser.add_argument("--data", type=Path, default=Path("yolo_cls_dataset"), help="Dataset root with train/val/test folders.")
    parser.add_argument("--yolov5-dir", type=Path, default=Path("yolov5"), help="YOLOv5 repository directory.")
    parser.add_argument("--model", default="yolov5s-cls.pt", help="YOLOv5 classification model or checkpoint.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=224, help="Image size.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", default="cpu", help="Device: cpu, 0, 0,1, etc.")
    parser.add_argument("--project", type=Path, default=Path("runs_yolov5_cls"), help="Output project directory.")
    parser.add_argument("--name", default="zoumaguangbei_cls", help="Run name.")
    parser.add_argument("--workers", type=int, default=0, help="Data loader workers. 0 is safer on Windows.")
    return parser.parse_args()


def count_images(class_dir: Path) -> int:
    return sum(1 for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def validate_split(split_dir: Path) -> list[str]:
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")
    classes = []
    for class_dir in sorted((p for p in split_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
        image_count = count_images(class_dir)
        if image_count == 0:
            raise ValueError(f"No images found in class directory: {class_dir}")
        classes.append(class_dir.name)
    if not classes:
        raise ValueError(f"No class directories found in: {split_dir}")
    return classes


def validate_dataset(data_dir: Path) -> None:
    train_classes = validate_split(data_dir / "train")
    val_classes = validate_split(data_dir / "val")
    if train_classes != val_classes:
        raise ValueError(f"Train/val classes differ. train={train_classes}, val={val_classes}")


def validate_yolov5_repo(yolov5_dir: Path) -> Path:
    train_script = yolov5_dir / "classify" / "train.py"
    if not train_script.exists():
        raise FileNotFoundError(
            "YOLOv5 classification train.py was not found.\n"
            f"Expected: {train_script}\n\n"
            "Prepare YOLOv5 first, for example:\n"
            "  git clone https://github.com/ultralytics/yolov5.git C:\\yolo\\yolov5\n"
            "  python -m pip install -r C:\\yolo\\yolov5\\requirements.txt\n\n"
            "Then run this script with:\n"
            "  python .\\train_yolov5_cls.py --yolov5-dir C:\\yolo\\yolov5"
        )
    return train_script


def build_yolov5_command(args: argparse.Namespace, train_script: Path, data_dir: Path, project_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(train_script),
        "--model",
        args.model,
        "--data",
        str(data_dir),
        "--epochs",
        str(args.epochs),
        "--img",
        str(args.imgsz),
        "--batch-size",
        str(args.batch_size),
        "--device",
        str(args.device),
        "--project",
        str(project_dir),
        "--name",
        args.name,
        "--workers",
        str(args.workers),
    ]


def train_yolov5(args: argparse.Namespace) -> None:
    data_dir = args.data.expanduser().resolve()
    yolov5_dir = args.yolov5_dir.expanduser().resolve()
    project_dir = args.project.expanduser().resolve()

    validate_dataset(data_dir)
    train_script = validate_yolov5_repo(yolov5_dir)
    cmd = build_yolov5_command(args, train_script, data_dir, project_dir)
    print("Running:")
    print(" ".join(f'"{item}"' if " " in item else item for item in cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    train_yolov5(parse_args())
