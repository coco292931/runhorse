import argparse
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLOv8 classification model.")
    parser.add_argument("--data", type=Path, default=Path("yolo_cls_dataset_camstyle"), help="Dataset root with train/val/test folders.")
    parser.add_argument("--model", default="yolov8n-cls.pt", help="YOLOv8 classification model or checkpoint.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=100, help="Image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", default="cpu", help="Device: cpu, 0, 0,1, etc.")
    parser.add_argument("--project", type=Path, default=Path("runs_yolov8_cls"), help="Output project directory.")
    parser.add_argument("--name", default="zoumaguangbei_cls_camstyle", help="Run name.")
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


def train_yolov8(args: argparse.Namespace) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Missing dependency: ultralytics. Install it with: python -m pip install ultralytics") from exc

    data_dir = args.data.expanduser().resolve()
    project_dir = args.project.expanduser().resolve()
    validate_dataset(data_dir)

    model = YOLO(args.model)
    model.train(
        data=str(data_dir),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(project_dir),
        name=args.name,
        workers=args.workers,
    )


if __name__ == "__main__":
    train_yolov8(parse_args())
