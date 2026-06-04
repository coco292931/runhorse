import argparse
from pathlib import Path


# 支持的图片文件扩展名
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLOv8 classification model.")
    parser.add_argument("--data", type=Path, default=Path("yolo_cls_dataset_camstyle"),
                        help="数据集根目录，需包含 train/val/test 子目录")
    parser.add_argument("--model", default="yolov8n-cls.pt",
                        help="YOLOv8 分类模型或检查点路径（默认从官方下载 yolov8n-cls.pt）")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=100, help="输入图片尺寸（宽=高）")
    parser.add_argument("--batch", type=int, default=16, help="批次大小")
    parser.add_argument("--device", default="cpu", help="训练设备：cpu、0（GPU）、0,1（多卡）等")
    parser.add_argument("--project", type=Path, default=Path("runs_yolov8_cls"),
                        help="训练结果输出目录")
    parser.add_argument("--name", default="zoumaguangbei_cls_camstyle",
                        help="本次训练的 run 名称，结果保存在 {project}/{name}/")
    parser.add_argument("--workers", type=int, default=0,
                        help="数据加载线程数。Windows 上设为 0 更安全")
    parser.add_argument("--lr0", type=float, default=None,
                        help="初始学习率。不指定则使用 ultralytics 默认值（0.01）。batch 增大时应等比增大 lr")
    parser.add_argument("--cache", default=None, choices=("ram", "disk"),
                        help="将数据集缓存到 RAM 或磁盘以减少磁盘 I/O。不指定则不缓存")
    return parser.parse_args()


def count_images(class_dir: Path) -> int:
    """统计一个类别目录中的图片文件数量"""
    return sum(1 for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def validate_split(split_dir: Path) -> list[str]:
    """
    验证一个 split 目录（train/val/test）的合法性，返回所有类别名。

    检查：
    - split 目录存在
    - 每个类别子目录中至少有一张图片
    - 至少有一个类别目录
    """
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
    """验证整个数据集：train 和 val 的类别列表必须完全一致"""
    train_classes = validate_split(data_dir / "train")
    val_classes = validate_split(data_dir / "val")
    if train_classes != val_classes:
        raise ValueError(f"Train/val classes differ. train={train_classes}, val={val_classes}")


def train_yolov8(args: argparse.Namespace) -> None:
    """
    主训练流程：
    1. 导入 ultralytics（检查依赖）
    2. 解析并验证数据集路径
    3. 校验数据集合法性
    4. 加载 YOLOv8 分类模型（支持预训练权重或已有检查点）
    5. 启动训练
    """
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
        **(dict(lr0=args.lr0) if args.lr0 is not None else {}),
        **(dict(cache=args.cache) if args.cache is not None else {}),
    )


if __name__ == "__main__":
    train_yolov8(parse_args())
