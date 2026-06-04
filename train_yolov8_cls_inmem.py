"""
train_yolov8_cls_inmem.py — 零磁盘 IO 的 YOLOv8 分类训练

整合了 prepare_yolo_cls_dataset.py 的数据集发现/划分/增强 + train_yolov8_cls.py 的训练流程，
数据在内存中完成预处理后直接喂给 PyTorch Dataset，不经过磁盘 IO（仅读取原始图片）。

用法：
    python train_yolov8_cls_inmem.py [--source-root .] [--model yolov8n-cls.pt]
                                     [--epochs 100] [--imgsz 100] [--batch 64]
                                     [--device cpu] [--workers 0]

注意：
    Ultralytics 的 model.val() 仍需文件系统路径，因此 val/test 数据仍然会写入磁盘。
    如果只想纯内存训练且不验证，训练完后删除临时目录即可。
"""

import argparse
import json
import math
import random
import shutil
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import Dataset, DataLoader


# ============================================================
#  常量 — 与 prepare_yolo_cls_dataset.py 保持一致
# ============================================================

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXCLUDED_DIRS = {
    ".claude", "verify_test",
    "yolo_cls_dataset", "yolo_cls_dataset_aug100", "yolo_cls_dataset_camstyle",
    "runs", "runs_printed_marker", "runs_yolov5_cls", "runs_yolov8_cls",
    "yolov5", "__pycache__",
}

# 数据增强参数（与磁盘版一致，可自行调整）
ROTATION_RANGE = (-10.0, 10.0)
CROP_SCALE_RANGE = (0.9, 1.1)
TRANSLATION_RATIO = 0.03
BRIGHTNESS_RANGE = (0.3, 0.8)
CONTRAST_RANGE = (2, 5)
NOISE_AMOUNT_RANGE = (22, 50)
BLUR_RADIUS_RANGE = (6, 15)


# ============================================================
#  数据集发现与划分（复用 prepare_yolo_cls_dataset.py 的逻辑）
# ============================================================

def should_skip_dir(path: Path) -> bool:
    name = path.name
    return name in EXCLUDED_DIRS or name.startswith("runs") or name.startswith(".")


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def discover_class_dirs(source_root: Path) -> dict[str, list[Path]]:
    """
    扫描源目录，发现所有类别及其图片列表。
    目录结构要求：source_root/一级分类/二级类别（最终分类名）/图片
    """
    result: dict[str, list[Path]] = {}
    dir_map: dict[str, Path] = {}
    for top in sorted(p for p in source_root.iterdir() if p.is_dir()):
        if should_skip_dir(top):
            continue
        for cls_dir in sorted(p for p in top.iterdir() if p.is_dir()):
            if should_skip_dir(cls_dir):
                continue
            imgs = sorted(p for p in cls_dir.iterdir() if is_image(p))
            if not imgs:
                continue
            name = cls_dir.name
            if name in result:
                raise ValueError(f"Duplicate class name: {name}\n{dir_map[name]}\n{cls_dir}")
            result[name] = imgs
            dir_map[name] = cls_dir
    if not result:
        raise ValueError(f"No class directories found under: {source_root}")
    return result


def split_images(
    images: list[Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[Path]]:
    """按比例随机划分 train/val/test，处理边界情况"""
    shuffled = list(images)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)

    if total == 1:
        return {"train": shuffled, "val": [], "test": []}
    if total == 2:
        return {"train": shuffled[:1], "val": shuffled[1:], "test": []}

    train_count = max(1, int(total * train_ratio))
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count

    if val_ratio > 0 and val_count == 0:
        val_count = 1
    if test_ratio > 0 and test_count == 0:
        test_count = 1
    while train_count + val_count + test_count > total:
        if train_count >= val_count and train_count > 1:
            train_count -= 1
        elif val_count >= test_count and val_count > 0:
            val_count -= 1
        elif test_count > 0:
            test_count -= 1
        else:
            break

    te = train_count
    ve = te + val_count
    return {"train": shuffled[:te], "val": shuffled[te:ve], "test": shuffled[ve:]}


# ============================================================
#  内存中数据增强（与 prepare_yolo_cls_dataset.py 相同）
# ============================================================

def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(v, hi))


def apply_augmentation(image: Image.Image, rng: random.Random, target_size: tuple[int, int]) -> np.ndarray:
    """
    在 PIL Image 上执行数据增强，返回 CHW uint8 numpy 数组（值范围 0-255）。
    增强流程：旋转 → 随机裁切 → 亮度 → 对比度 → 噪声 → 模糊 → resize
    """
    # 旋转
    image = image.rotate(
        rng.uniform(*ROTATION_RANGE),
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=(0, 0, 0),
    )
    # 随机裁切
    w, h = image.size
    cs = rng.uniform(*CROP_SCALE_RANGE)
    cw = max(1, round(w * cs))
    ch = max(1, round(h * cs))
    sx = round(w * TRANSLATION_RATIO)
    sy = round(h * TRANSLATION_RATIO)
    cx = w // 2 + rng.randint(-sx, sx)
    cy = h // 2 + rng.randint(-sy, sy)
    left = clamp(cx - cw // 2, 0, w - cw)
    top = clamp(cy - ch // 2, 0, h - ch)
    image = image.crop((left, top, left + cw, top + ch))

    # 亮度 / 对比度
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(*BRIGHTNESS_RANGE))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(*CONTRAST_RANGE))

    # 噪声
    noise_amt = rng.randint(*NOISE_AMOUNT_RANGE)
    noise = Image.effect_noise(image.size, noise_amt).convert("L")
    image = Image.blend(image, Image.merge("RGB", (noise, noise, noise)), 0.18)

    # 模糊
    image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(*BLUR_RADIUS_RANGE)))

    # resize 到目标尺寸 → 转为 CHW uint8
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    arr = np.asarray(image, dtype=np.uint8)           # HWC
    return np.transpose(arr, (2, 0, 1))               # CHW


# ============================================================
#  自定义 PyTorch Dataset — 在内存中实时做数据增强
# ============================================================

class InMemoryClsDataset(Dataset):
    """
    完全在内存中读取和增强图片的 PyTorch Dataset。
    
    - __getitem__ 返回 (CHW_uint8_tensor, class_index)
    - 每个 epoch 内对同一张图片的增强结果都不同（因 random seed 不同）
    - 图片原文件只读一次到 PIL Image 后 cache 住，增强在 cache 上反复应用
    """

    def __init__(
        self,
        image_paths: list[Path],
        class_names: list[str],
        target_size: tuple[int, int],
        augment: bool = True,
        seed: int = 42,
    ) -> None:
        self.image_paths = image_paths
        self.class_names = class_names
        self.name_to_idx = {n: i for i, n in enumerate(class_names)}
        self.target_size = target_size
        self.augment = augment
        self.base_seed = seed

        # 标签
        self.labels = [self.name_to_idx[p.parent.name] for p in image_paths]

        # 预加载所有图片到内存（PIL Image 格式，节省每次重新解码的开销）
        self._images: list[Image.Image] = []
        for p in image_paths:
            with Image.open(p) as img:
                img = ImageOps.exif_transpose(img).convert("RGB")
                self._images.append(img.copy())

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        img = self._images[index]
        label = self.labels[index]
        if self.augment:
            # 每个 epoch 内不同增强：用 index+epoch 做随机种子（epoch 由外部设置）
            epoch = getattr(self, "_epoch", 0)
            rng = random.Random(self.base_seed + index * 1000003 + epoch)
            arr = apply_augmentation(img, rng, self.target_size)
        else:
            # 验证/测试：只 resize，不做增强
            arr = np.asarray(
                img.resize(self.target_size, Image.Resampling.LANCZOS),
                dtype=np.uint8,
            )
            arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr).float() / 255.0, label

    def set_epoch(self, epoch: int) -> None:
        """供外部在每个 epoch 前设置，使增强结果随 epoch 变化"""
        self._epoch = epoch


# ============================================================
#  整合训练入口
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLOv8 分类训练（零磁盘 IO 版）—— 数据集发现/增强/训练全在内存中完成"
    )
    # 数据集参数
    parser.add_argument("--source-root", type=Path, default=Path("."),
                        help="原始数据集根目录（包含 一级分类/二级类别/图片）")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="测试集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--overwrite", action="store_true",
                        help="删除已有的临时磁盘目录后重新创建（用于 val/test 写入）")

    # 训练参数
    parser.add_argument("--model", default="yolov8n-cls.pt",
                        help="YOLOv8 分类模型或检查点路径")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=100, help="输入图片尺寸（宽=高）")
    parser.add_argument("--batch", type=int, default=64, help="批次大小（可开大，因为无磁盘瓶颈）")
    parser.add_argument("--device", default="cpu", help="训练设备：cpu、0（GPU）等")
    parser.add_argument("--workers", type=int, default=0,
                        help="DataLoader 子进程数。Windows 建议 0")
    parser.add_argument("--project", type=Path, default=Path("runs_yolov8_cls"),
                        help="输出目录")
    parser.add_argument("--name", default="zoumaguangbei_cls_inmem",
                        help="本次训练的 run 名称")
    return parser.parse_args()


def build_class_to_images(source_root: Path, args: argparse.Namespace) -> dict[str, list[Path]]:
    """发现类别并划分，返回 {类别名: [图片路径]}"""
    class_to_images = discover_class_dirs(source_root)
    print(f"Found {len(class_to_images)} classes:")
    for name, imgs in sorted(class_to_images.items()):
        print(f"  {name}: {len(imgs)} images")
    return class_to_images


def build_splits(
    class_to_images: dict[str, list[Path]],
    args: argparse.Namespace,
) -> dict[str, dict[str, list[Path]]]:
    """
    划分数据集。

    返回: {类别名: {"train": [paths], "val": [paths], "test": [paths]}}
    """
    splits: dict[str, dict[str, list[Path]]] = {}
    for idx, (name, imgs) in enumerate(sorted(class_to_images.items())):
        splits[name] = split_images(imgs, args.train_ratio, args.val_ratio,
                                     args.test_ratio, args.seed + idx)
    return splits


def prepare_val_test_on_disk(
    class_splits: dict[str, dict[str, list[Path]]],
    target_size: tuple[int, int],
    temp_dir: Path,
    overwrite: bool,
) -> Path:
    """
    val/test 数据写入临时目录（Ultralytics model.val() 需要文件系统路径）。
    train 数据不写入磁盘（由 InMemoryClsDataset 提供）。
    
    返回 temp_dir 路径。
    """
    if temp_dir.exists():
        if overwrite:
            shutil.rmtree(temp_dir)
        else:
            return temp_dir

    for split_name in ("val", "test"):
        for class_name, splits in class_splits.items():
            paths = splits.get(split_name, [])
            if not paths:
                continue
            out_dir = temp_dir / split_name / class_name
            out_dir.mkdir(parents=True, exist_ok=True)
            rng = random.Random(42)  # 确定性增强
            for src in paths:
                with Image.open(src) as img:
                    img = ImageOps.exif_transpose(img).convert("RGB")
                    # 对 val/test 也做相同增强，保持一致性
                    arr = apply_augmentation(img, rng, target_size)
                    arr_hwc = np.transpose(arr, (1, 2, 0))
                    Image.fromarray(arr_hwc).save(out_dir / src.name, quality=95)
            print(f"  {split_name}/{class_name}: {len(paths)} images -> {out_dir}")
    return temp_dir


def write_training_summary(temp_dir: Path | None, args: argparse.Namespace,
                           class_splits: dict[str, dict[str, list[Path]]]) -> None:
    """将训练配置写入 summary JSON"""
    summary = {
        "source_root": str(Path(".").resolve()),
        "temp_dir": str(temp_dir) if temp_dir else "(in-memory only)",
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "seed": args.seed,
        "training": {
            "model": args.model,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
        },
        "augmentation": {
            "output_size": [args.imgsz, args.imgsz],
            "rotation_range_degrees": list(ROTATION_RANGE),
            "crop_scale_range": list(CROP_SCALE_RANGE),
            "translation_ratio": TRANSLATION_RATIO,
            "brightness_range": list(BRIGHTNESS_RANGE),
            "contrast_range": list(CONTRAST_RANGE),
            "noise_amount_range": list(NOISE_AMOUNT_RANGE),
            "blur_radius_range": list(BLUR_RADIUS_RANGE),
        },
        "classes": [
            {
                "name": name,
                "total": sum(len(v) for v in splits.values()),
                "splits": {k: len(v) for k, v in splits.items()},
            }
            for name, splits in sorted(class_splits.items())
        ],
    }
    summary_path = Path(args.project) / args.name / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary: {summary_path}")


def main() -> None:
    args = parse_args()

    # --------------------------------------------------
    # 1. 发现类别 & 划分
    # --------------------------------------------------
    source_root = Path(args.source_root).expanduser().resolve()
    print(f"Scanning: {source_root}")
    class_to_images = build_class_to_images(source_root, args)
    class_splits = build_splits(class_to_images, args)
    class_names = sorted(class_splits.keys())

    # --------------------------------------------------
    # 2. 构建内存训练集
    # --------------------------------------------------
    train_paths: list[Path] = []
    for name in class_names:
        train_paths.extend(class_splits[name].get("train", []))
    if not train_paths:
        raise ValueError("No training images found!")

    target_size = (args.imgsz, args.imgsz)
    train_dataset = InMemoryClsDataset(train_paths, class_names, target_size,
                                       augment=True, seed=args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(args.device != "cpu"),
    )
    print(f"Train: {len(train_dataset)} images ({len(class_names)} classes), "
          f"augmented in memory (no disk writes)")

    # --------------------------------------------------
    # 3. val/test 写入临时目录（Ultralytics 需要文件路径）
    # --------------------------------------------------
    temp_dir = Path(tempfile.gettempdir()) / f"yolo_cls_temp_{args.name}"
    prepare_val_test_on_disk(class_splits, target_size, temp_dir, args.overwrite)

    # --------------------------------------------------
    # 4. 训练
    # --------------------------------------------------
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("pip install ultralytics") from exc

    model = YOLO(args.model)

    # 注册自定义 Dataset 到 Ultralytics（通过 monkey-patch data loader）
    # 但更干净的方式：直接使用 PyTorch 的训练循环 + Ultralytics 的模型结构/损失
    #
    # 方案 A（推荐）：用 ultralytics 的 model.val() 做验证，训练则用自定义循环
    # 方案 B：完全用 ultralytics model.train(data=路径)，但 train 数据也写磁盘
    #
    # 下面实现方案 A。
    # --------------------------------------------------

    device = torch.device(args.device if args.device != "cpu" else "cpu")
    model.model.to(device)
    model.model.train()

    # 分类损失
    criterion = torch.nn.CrossEntropyLoss()

    # 优化器（使用 Ultralytics 内置的优化器逻辑太重，自己配简单的）
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=0.001, weight_decay=5e-4)

    # 学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 训练循环
    print(f"\nTraining on {device} ...")
    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)   # (B, C, H, W), float32, [0,1]
            labels = labels.to(device)

            optimizer.zero_grad()
            # model(inputs) 返回 ultralytics 的 results 对象，需要取 logits
            # 用 model.model(inputs) 直接拿到裸 logits
            logits = model.model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        epoch_loss = running_loss / total
        epoch_acc = correct / total * 100.0
        print(f"  Epoch {epoch:3d}/{args.epochs}  loss={epoch_loss:.4f}  acc={epoch_acc:.2f}%  "
              f"lr={scheduler.get_last_lr()[0]:.6f}")

        # 每 10 个 epoch 做一次验证
        if epoch % 10 == 0 or epoch == args.epochs:
            val_results = model.val(data=str(temp_dir), split="val",
                                    batch=args.batch, imgsz=args.imgsz,
                                    device=args.device, workers=0, verbose=False)
            if hasattr(val_results, "top1"):
                print(f"    Val top1={val_results.top1:.2f}%")

    # --------------------------------------------------
    # 5. 保存模型 & 最终验证
    # --------------------------------------------------
    project_dir = Path(args.project).expanduser().resolve()
    run_dir = project_dir / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    # 保存权重
    ckpt_path = run_dir / "best.pt"
    torch.save(model.model.state_dict(), ckpt_path)
    print(f"Model saved: {ckpt_path}")

    # 最终验证
    final_results = model.val(data=str(temp_dir), split="test",
                              batch=args.batch, imgsz=args.imgsz,
                              device=args.device, workers=0, verbose=False)
    top1 = getattr(final_results, "top1", "N/A")
    top5 = getattr(final_results, "top5", "N/A")
    print(f"Test top1={top1}%  top5={top5}%")

    # 写入总结
    write_training_summary(temp_dir, args, class_splits)

    # 清理临时目录
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        print(f"Temp dir cleaned: {temp_dir}")


if __name__ == "__main__":
    main()
