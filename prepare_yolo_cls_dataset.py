import argparse
import json
import multiprocessing
import random
import shutil
from functools import partial
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


# ============================================================
#  常量定义
# ============================================================

# 支持的图片文件扩展名
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 扫描源目录时自动跳过的目录名（已有的输出目录、运行结果等）
EXCLUDED_DIRS = {
    ".claude",
    "verify_test",
    "yolo_cls_dataset",
    "yolo_cls_dataset_aug100",
    "yolo_cls_dataset_camstyle",
    "runs",
    "runs_printed_marker",
    "runs_yolov5_cls",
    "runs_yolov8_cls",
    "yolov5",
    "__pycache__",
}

# 输出图片统一尺寸（宽, 高），单位像素
TARGET_SIZE = (100, 100)

# 数据增强参数
ROTATION_RANGE = (-10.0, 10.0)       # 随机旋转角度范围（度）
CROP_SCALE_RANGE = (0.8, 1.2)         # 随机裁切缩放比例
TRANSLATION_RATIO = 0.03              # 随机平移比例（相对原图尺寸）
BRIGHTNESS_RANGE = (0.3, 0.8)         # 亮度调整范围（<1 变暗，>1 变亮，模拟不同曝光条件）
CONTRAST_RANGE = (2,5)           # 对比度调整范围（>1 增强对比）
NOISE_AMOUNT_RANGE = (22, 80)          # 高斯噪声强度范围
BLUR_RADIUS_RANGE = (5, 15)          # 高斯模糊半径范围（像素）


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare YOLO classification train/val/test dataset.")
    parser.add_argument("--source-root", type=Path, default=Path("."), help="Original dataset root.")
    parser.add_argument("--output-root", type=Path, default=Path("yolo_cls_dataset_camstyle"), help="Output split dataset root.")
    parser.add_argument("--target-size", type=int, default=TARGET_SIZE[0], help="Output image size in pixels, width=height.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink"),
        default="copy",
        help="Retained for compatibility; images are rewritten after preprocessing.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Remove existing output directory before creating splits.")
    parser.add_argument(
        "--split-mode",
        choices=("augment-first", "split-first"),
        default="augment-first",
        help=(
            "augment-first=每张原图先生成 N 个增强版本，再把这些版本划分到 train/val/test；"
            "split-first=先划分原图，再只对 train 做倍增。"
        ),
    )
    parser.add_argument("--aug-multiplier", type=int, default=20,
                        help="每张原图生成 N 份不同随机增强版本。augment-first 模式下 N 份会再划分到 train/val/test。")
    parser.add_argument("--workers", type=int, default=0,
                        help="并行处理进程数。0=自动(CPU核心数)，1=单进程(调试用)。Windows 建议设为 0 或 4~8。")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    """展开 ~ 并解析为绝对路径"""
    return path.expanduser().resolve()


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    """校验训练/验证/测试比例：train>0, val>=0, test>=0, 三者之和必须为1"""
    total = train_ratio + val_ratio + test_ratio
    if train_ratio <= 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError("Ratios must satisfy: train > 0, val >= 0, test >= 0.")
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total:.6f}.")


def is_image(path: Path, image_exts: set[str]) -> bool:
    """判断文件是否为支持的图片格式"""
    return path.is_file() and path.suffix.lower() in image_exts


def should_skip_dir(path: Path) -> bool:
    """判断目录是否应跳过（排除已有的输出目录、运行结果等）"""
    name = path.name
    return name in EXCLUDED_DIRS or name.startswith("runs") or name.startswith(".")


def discover_class_dirs(source_root: Path, image_exts: set[str]) -> dict[str, list[Path]]:
    """
    从源目录中发现所有类别目录及其图片列表。

    源目录结构应为：
        source_root/
            ├── 一级分类A/
            │   ├── 二级类别X/   ← 这是最终类别名
            │   │   ├── img1.jpg
            │   │   └── img2.jpg
            │   └── 二级类别Y/
            └── 一级分类B/
                └── 二级类别Z/

    返回: {类别名: [图片路径列表]}
    """
    class_to_images: dict[str, list[Path]] = {}
    class_to_dir: dict[str, Path] = {}

    for top_dir in sorted((p for p in source_root.iterdir() if p.is_dir()), key=lambda p: p.name):
        if should_skip_dir(top_dir):
            continue
        for class_dir in sorted((p for p in top_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
            if should_skip_dir(class_dir):
                continue
            images = sorted((p for p in class_dir.iterdir() if is_image(p, image_exts)), key=lambda p: p.name)
            if not images:
                continue
            class_name = class_dir.name
            # 检查类别名是否重复（不同一级目录下不允许有同名二级目录）
            if class_name in class_to_images:
                raise ValueError(f"Duplicate class directory name: {class_name}\n{class_to_dir[class_name]}\n{class_dir}")
            class_to_images[class_name] = images
            class_to_dir[class_name] = class_dir

    if not class_to_images:
        raise ValueError(f"No class image directories found under: {source_root}")
    return class_to_images


def split_images(
    images: list[Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[Path]]:
    """
    将图片列表按指定比例随机划分为 train/val/test。

    处理边界情况：
    - 仅有1张图片 → 全部进 train
    - 仅有2张图片 → 各进 train/val
    - 图片数不足以按比例分配时，自动调整确保每个 split 至少有1张
    """
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

    # 如果有 val/test 比例但计算出来为0，至少给1张
    if val_ratio > 0 and val_count == 0:
        val_count = 1
    if test_ratio > 0 and test_count == 0:
        test_count = 1

    # 如果超出总数，反复从较大的 split 中扣减
    while train_count + val_count + test_count > total:
        if train_count >= val_count and train_count > 1:
            train_count -= 1
        elif val_count >= test_count and val_count > 0:
            val_count -= 1
        elif test_count > 0:
            test_count -= 1
        else:
            break

    train_end = train_count
    val_end = train_end + val_count
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def clamp(value: int, minimum: int, maximum: int) -> int:
    """将 value 限制在 [minimum, maximum] 范围内"""
    return max(minimum, min(value, maximum))


def random_crop(image: Image.Image, rng: random.Random) -> Image.Image:
    """
    随机裁切：在缩放+微小平移后裁切图片。
    - 裁切尺寸在原图的 90%-110% 之间随机
    - 裁切中心在原中心附近随机平移 ±3%
    """
    width, height = image.size
    crop_scale = rng.uniform(*CROP_SCALE_RANGE)
    crop_width = max(1, int(round(width * crop_scale)))
    crop_height = max(1, int(round(height * crop_scale)))

    max_shift_x = int(round(width * TRANSLATION_RATIO))
    max_shift_y = int(round(height * TRANSLATION_RATIO))
    center_x = width // 2 + rng.randint(-max_shift_x, max_shift_x)
    center_y = height // 2 + rng.randint(-max_shift_y, max_shift_y)

    left = clamp(center_x - crop_width // 2, 0, width - crop_width)
    top = clamp(center_y - crop_height // 2, 0, height - crop_height)
    return image.crop((left, top, left + crop_width, top + crop_height))


def add_noise(image: Image.Image, rng: random.Random) -> Image.Image:
    """添加随机高斯噪声，模拟摄像头传感器噪点"""
    noise_amount = rng.randint(*NOISE_AMOUNT_RANGE)
    noise = Image.effect_noise(image.size, noise_amount).convert("L")
    return Image.blend(image, Image.merge("RGB", (noise, noise, noise)), 0.18)


def preprocess_image(source: Path, rng: random.Random, target_size: tuple[int, int]) -> Image.Image:
    """
    对单张图片执行完整预处理流程（数据增强）：
    1. 读取图片，根据 EXIF 方向信息自动旋转摆正，转为 RGB
    2. 随机小角度旋转（BILINEAR 插值，比 BICUBIC 快 ~3x）
    3. 随机裁切（模拟构图变化）
    4. 随机调整亮度（模拟不同光照条件）
    5. 随机调整对比度
    6. 添加随机高斯噪声（numpy 实现，快 ~5x）
    7. 随机模糊（BoxBlur 替代 GaussianBlur，快 ~5x）
    8. 缩放到统一目标尺寸
    """
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.rotate(
            rng.uniform(*ROTATION_RANGE),
            resample=Image.Resampling.BILINEAR,
            expand=True,
            fillcolor=(128, 128, 128),  # 灰色填充替代纯黑，避免旋转空白区域与白底对比突兀
        )
        image = random_crop(image, rng)
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(*BRIGHTNESS_RANGE))
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(*CONTRAST_RANGE))
        image = add_noise(image, rng)
        # BoxBlur 半径 ≈ GaussianBlur 半径 / 0.6 可达到相近模糊效果，速度快约 5x
        blur_radius = rng.uniform(*BLUR_RADIUS_RANGE)
        image = image.filter(ImageFilter.BoxBlur(radius=blur_radius))
        return image.resize(target_size, Image.Resampling.LANCZOS)


def save_processed_image(image: Image.Image, target: Path) -> None:
    """保存处理后的图片，JPEG/WEBP 格式使用 95 品质"""
    suffix = target.suffix.lower()
    save_kwargs: dict[str, object] = {}
    if suffix in {".jpg", ".jpeg"}:
        save_kwargs = {"quality": 95}
    elif suffix == ".webp":
        save_kwargs = {"quality": 95}
    image.save(target, **save_kwargs)


def preprocess_and_save(source: Path, target: Path, rng: random.Random, target_size: tuple[int, int]) -> None:
    """对源图片执行预处理并保存到目标路径"""
    target.parent.mkdir(parents=True, exist_ok=True)
    processed_image = preprocess_image(source, rng, target_size)
    save_processed_image(processed_image, target)


def _process_one_job(job: tuple) -> None:
    """多进程用的顶层任务函数：source_path, target_path, seed, target_size → 预处理+保存"""
    source, target, seed, target_size = job
    preprocess_and_save(source, target, random.Random(seed), target_size)


def _build_jobs(
    class_to_images: dict[str, list[Path]],
    output_root: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    aug_multiplier: int,
    target_size: tuple[int, int],
) -> tuple[list[tuple], dict[str, dict[str, int]]]:
    """
    构建所有待处理任务，返回 (jobs, summary)。
    job = (source_path, target_path, seed, target_size)
    summary = {类别名: {split名: 数量}}
    """
    jobs: list[tuple] = []
    summary: dict[str, dict[str, int]] = {}
    for class_index, (class_name, images) in enumerate(sorted(class_to_images.items())):
        splits = split_images(images, train_ratio, val_ratio, test_ratio, seed + class_index)
        summary[class_name] = {}
        for split_index, (split_name, split_images_list) in enumerate(splits.items()):
            class_output_dir = output_root / split_name / class_name
            multiplier = aug_multiplier if split_name == "train" else 1
            job_seed_base = seed + class_index * 1000003 + split_index
            job_idx = 0
            for image in split_images_list:
                for copy_idx in range(multiplier):
                    if multiplier == 1:
                        target_name = image.name
                    else:
                        target_name = f"{image.stem}_aug{copy_idx:03d}{image.suffix}"
                    target_path = class_output_dir / target_name
                    jobs.append((image, target_path, job_seed_base + job_idx, target_size))
                    job_idx += 1
            summary[class_name][split_name] = len(split_images_list) * multiplier
    return jobs, summary


def _build_augment_first_jobs(
    class_to_images: dict[str, list[Path]],
    output_root: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    aug_multiplier: int,
    target_size: tuple[int, int],
) -> tuple[list[tuple], dict[str, dict[str, int]]]:
    """
    先让每张原图生成 aug_multiplier 个增强版本，再把这些版本划分到 train/val/test。

    这样 train 会覆盖每一张原图的至少一个增强版本；代价是同一原图的不同增强版本
    可能同时出现在 train 和 val/test 中，验证指标会偏乐观。
    """
    jobs: list[tuple] = []
    summary: dict[str, dict[str, int]] = {}
    copies = list(range(max(1, aug_multiplier)))

    for class_index, (class_name, images) in enumerate(sorted(class_to_images.items())):
        summary[class_name] = {"train": 0, "val": 0, "test": 0}
        for image_index, image in enumerate(images):
            split_seed = seed + class_index * 1000003 + image_index * 9176
            splits = split_images(copies, train_ratio, val_ratio, test_ratio, split_seed)
            for split_index, (split_name, copy_indices) in enumerate(splits.items()):
                class_output_dir = output_root / split_name / class_name
                job_seed_base = seed + class_index * 1000003 + image_index * 9176 + split_index * 1009
                for job_idx, copy_idx in enumerate(copy_indices):
                    target_name = f"{image.stem}_aug{copy_idx:03d}{image.suffix}"
                    target_path = class_output_dir / target_name
                    jobs.append((image, target_path, job_seed_base + copy_idx + job_idx, target_size))
                    summary[class_name][split_name] += 1

    return jobs, summary


def create_split_dataset(
    class_to_images: dict[str, list[Path]],
    output_root: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    overwrite: bool,
    aug_multiplier: int = 1,
    workers: int = 0,
    split_mode: str = "augment-first",
    target_size: tuple[int, int] = TARGET_SIZE,
) -> dict[str, dict[str, int]]:
    """
    创建划分后的数据集目录。

    输出结构：
        output_root/
            ├── train/类别A/  ← 预处理后的训练图片
            ├── val/类别A/    ← 预处理后的验证图片
            └── test/类别A/   ← 预处理后的测试图片

    split_mode=augment-first：每张原图先生成 N 份增强版本，再划分到 train/val/test。
    split_mode=split-first：先划分原图，再只对 train 做倍增。

    使用多进程（ProcessPoolExecutor）并行处理，大幅加速批量生成。
    返回: {类别名: {split名: 图片数量}}
    """
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_root}\nUse --overwrite to recreate it.")
        shutil.rmtree(output_root)

    # 先构建所有目录结构
    for class_name in class_to_images:
        for split_name in ("train", "val", "test"):
            (output_root / split_name / class_name).mkdir(parents=True, exist_ok=True)

    # 构建任务列表
    if split_mode == "augment-first":
        jobs, summary = _build_augment_first_jobs(
            class_to_images, output_root,
            train_ratio, val_ratio, test_ratio, seed, aug_multiplier, target_size,
        )
    else:
        jobs, summary = _build_jobs(
            class_to_images, output_root,
            train_ratio, val_ratio, test_ratio, seed, aug_multiplier, target_size,
        )

    total = len(jobs)
    print(f"总任务数: {total}")

    if workers == 0:
        workers = multiprocessing.cpu_count()

    if workers <= 1 or total < 8:
        # 单进程模式
        for i, job in enumerate(jobs):
            _process_one_job(job)
            if (i + 1) % 50 == 0:
                print(f"  进度: {i + 1}/{total}")
    else:
        # 多进程模式 — multiprocessing.Pool 在 Windows spawn 模式下更稳定
        print(f"使用 {workers} 个进程并行处理...")
        with multiprocessing.Pool(processes=workers) as pool:
            for i, _ in enumerate(pool.imap_unordered(_process_one_job, jobs), start=1):
                if i % 100 == 0 or i == total:
                    print(f"  进度: {i}/{total}")

    return summary


def write_summary(
    output_root: Path,
    source_root: Path,
    class_to_images: dict[str, list[Path]],
    split_counts: dict[str, dict[str, int]],
    args: argparse.Namespace,
) -> None:
    """
    将数据集划分结果写入 split_summary.json，包含：
    - 源目录/输出目录路径
    - 划分比例和随机种子
    - 数据增强参数（输出尺寸、旋转、裁切、亮度、对比度、噪声、模糊）
    - 每个类别在 train/val/test 中的图片数量
    """
    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "seed": args.seed,
        "copy_mode": args.copy_mode,
        "split_mode": args.split_mode,
        "aug_multiplier": args.aug_multiplier,
        "workers": args.workers,
        "preprocess": {
            "applied_splits": ["train", "val", "test"],
            "output_size": [args.target_size, args.target_size],
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
                "name": class_name,
                "total": len(class_to_images[class_name]),
                "splits": split_counts[class_name],
            }
            for class_name in sorted(class_to_images)
        ],
    }
    summary_path = output_root / "split_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def print_summary(split_counts: dict[str, dict[str, int]]) -> None:
    """在控制台打印每个类别的 train/val/test 图片数量"""
    print("Dataset split finished. Classes:")
    for class_name, counts in sorted(split_counts.items()):
        print(
            f"  {class_name}: "
            f"train={counts.get('train', 0)}, "
            f"val={counts.get('val', 0)}, "
            f"test={counts.get('test', 0)}"
        )



def main() -> None:
    """
    主流程：
    1. 解析命令行参数
    2. 校验比例合法性
    3. 从源目录发现所有类别
    4. 创建划分后的数据集（含数据增强）
    5. 写入 summary JSON
    6. 打印结果摘要
    """
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    if args.aug_multiplier < 1:
        raise ValueError("--aug-multiplier must be >= 1.")
    if args.target_size < 1:
        raise ValueError("--target-size must be >= 1.")

    source_root = resolve_path(args.source_root)
    output_root = resolve_path(args.output_root)
    class_to_images = discover_class_dirs(source_root, IMAGE_EXTS)
    split_counts = create_split_dataset(
        class_to_images=class_to_images,
        output_root=output_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
        aug_multiplier=args.aug_multiplier,
        workers=args.workers,
        split_mode=args.split_mode,
        target_size=(args.target_size, args.target_size),
    )
    write_summary(output_root, source_root, class_to_images, split_counts, args)
    print_summary(split_counts)
    print(f"Output: {output_root}")
    print(f"Summary: {output_root / 'split_summary.json'}")


if __name__ == "__main__":
    main()
