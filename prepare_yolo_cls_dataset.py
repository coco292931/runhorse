import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
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
TARGET_SIZE = (100, 100)
ROTATION_RANGE = (-10.0, 10.0)
CROP_SCALE_RANGE = (0.9, 1.1)
TRANSLATION_RATIO = 0.03
BRIGHTNESS_RANGE = (0.5, 0.8)
CONTRAST_RANGE = (1.2, 1.6)
NOISE_AMOUNT_RANGE = (6, 22)
BLUR_RADIUS_RANGE = (0.8, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare YOLO classification train/val/test dataset.")
    parser.add_argument("--source-root", type=Path, default=Path("."), help="Original dataset root.")
    parser.add_argument("--output-root", type=Path, default=Path("yolo_cls_dataset_camstyle"), help="Output split dataset root.")
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
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if train_ratio <= 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError("Ratios must satisfy: train > 0, val >= 0, test >= 0.")
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total:.6f}.")


def is_image(path: Path, image_exts: set[str]) -> bool:
    return path.is_file() and path.suffix.lower() in image_exts


def should_skip_dir(path: Path) -> bool:
    name = path.name
    return name in EXCLUDED_DIRS or name.startswith("runs") or name.startswith(".")


def discover_class_dirs(source_root: Path, image_exts: set[str]) -> dict[str, list[Path]]:
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

    train_end = train_count
    val_end = train_end + val_count
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def random_crop(image: Image.Image, rng: random.Random) -> Image.Image:
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
    noise_amount = rng.randint(*NOISE_AMOUNT_RANGE)
    noise = Image.effect_noise(image.size, noise_amount).convert("L")
    return Image.blend(image, Image.merge("RGB", (noise, noise, noise)), 0.18)


def preprocess_image(source: Path, rng: random.Random) -> Image.Image:
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.rotate(
            rng.uniform(*ROTATION_RANGE),
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=(0, 0, 0),
        )
        image = random_crop(image, rng)
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(*BRIGHTNESS_RANGE))
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(*CONTRAST_RANGE))
        image = add_noise(image, rng)
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(*BLUR_RADIUS_RANGE)))
        return image.resize(TARGET_SIZE, Image.Resampling.LANCZOS)


def save_processed_image(image: Image.Image, target: Path) -> None:
    suffix = target.suffix.lower()
    save_kwargs: dict[str, object] = {}
    if suffix in {".jpg", ".jpeg"}:
        save_kwargs = {"quality": 95}
    elif suffix == ".webp":
        save_kwargs = {"quality": 95}
    image.save(target, **save_kwargs)


def preprocess_and_save(source: Path, target: Path, rng: random.Random) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    processed_image = preprocess_image(source, rng)
    save_processed_image(processed_image, target)


def create_split_dataset(
    class_to_images: dict[str, list[Path]],
    output_root: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    overwrite: bool,
) -> dict[str, dict[str, int]]:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_root}\nUse --overwrite to recreate it.")
        shutil.rmtree(output_root)

    summary: dict[str, dict[str, int]] = {}
    for class_index, (class_name, images) in enumerate(sorted(class_to_images.items())):
        splits = split_images(images, train_ratio, val_ratio, test_ratio, seed + class_index)
        summary[class_name] = {}
        for split_index, (split_name, split_images_list) in enumerate(splits.items()):
            class_output_dir = output_root / split_name / class_name
            class_output_dir.mkdir(parents=True, exist_ok=True)
            used_names: set[str] = set()
            split_rng = random.Random(seed + class_index * 1000003 + split_index)
            for image in split_images_list:
                target_name = image.name
                if target_name in used_names:
                    target_name = f"{image.stem}_{abs(hash(str(image.parent))) % 100000}{image.suffix}"
                used_names.add(target_name)
                preprocess_and_save(image, class_output_dir / target_name, split_rng)
            summary[class_name][split_name] = len(split_images_list)
    return summary


def write_summary(
    output_root: Path,
    source_root: Path,
    class_to_images: dict[str, list[Path]],
    split_counts: dict[str, dict[str, int]],
    args: argparse.Namespace,
) -> None:
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
        "preprocess": {
            "applied_splits": ["train", "val", "test"],
            "output_size": list(TARGET_SIZE),
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
    print("Dataset split finished. Classes:")
    for class_name, counts in sorted(split_counts.items()):
        print(
            f"  {class_name}: "
            f"train={counts.get('train', 0)}, "
            f"val={counts.get('val', 0)}, "
            f"test={counts.get('test', 0)}"
        )



def main() -> None:
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

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
    )
    write_summary(output_root, source_root, class_to_images, split_counts, args)
    print_summary(split_counts)
    print(f"Output: {output_root}")
    print(f"Summary: {output_root / 'split_summary.json'}")


if __name__ == "__main__":
    main()
