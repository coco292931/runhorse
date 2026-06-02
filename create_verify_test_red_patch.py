import argparse
import json
import math
import random
import shutil
from pathlib import Path

from PIL import Image, ImageOps


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CM_PER_INCH = 2.54


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create printable verification images with a bottom red area.")
    parser.add_argument("--source-root", type=Path, default=Path("yolo_cls_dataset/test"), help="Source image root with class folders.")
    parser.add_argument("--output-root", type=Path, default=Path("verify_test"), help="Output folder.")
    parser.add_argument("--samples-per-class", type=int, default=1, help="Random images to select per class.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--image-size-cm", type=float, default=12.0, help="Square image area size in centimeters.")
    parser.add_argument("--red-height-cm", type=float, default=5.0, help="Bottom red area height in centimeters.")
    parser.add_argument("--dpi", type=int, default=300, help="Print DPI.")
    parser.add_argument("--red-color", default="255,0,0", help="RGB color, for example 255,0,0.")
    parser.add_argument("--pdf-name", default="verify_test_print.pdf", help="Printable PDF file name inside output folder.")
    parser.add_argument("--overwrite", action="store_true", help="Remove existing output folder before writing.")
    return parser.parse_args()


def parse_rgb(value: str) -> tuple[int, int, int]:
    parts = value.split(",")
    if len(parts) != 3:
        raise ValueError("--red-color must be RGB format, for example: 255,0,0")
    rgb = tuple(int(part.strip()) for part in parts)
    if any(channel < 0 or channel > 255 for channel in rgb):
        raise ValueError("RGB channel values must be in [0, 255].")
    return rgb


def cm_to_px(cm: float, dpi: int) -> int:
    return max(1, round(cm / CM_PER_INCH * dpi))


def validate_args(args: argparse.Namespace) -> None:
    if args.samples_per_class <= 0:
        raise ValueError("--samples-per-class must be greater than 0.")
    if args.image_size_cm <= 0:
        raise ValueError("--image-size-cm must be greater than 0.")
    if args.red_height_cm <= 0:
        raise ValueError("--red-height-cm must be greater than 0.")
    if args.dpi <= 0:
        raise ValueError("--dpi must be greater than 0.")


def discover_class_images(source_root: Path) -> dict[str, list[Path]]:
    if not source_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")

    class_to_images: dict[str, list[Path]] = {}
    for class_dir in sorted((p for p in source_root.iterdir() if p.is_dir()), key=lambda p: p.name):
        images = sorted((p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS), key=lambda p: p.name)
        if images:
            class_to_images[class_dir.name] = images

    if not class_to_images:
        raise ValueError(f"No class image folders found under: {source_root}")
    return class_to_images


def select_images(class_to_images: dict[str, list[Path]], samples_per_class: int, seed: int) -> dict[str, list[Path]]:
    rng = random.Random(seed)
    selected: dict[str, list[Path]] = {}
    for class_name, images in sorted(class_to_images.items()):
        sample_count = min(samples_per_class, len(images))
        selected[class_name] = sorted(rng.sample(images, sample_count), key=lambda p: p.name)
    return selected


def create_printable_image(
    source: Path,
    target: Path,
    image_size_px: int,
    red_height_px: int,
    dpi: int,
    red_color: tuple[int, int, int],
) -> dict[str, int]:
    target.parent.mkdir(parents=True, exist_ok=True)
    output_height = image_size_px + red_height_px
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        original_width, original_height = image.size
        square_image = ImageOps.fit(image, (image_size_px, image_size_px), method=Image.Resampling.LANCZOS)
        output_image = Image.new("RGB", (image_size_px, output_height), red_color)
        output_image.paste(square_image, (0, 0))
        output_image.save(target, quality=95, dpi=(dpi, dpi))
    return {
        "original_width": original_width,
        "original_height": original_height,
        "output_width": image_size_px,
        "output_height": output_height,
        "image_area_height": image_size_px,
        "patch_top": image_size_px,
        "patch_height": red_height_px,
    }


def prepare_output(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output root already exists: {output_root}\nUse --overwrite to recreate it.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def make_pdf_pages(image_paths: list[Path], image_size_px: int, red_height_px: int, dpi: int) -> list[Image.Image]:
    a4_width_px = cm_to_px(21.0, dpi)
    a4_height_px = cm_to_px(29.7, dpi)
    margin_px = cm_to_px(1.0, dpi)

    item_width = image_size_px + red_height_px
    item_height = image_size_px
    columns = 1
    rows = 2
    per_page = columns * rows
    horizontal_gap_px = 0
    vertical_gap_px = max(0, (a4_height_px - 2 * margin_px - rows * item_height) // max(1, rows - 1)) if rows > 1 else 0
    pages: list[Image.Image] = []

    for page_start in range(0, len(image_paths), per_page):
        page = Image.new("RGB", (a4_width_px, a4_height_px), "white")
        x = max(0, (a4_width_px - item_width) // 2)
        for index, image_path in enumerate(image_paths[page_start : page_start + per_page]):
            row = index // columns
            y = margin_px + row * (item_height + vertical_gap_px)
            with Image.open(image_path) as image:
                rotated_image = image.convert("RGB").rotate(90, expand=True)
                page.paste(rotated_image, (x, y))
        pages.append(page)
    return pages


def save_printable_pdf(image_paths: list[Path], pdf_path: Path, image_size_px: int, red_height_px: int, dpi: int) -> None:
    pages = make_pdf_pages(image_paths, image_size_px, red_height_px, dpi)
    if not pages:
        return
    first_page, rest_pages = pages[0], pages[1:]
    first_page.save(pdf_path, save_all=True, append_images=rest_pages, resolution=dpi)


def create_verify_dataset(args: argparse.Namespace) -> None:
    validate_args(args)
    red_color = parse_rgb(args.red_color)
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    image_size_px = cm_to_px(args.image_size_cm, args.dpi)
    red_height_px = cm_to_px(args.red_height_cm, args.dpi)

    class_to_images = discover_class_images(source_root)
    selected = select_images(class_to_images, args.samples_per_class, args.seed)
    prepare_output(output_root, args.overwrite)

    records = []
    output_images: list[Path] = []
    for class_name, images in selected.items():
        for image_path in images:
            target_path = output_root / class_name / image_path.name
            image_info = create_printable_image(image_path, target_path, image_size_px, red_height_px, args.dpi, red_color)
            output_images.append(target_path)
            records.append(
                {
                    "class": class_name,
                    "source": str(image_path),
                    "output": str(target_path),
                    "red_color": red_color,
                    "dpi": args.dpi,
                    "image_size_cm": args.image_size_cm,
                    "red_height_cm": args.red_height_cm,
                    "output_size_cm": [args.image_size_cm, args.image_size_cm + args.red_height_cm],
                    **image_info,
                }
            )

    pdf_path = output_root / args.pdf_name
    save_printable_pdf(output_images, pdf_path, image_size_px, red_height_px, args.dpi)

    record_path = output_root / "verify_test_summary.json"
    record_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Created {len(records)} printable images in: {output_root}")
    print(f"Image area: {args.image_size_cm} x {args.image_size_cm} cm")
    print(f"Red area: {args.image_size_cm} x {args.red_height_cm} cm")
    print(f"Printable PDF: {pdf_path}")
    print(f"Summary: {record_path}")


if __name__ == "__main__":
    create_verify_dataset(parse_args())
