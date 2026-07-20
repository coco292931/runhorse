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
TARGET_SIZE = (64,64)
ENABLE_WHOLE_ROTATION = False          # 是否启用整体旋转增强（0/90/180/270 度）

# 数据增强参数
ROTATION_RANGE = (-15.0, 15.0)       # 随机旋转角度范围（度）
GEOMETRY_ANCHOR_RATIO = (0.5, 1.0)   # 几何变换锚点比例：下底边中点
PERSPECTIVE_H_ANGLE_RANGE = (-4,4)   # 透视水平偏转角范围（度），正值=向左旋转（右边变窄），负值=向右旋转（左边变窄）
PERSPECTIVE_V_ANGLE_RANGE = (-4.0, 0)     # 透视竖直偏转角范围（度），正值=向上旋转（下边变窄，俯视效果），负值=向下旋转（上边变窄，仰视效果）
CROP_SCALE_RANGE = (0.8, 1.2)         # 随机裁切缩放比例
TRANSLATION_RATIO = 0.02              # 随机平移比例（相对原图尺寸）

BRIGHTNESS_RANGE = (0.7, 0.9)         # 亮度调整范围（<1 变暗，>1 变亮，模拟不同曝光条件）
CONTRAST_RANGE = (0.8, 2.5)           # 对比度调整范围（>1 增强对比）
SATURATION_RANGE = (0.65, 0.9)         # 饱和度调整范围（<1 降低饱和度，>1 增强饱和度）

SHARPNESS_RANGE = (1, 3.0)          # 锐化调整范围（<1 变模糊，>1 变锐利，0=完全模糊）

# —— 以下退化在缩放到目标尺寸(TARGET_SIZE)之后执行，参数按目标像素(如 64px)标定 ——
# 低分辨率摄像头模拟：先降采样到 target 的该比例，再用双线性放大回 target，
# 物理性丢弃高频细节（比单纯加模糊更接近真实低清传感器）。
LOW_RES_SCALE_RANGE = (0.7, 0.9)     # 降采样比例范围（相对目标尺寸），0.35≈64→22px 再放大回 64
GAUSSIAN_BLUR_RANGE = (0, 0.2)       # 目标尺度高斯模糊半径（像素），模拟镜头/透视矫正插值模糊
NOISE_AMOUNT_RANGE = (5, 15)          # 高斯噪声强度范围
COLOR_NOISE_STD_RANGE = (5.0, 12.0)   # 彩色高斯噪声标准差（像素加性扰动，越大彩色颗粒越明显）
JPEG_QUALITY_RANGE = (100, 100)         # JPEG 压缩质量范围（模拟传输压缩损失）


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare YOLO classification train/val/test dataset.")
    parser.add_argument("--source-root", type=Path, default=Path("."), help="Original dataset root.")
    parser.add_argument("--output-root", type=Path, default=Path("yolo_cls_dataset_camstyle"), help="Output split dataset root.")
    parser.add_argument("--target-size", type=int, default=TARGET_SIZE[0], help="Output image size in pixels, width=height.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=random.randint(0,32768), help="Random seed.")
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
    parser.add_argument("--jpeg-quality", type=int, nargs=2, default=list(JPEG_QUALITY_RANGE),
                        metavar=("LOW", "HIGH"),
                        help="JPEG 压缩质量范围（模拟传输压缩损失），low>=100 时跳过该步骤，例如 --jpeg-quality 40 95")
    parser.add_argument("--no-whole-rotation", action="store_true", default=not ENABLE_WHOLE_ROTATION,
                        help="禁用整体旋转增强（0/90/180/270 度）")
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


def geometry_anchor(image: Image.Image) -> tuple[float, float]:
    """返回几何变换锚点：默认下底边中点"""
    width, height = image.size
    return width * GEOMETRY_ANCHOR_RATIO[0], height * GEOMETRY_ANCHOR_RATIO[1]


def random_crop(image: Image.Image, rng: random.Random) -> Image.Image:
    """
    随机裁切：在缩放+微小平移后裁切图片。
    - crop_scale <= 1 时裁切后放大，裁切中心仍在原中心附近随机平移 ±3%
    - crop_scale > 1 时缩小后贴回原画布，以下底边中点为缩放锚点
    """
    width, height = image.size
    crop_scale = rng.uniform(*CROP_SCALE_RANGE)

    if crop_scale > 1.0:
        scaled_width = max(1, int(round(width / crop_scale)))
        scaled_height = max(1, int(round(height / crop_scale)))
        scaled_image = image.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)

        anchor_x = int(round(width * GEOMETRY_ANCHOR_RATIO[0]))
        anchor_y = int(round(height * GEOMETRY_ANCHOR_RATIO[1]))
        scaled_anchor_x = int(round(scaled_width * GEOMETRY_ANCHOR_RATIO[0]))
        scaled_anchor_y = int(round(scaled_height * GEOMETRY_ANCHOR_RATIO[1]))

        canvas = Image.new("RGB", (width, height), (255, 255, 255))
        canvas.paste(scaled_image, (anchor_x - scaled_anchor_x, anchor_y - scaled_anchor_y))
        return canvas

    crop_width = max(1, int(round(width * crop_scale)))
    crop_height = max(1, int(round(height * crop_scale)))

    max_shift_x = int(round(width * TRANSLATION_RATIO))
    max_shift_y = int(round(height * TRANSLATION_RATIO))
    center_x = width // 2 + rng.randint(-max_shift_x, max_shift_x)
    center_y = height // 2 + rng.randint(-max_shift_y, max_shift_y)

    left = center_x - crop_width // 2
    top = center_y - crop_height // 2
    right = left + crop_width
    bottom = top + crop_height

    crop = Image.new("RGB", (crop_width, crop_height), (255, 255, 255))
    src_left = clamp(left, 0, width)
    src_top = clamp(top, 0, height)
    src_right = clamp(right, 0, width)
    src_bottom = clamp(bottom, 0, height)

    if src_right > src_left and src_bottom > src_top:
        cropped_region = image.crop((src_left, src_top, src_right, src_bottom))
        crop.paste(cropped_region, (src_left - left, src_top - top))
    return crop


def downscale_upscale(image: Image.Image, rng: random.Random) -> Image.Image:
    """
    低分辨率摄像头模拟：先把图片降采样到目标尺寸的一个较小比例（BILINEAR 平均掉
    高频细节），再用 BILINEAR 放大回原尺寸。

    与单纯加模糊不同，这一步会真正丢弃高频信息——放大时无法恢复，从而复现
    低清传感器下"字符边缘发糊、细笔画糊成一团"的观感。应在图片已缩到
    TARGET_SIZE 之后调用，比例才与实车 imgsz 对应。
    """
    width, height = image.size
    scale = rng.uniform(*LOW_RES_SCALE_RANGE)
    if scale >= 0.999:
        return image
    low_w = max(1, int(round(width * scale)))
    low_h = max(1, int(round(height * scale)))
    small = image.resize((low_w, low_h), Image.Resampling.BILINEAR)
    return small.resize((width, height), Image.Resampling.BILINEAR)


def add_noise(image: Image.Image, rng: random.Random) -> Image.Image:
    """添加随机高斯噪声，模拟摄像头传感器噪点"""
    noise_amount = rng.randint(*NOISE_AMOUNT_RANGE)
    noise = Image.effect_noise(image.size, noise_amount).convert("L")
    return Image.blend(image, Image.merge("RGB", (noise, noise, noise)), 0.18)


def add_color_noise(image: Image.Image, rng: random.Random) -> Image.Image:
    """添加三通道加性彩色噪声，避免把图像整体混合成灰色"""
    import numpy as np

    noise_std = rng.uniform(*COLOR_NOISE_STD_RANGE)
    np_rng = np.random.default_rng(rng.randrange(2**32))
    pixels = np.asarray(image, dtype=np.int16)
    noise = np_rng.normal(0.0, noise_std, pixels.shape)
    noisy_pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_pixels, mode="RGB")


def whole_rotate(image: Image.Image, rng: random.Random) -> Image.Image:
    """随机整体旋转 0/90/180/270 度，模拟摄像头安装方向变化"""
    angle = rng.choice([0, 90, 180, 270])
    if angle == 0:
        return image
    return image.rotate(angle, expand=True)


def simulate_jpeg_compress(image: Image.Image, rng: random.Random, quality_range: tuple[int, int]) -> Image.Image:
    """模拟 JPEG 压缩伪影：先以指定质量保存到 BytesIO 再读回。quality_range 下限 >=100 时跳过。"""
    if quality_range[0] >= 100:
        return image
    import io
    quality = rng.randint(*quality_range)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def perspective_transform(
    image: Image.Image,
    rng: random.Random,
    h_angle_range: tuple[float, float],
    v_angle_range: tuple[float, float],
) -> Image.Image:
    """
    随机透视变换，模拟摄像头水平/竖直偏转。

    原理：将图片的四个角想象为三维空间中的一个矩形平面，
    绕竖直轴旋转（水平偏转→左右倾斜），
    绕水平轴旋转（竖直偏转→俯仰倾斜），
    再投影回二维平面得到目标四边形。

    水平偏转（h_angle_range）：正值=相机向左转（右边变窄），负值=相机向右转（左边变窄）
    竖直偏转（v_angle_range）：正值=相机向上转（下边变窄，俯视效果），负值=相机向下转（上边变窄，仰视效果）
    """
    import math
    width, height = image.size
    h_rad = math.radians(rng.uniform(*h_angle_range))
    v_rad = math.radians(rng.uniform(*v_angle_range))
    # 焦距（取较长边的一半作为视距，保证变换后内容基本在画面内）
    focal = max(width, height) / 2.0
    cx, cy = width / 2.0, height / 2.0
    # 原始四个角（以中心为原点）
    corners = [(-cx, -cy), (cx, -cy), (cx, cy), (-cx, cy)]
    dst = []
    for x, y in corners:
        # 绕 Y 轴旋转 h_rad，绕 X 轴旋转 v_rad
        # 3D 点 (x, y, 0)，先绕 Y 轴旋转
        cos_h, sin_h = math.cos(h_rad), math.sin(h_rad)
        x1 = x * cos_h
        z1 = x * sin_h
        # 再绕 X 轴旋转
        cos_v, sin_v = math.cos(v_rad), math.sin(v_rad)
        y1 = y * cos_v - 0 * sin_v  # z=0 时绕 X 轴旋转简化为 y1 = y*cos_v
        z2 = y * sin_v + 0 * cos_v
        z_total = z1 + z2 + focal  # 加上焦距作为 Z 偏移
        # 透视投影到 2D
        px = x1 * focal / z_total + cx
        py = y1 * focal / z_total + cy
        dst.append((px, py))
    coeffs = _find_perspective_coeffs(corners, dst, cx, cy)
    return image.transform(
        image.size, Image.Transform.PERSPECTIVE, coeffs,
        resample=Image.Resampling.BILINEAR, fillcolor=(255, 255, 255),
    )


def _find_perspective_coeffs(
    src: list[tuple[float, float]],
    dst: list[tuple[float, float]],
    cx: float, cy: float,
) -> list[float]:
    """
    计算从原始坐标到目标坐标的透视变换系数。
    src/dst 是相对中心 (cx,cy) 的偏移量，返回 PIL PERSPECTIVE 所需的 8 个系数。
    """
    import numpy as np
    # 转为绝对坐标
    src_abs = [(sx + cx, sy + cy) for sx, sy in src]
    dst_abs = [(dx, dy) for dx, dy in dst]
    matrix = []
    for i in range(4):
        sx, sy = src_abs[i]
        dx, dy = dst_abs[i]
        matrix.extend([sx, sy, 1, 0, 0, 0, -dx * sx, -dx * sy])
        matrix.extend([0, 0, 0, sx, sy, 1, -dy * sx, -dy * sy])
    A = np.array(matrix, dtype=float).reshape(8, 8)
    B = np.array([p for pair in dst_abs for p in pair], dtype=float)
    return np.linalg.solve(A, B).tolist()


def preprocess_image(
    source: Path,
    rng: random.Random,
    target_size: tuple[int, int],
    enable_whole_rotation: bool = True,
    jpeg_quality_range: tuple[int, int] = JPEG_QUALITY_RANGE,
) -> Image.Image:
    """
    对单张图片执行完整预处理流程（数据增强）：

    【几何 / 色彩阶段】在原图分辨率上做，保证插值质量：
     1. 读取图片，根据 EXIF 方向信息自动旋转摆正，转为 RGB
     2. 随机整体旋转 0/90/180/270 度（可通过 --no-whole-rotation 关闭）
     3. 以下底边中点为中心随机小角度旋转，保持当前画布尺寸（BILINEAR 插值）
     4. 随机透视变换（模拟摄像头偏转 + 矫正）
     5. 随机裁切（模拟构图变化）
     6. 随机调整亮度、对比度、饱和度
     7. 随机锐化

    【退化阶段】先缩到目标尺寸，再在目标尺度上按真实 imgsz 施加退化，
    避免最后一次 LANCZOS 缩放把噪声/压缩/模糊重新抹平：
     8. 缩放到统一目标尺寸（LANCZOS）
     9. 低分辨率降采样再升采样（真正丢弃高频细节，模拟低清摄像头）
    10. 目标尺度高斯模糊（模拟镜头/透视矫正插值模糊）
    11. 添加高斯噪声 + 彩色噪声（低清后颗粒才明显）
    12. 模拟 JPEG 压缩伪影（设 --jpeg-quality 100 100 关闭）
    """
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        # —— 几何 / 色彩阶段：原图分辨率 ——
        if enable_whole_rotation:
            image = whole_rotate(image, rng)
        image = image.rotate(
            rng.uniform(*ROTATION_RANGE),
            resample=Image.Resampling.BILINEAR,
            expand=False,
            center=geometry_anchor(image),
            fillcolor=(255, 255, 255),
        )
        if PERSPECTIVE_H_ANGLE_RANGE != (0, 0) or PERSPECTIVE_V_ANGLE_RANGE != (0, 0):
            image = perspective_transform(image, rng, PERSPECTIVE_H_ANGLE_RANGE, PERSPECTIVE_V_ANGLE_RANGE)
        image = random_crop(image, rng)
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(*BRIGHTNESS_RANGE))
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(*CONTRAST_RANGE))
        image = ImageEnhance.Color(image).enhance(rng.uniform(*SATURATION_RANGE))
        image = ImageEnhance.Sharpness(image).enhance(rng.uniform(*SHARPNESS_RANGE))

        # —— 退化阶段：先缩到目标尺寸，再在目标尺度上退化 ——
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        image = downscale_upscale(image, rng)
        blur_radius = rng.uniform(*GAUSSIAN_BLUR_RANGE)
        if blur_radius > 0:
            image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        image = add_noise(image, rng)
        image = add_color_noise(image, rng)
        image = simulate_jpeg_compress(image, rng, jpeg_quality_range)
        return image


def save_processed_image(image: Image.Image, target: Path) -> None:
    """保存处理后的图片，JPEG/WEBP 格式使用 95 品质"""
    suffix = target.suffix.lower()
    save_kwargs: dict[str, object] = {}
    if suffix in {".jpg", ".jpeg"}:
        save_kwargs = {"quality": 95}
    elif suffix == ".webp":
        save_kwargs = {"quality": 95}
    image.save(target, **save_kwargs)


def preprocess_and_save(
    source: Path,
    target: Path,
    rng: random.Random,
    target_size: tuple[int, int],
    aug_kwargs: dict | None = None,
) -> None:
    """对源图片执行预处理并保存到目标路径"""
    target.parent.mkdir(parents=True, exist_ok=True)
    if aug_kwargs is None:
        aug_kwargs = {}
    processed_image = preprocess_image(source, rng, target_size, **aug_kwargs)
    save_processed_image(processed_image, target)


def _process_one_job(job: tuple) -> None:
    """多进程用的顶层任务函数：source, target, seed, target_size, aug_kwargs → 预处理+保存"""
    source, target, seed, target_size, aug_kwargs = job
    preprocess_and_save(source, target, random.Random(seed), target_size, aug_kwargs)


def _build_jobs(
    class_to_images: dict[str, list[Path]],
    output_root: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    aug_multiplier: int,
    target_size: tuple[int, int],
    aug_kwargs: dict | None = None,
) -> tuple[list[tuple], dict[str, dict[str, int]]]:
    """
    构建所有待处理任务，返回 (jobs, summary)。
    job = (source, target, seed, target_size, aug_kwargs)
    summary = {类别名: {split名: 数量}}
    """
    if aug_kwargs is None:
        aug_kwargs = {}
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
                    jobs.append((image, target_path, job_seed_base + job_idx, target_size, aug_kwargs))
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
    aug_kwargs: dict | None = None,
) -> tuple[list[tuple], dict[str, dict[str, int]]]:
    """
    先让每张原图生成 aug_multiplier 个增强版本，再把这些版本划分到 train/val/test。

    这样 train 会覆盖每一张原图的至少一个增强版本；代价是同一原图的不同增强版本
    可能同时出现在 train 和 val/test 中，验证指标会偏乐观。
    """
    if aug_kwargs is None:
        aug_kwargs = {}
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
                    jobs.append((image, target_path, job_seed_base + copy_idx + job_idx, target_size, aug_kwargs))
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
    aug_kwargs: dict | None = None,
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
    if aug_kwargs is None:
        aug_kwargs = {}
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
            train_ratio, val_ratio, test_ratio, seed, aug_multiplier, target_size, aug_kwargs,
        )
    else:
        jobs, summary = _build_jobs(
            class_to_images, output_root,
            train_ratio, val_ratio, test_ratio, seed, aug_multiplier, target_size, aug_kwargs,
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
            "geometry_anchor_ratio": list(GEOMETRY_ANCHOR_RATIO),
            "enabled": {
                "whole_rotation": not args.no_whole_rotation,
            },
            "perspective_h_angle_range_degrees": list(PERSPECTIVE_H_ANGLE_RANGE),
            "perspective_v_angle_range_degrees": list(PERSPECTIVE_V_ANGLE_RANGE),
            "crop_scale_range": list(CROP_SCALE_RANGE),
            "translation_ratio": TRANSLATION_RATIO,
            "brightness_range": list(BRIGHTNESS_RANGE),
            "contrast_range": list(CONTRAST_RANGE),
            "saturation_range": list(SATURATION_RANGE),
            "sharpness_range": list(SHARPNESS_RANGE),
            "low_res_scale_range": list(LOW_RES_SCALE_RANGE),
            "gaussian_blur_range": list(GAUSSIAN_BLUR_RANGE),
            "noise_amount_range": list(NOISE_AMOUNT_RANGE),
            "color_noise_std_range": list(COLOR_NOISE_STD_RANGE),
            "jpeg_quality_range": list(JPEG_QUALITY_RANGE),
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

    # 构建数据增强开关参数（传给 preprocess_image）
    aug_kwargs = {
        "enable_whole_rotation": not args.no_whole_rotation,
        "jpeg_quality_range": tuple(args.jpeg_quality),
    }

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
        aug_kwargs=aug_kwargs,
    )
    write_summary(output_root, source_root, class_to_images, split_counts, args)
    print_summary(split_counts)
    print(f"Output: {output_root}")
    print(f"Summary: {output_root / 'split_summary.json'}")


if __name__ == "__main__":
    main()
