# aug30v2 复现实验说明

本文件记录 `yolo_cls_dataset_aug30v2` 数据集生成、`zoumaguangbei_cls_aug30v2` 模型训练，以及当前图像识别阶段配置。

目标是按这份文档尽可能复现同一批结果，而不是只复现运行环境。

## 1. 原始数据集

原始目录根路径：

```text
C:\Users\tonyp\Downloads\runhorse
```

原始类别与张数：

```text
武器/A-枪支        54
武器/B-爆炸物      50
物资/C-急救包      51
物资/D-望远镜      53
载具/E-装甲车      41
载具/F-救护车      55
总原图            304
```

## 2. 数据集生成

实际生成命令：

```powershell
python .\prepare_yolo_cls_dataset.py --source-root . --output-root .\yolo_cls_dataset_aug30v2 --target-size 64 --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1 --seed 42 --aug-multiplier 30 --split-mode augment-first --workers 0 --overwrite
```

实际生成参数：

```text
source_root=.
output_root=.\yolo_cls_dataset_aug30v2
target_size=64x64
train_ratio=0.7
val_ratio=0.2
test_ratio=0.1
seed=42
aug_multiplier=30
split_mode=augment-first
workers=0
copy_mode=copy
overwrite=True
```

关键逻辑：

```text
不是先划分原图，再增强 train。
而是每张原图先生成 30 张增强图，再把这 30 张增强图按 0.7/0.2/0.1 分到 train/val/test。
```

这意味着：

```text
每张原图都会有增强版本进入训练集。
但同一原图的不同增强版本也可能同时出现在 train 和 val/test 中。
```

增强与预处理顺序：

```text
1. 读取图片，按 EXIF 纠正方向，转 RGB
2. 随机旋转：[-10.0, 10.0] 度
3. 随机裁切缩放：crop_scale [0.8, 1.2]
4. 随机平移：translation_ratio 0.03
5. 亮度增强：brightness [0.3, 0.8]
6. 对比度增强：contrast [2, 5]
7. 加噪声：noise_amount [22, 80]
8. BoxBlur 模糊：blur_radius [5, 15]
9. resize 到 64x64，LANCZOS
```

实现细节：

```text
rotate resample=BILINEAR
rotate fillcolor=(128,128,128)
noise = Image.effect_noise(...).convert("L")
noise blend alpha = 0.18
blur = ImageFilter.BoxBlur
```

切分随机种子逻辑：

```text
split_seed = 42 + class_index * 1000003 + image_index * 9176
job_seed_base = 42 + class_index * 1000003 + image_index * 9176 + split_index * 1009
```

生成结果：

```text
A-枪支   total=54  train=1134  val=324  test=162
B-爆炸物 total=50  train=1050  val=300  test=150
C-急救包 total=51  train=1071  val=306  test=153
D-望远镜 total=53  train=1113  val=318  test=159
E-装甲车 total=41  train=861   val=246  test=123
F-救护车 total=55  train=1155  val=330  test=165

总计 train=6384  val=1824  test=912
总计样本数 9120
```

数据集摘要文件：

```text
.\yolo_cls_dataset_aug30v2\split_summary.json
```

## 3. 训练

实际启动命令：

```powershell
python .\train_yolov8_cls.py --data .\yolo_cls_dataset_aug30v2 --model .\yolov8n-cls.pt --epochs 50 --imgsz 64 --batch 16 --device 0 --workers 8 --name zoumaguangbei_cls_aug30v2
```

训练输入数据：

```text
train=6384 images
val=1824 images
test=912 images
classes=6
```

训练主要参数：

```text
task=classify
model=.\yolov8n-cls.pt
project=.\runs_yolov8_cls
name=zoumaguangbei_cls_aug30v2
save_dir=.\runs_yolov8_cls\zoumaguangbei_cls_aug30v2
epochs=50
imgsz=64
batch=16
device=0
workers=8
val=True
deterministic=True
seed=0
amp=True
pretrained=True
```

Ultralytics 实际生效训练参数：

```text
agnostic_nms=False
angle=1.0
augment=False
auto_augment=randaugment
bgr=0.0
box=7.5
cache=False
classes=None
close_mosaic=10
cls=0.5
cls_pw=0.0
compile=False
conf=None
copy_paste=0.0
copy_paste_mode=flip
cos_lr=False
cutmix=0.0
degrees=0.0
dfl=1.5
dnn=False
dropout=0.0
dynamic=False
embed=None
end2end=None
erasing=0.4
exist_ok=False
fliplr=0.5
flipud=0.0
format=torchscript
fraction=1.0
freeze=None
half=False
hsv_h=0.015
hsv_s=0.7
hsv_v=0.4
int8=False
iou=0.7
keras=False
kobj=1.0
line_width=None
lrf=0.01
mask_ratio=4
max_det=300
mixup=0.0
mosaic=1.0
multi_scale=0.0
nbs=64
nms=False
opset=None
optimize=False
optimizer=auto
overlap_mask=True
patience=100
perspective=0.0
plots=True
pose=12.0
profile=False
rect=False
resume=False
retina_masks=False
rle=1.0
save=True
save_conf=False
save_crop=False
save_frames=False
save_json=False
save_period=-1
save_txt=False
scale=0.5
shear=0.0
show=False
show_boxes=True
show_conf=True
show_labels=True
simplify=True
single_cls=False
source=None
split=val
stream_buffer=False
time=None
tracker=botsort.yaml
translate=0.1
verbose=True
vid_stride=1
visualize=False
warmup_bias_lr=0.1
warmup_epochs=3.0
warmup_momentum=0.8
weight_decay=0.0005
workspace=None
```

优化器实际值：

```text
optimizer=auto
实际优化器 = AdamW
实际 lr = 0.001
实际 momentum = 0.9
weight_decay = 0.0005
```

说明：

```text
Ultralytics 在这次训练中忽略了手动 lr0，自动选择了 AdamW(lr=0.001, momentum=0.9)。
所以如果要严格复现结果，应该按“实际生效值”而不是“脚本传入值”理解。
```

模型输出目录：

```text
.\runs_yolov8_cls\zoumaguangbei_cls_aug30v2
```

训练日志：

```text
.\train_zoumaguangbei_cls_aug30v2_gpu.log
```

## 4. 图像识别

识别阶段使用的几何假设固定为：

```text
上方图片区域：12 x 12 cm
下方红色参考块：12 x 5 cm
总打印图：12 x 17 cm
EXTEND_RATIO = 12 / 5 = 2.4
```

YOLO 输入尺寸约束：

```text
YOLO_IMGSZ_MULTIPLE = 32
imgsz 会自动向上取整到 32 的整数倍
本次若显式传 imgsz=64，则不会再调整
```

建议复现实验时的识别启动命令：

```powershell
python .\recognize_printed_marker_tkinter.py --model .\runs_yolov8_cls\zoumaguangbei_cls_aug30v2\weights\best.pt --device 0 --imgsz 64 --output-width 720 --camera 0 --camera-width 1280 --camera-height 720 --camera-backend auto --mode usbcam --conf-thres 0.25 --first-aid-conf-thres 0.80 --red-h-low1 0 --red-h-high1 0 --red-h-low2 164 --red-h-high2 180 --red-s-min 0 --red-v-min 109 --min-red-area 610.0557620817843 --red-aspect-min 1.0 --red-aspect-max 6.0 --morph-kernel 8 --frame-skip 1 --smooth-confirm-frames 2 --smooth-unknown-hold-frames 4 --refresh-ms 30 --preview-width 640 --crop-preview-width 320 --crop-scale 1.0813148788927336 --crop-exposure 0.9799307958477508 --crop-contrast 3.0 --crop-blur 0.0 --camera-brightness 13.271375464684015 --camera-exposure -5.171003717472118 --camera-gain 98.58736059479554
```

当前 GUI 配置文件中的实际控制值：

```text
camera_brightness = 13.271375464684015
camera_exposure = -5.171003717472118
camera_gain = 98.58736059479554
red_h_low1 = 0
red_h_high1 = 0
red_h_low2 = 164
red_h_high2 = 180
red_s_min = 0
red_v_min = 109
min_red_area = 610.0557620817843
morph_kernel = 8
crop_scale = 1.0813148788927336
crop_exposure = 0.9799307958477508
crop_contrast = 3.0
crop_blur = 0.0
imgsz = 64
```

识别核心阈值与逻辑：

```text
conf_thres = 0.25
first_aid_conf_thres = 0.80
red_aspect_min = 1.0
red_aspect_max = 6.0
frame_skip = 1
smooth_confirm_frames = 2
smooth_unknown_hold_frames = 4
output_width = 720
```

识别阶段的图像后处理顺序：

```text
1. 从红块推算图片区域并做透视矫正
2. crop_scale
3. crop_exposure
4. crop_contrast
5. crop_blur
6. resize 到 YOLO imgsz
7. 输入分类模型
```

特殊类别处理：

```text
如果 top1 类别是 C-急救包，且 confidence < 0.80
则最终输出强制改成 unknown
```

注意：

```text
当前 config 文件里如果仍看到 crop_sharpness 旧字段，可以忽略。
现代码实际使用的是 crop_blur。
```

## 5. 最短复现路径

如果只关心尽量复现这一次结果，按下面三步：

```powershell
python .\prepare_yolo_cls_dataset.py --source-root . --output-root .\yolo_cls_dataset_aug30v2 --target-size 64 --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1 --seed 42 --aug-multiplier 30 --split-mode augment-first --workers 0 --overwrite
```

```powershell
python .\train_yolov8_cls.py --data .\yolo_cls_dataset_aug30v2 --model .\yolov8n-cls.pt --epochs 50 --imgsz 64 --batch 16 --device 0 --workers 8 --name zoumaguangbei_cls_aug30v2
```

```powershell
python .\recognize_printed_marker_tkinter.py --model .\runs_yolov8_cls\zoumaguangbei_cls_aug30v2\weights\best.pt --device 0 --imgsz 64 --output-width 720 --camera 0 --camera-width 1280 --camera-height 720 --camera-backend auto --mode usbcam --conf-thres 0.25 --first-aid-conf-thres 0.80 --red-h-low1 0 --red-h-high1 0 --red-h-low2 164 --red-h-high2 180 --red-s-min 0 --red-v-min 109 --min-red-area 610.0557620817843 --red-aspect-min 1.0 --red-aspect-max 6.0 --morph-kernel 8 --frame-skip 1 --smooth-confirm-frames 2 --smooth-unknown-hold-frames 4 --refresh-ms 30 --preview-width 640 --crop-preview-width 320 --crop-scale 1.0813148788927336 --crop-exposure 0.9799307958477508 --crop-contrast 3.0 --crop-blur 0.0 --camera-brightness 13.271375464684015 --camera-exposure -5.171003717472118 --camera-gain 98.58736059479554
```
