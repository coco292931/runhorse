# YOLOv5 / YOLOv8 图像分类训练说明

本目录脚本用于把当前数据集自动划分为 YOLO 分类训练格式，并分别训练 YOLOv5、YOLOv8 分类模型。

## 1. 数据集格式

原始数据集应保持如下结构：

```text
根目录
├─ 武器
│  ├─ A-枪支
│  └─ B-爆炸物
├─ 物资
│  ├─ C-急救包
│  └─ D-望远镜
└─ 载具
   ├─ E-装甲车
   └─ F-救护车
```

脚本会把二级目录作为最终分类类别：

```text
A-枪支
B-爆炸物
C-急救包
D-望远镜
E-装甲车
F-救护车
```

原始图片不会被移动、删除或重命名。脚本会生成新的分类训练目录：

```text
yolo_cls_dataset
├─ train
├─ val
└─ test
```

支持图片格式：`.jpg`、`.jpeg`、`.png`、`.bmp`、`.webp`。

## 2. 自动划分数据集

在当前目录运行：

```powershell
python ".\prepare_yolo_cls_dataset.py" --source-root "." --output-root ".\yolo_cls_dataset" --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1 --seed 42 --overwrite
```

默认划分比例：

```text
train: 70%
val:   20%
test:  10%
```

划分完成后会生成：

```text
yolo_cls_dataset\split_summary.json
```

里面记录每个类别在 train / val / test 中的图片数量。

## 3. 训练 YOLOv8 分类模型

安装依赖：

```powershell
python -m pip install --upgrade pip
python -m pip install ultralytics
```

CPU 训练：

```powershell
python ".\train_yolov8_cls.py" --data ".\yolo_cls_dataset" --epochs 50 --imgsz 224 --batch 16 --device cpu
```

GPU 训练：

```powershell
python ".\train_yolov8_cls.py" --data ".\yolo_cls_dataset" --epochs 50 --imgsz 224 --batch 16 --device 0
```

训练结果默认保存到：

```text
runs_yolov8_cls\zoumaguangbei_cls
```

验证模型：

```powershell
yolo classify val model=".\runs_yolov8_cls\zoumaguangbei_cls\weights\best.pt" data=".\yolo_cls_dataset" imgsz=224
```

预测测试集中的某一类图片：

```powershell
yolo classify predict model=".\runs_yolov8_cls\zoumaguangbei_cls\weights\best.pt" source=".\yolo_cls_dataset\test\A-枪支" imgsz=224
```

## 4. 训练 YOLOv5 分类模型

YOLOv5 需要先准备官方仓库。建议把 YOLOv5 放到英文路径，降低第三方库对中文路径不兼容的风险。

示例：

```powershell
mkdir C:\yolo
git clone https://github.com/ultralytics/yolov5.git C:\yolo\yolov5
python -m pip install -r C:\yolo\yolov5\requirements.txt
```

CPU 训练：

```powershell
python ".\train_yolov5_cls.py" --data ".\yolo_cls_dataset" --yolov5-dir "C:\yolo\yolov5" --epochs 50 --imgsz 224 --batch-size 16 --device cpu
```

GPU 训练：

```powershell
python ".\train_yolov5_cls.py" --data ".\yolo_cls_dataset" --yolov5-dir "C:\yolo\yolov5" --epochs 50 --imgsz 224 --batch-size 16 --device 0
```

训练结果默认保存到：

```text
runs_yolov5_cls\zoumaguangbei_cls
```

验证模型：

```powershell
python "C:\yolo\yolov5\classify\val.py" --weights ".\runs_yolov5_cls\zoumaguangbei_cls\weights\best.pt" --data ".\yolo_cls_dataset" --img 224
```

预测测试集中的某一类图片：

```powershell
python "C:\yolo\yolov5\classify\predict.py" --weights ".\runs_yolov5_cls\zoumaguangbei_cls\weights\best.pt" --source ".\yolo_cls_dataset\test\A-枪支" --img 224
```

## 5. GPU 可用性检查

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

如果输出 `True`，训练命令可以使用 `--device 0`。否则使用 `--device cpu`。

## 6. 常见调整

如果显存不足或训练太慢，可以降低 batch 或图片尺寸：

```powershell
--batch 8 --imgsz 160
```

YOLOv5 对应参数是：

```powershell
--batch-size 8 --imgsz 160
```

当前数据量较小，建议先使用小模型：

```text
yolov8n-cls.pt
yolov5s-cls.pt
```

也可以先做 1 个 epoch 的快速测试：

```powershell
python ".\train_yolov8_cls.py" --data ".\yolo_cls_dataset" --epochs 1 --imgsz 224 --batch 4 --device cpu
```

```powershell
python ".\train_yolov5_cls.py" --data ".\yolo_cls_dataset" --yolov5-dir "C:\yolo\yolov5" --epochs 1 --imgsz 224 --batch-size 4 --device cpu
```

## 7. 小车摄像头实拍识别

打印图尺寸固定为：上方图片区域 `12×12cm`，下方红色参考块 `12×5cm`，总尺寸 `12×17cm`。

脚本会先检测红色参考块，输出红色块坐标，然后根据红色块位置关系还原 `12×17cm` 正视图，裁切上方 `12×12cm` 区域并调用 YOLOv8 分类模型。

单图片识别：

```powershell
python ".\recognize_printed_marker_yolov8.py" --mode image --source ".\Cache_19fc14d1bc378a86.png" --model ".\runs_yolov8_cls\zoumaguangbei_cls\weights\best.pt" --save-dir ".\runs_printed_marker" --device cpu --imgsz 224 --output-width 720 --save-debug
```

如果红色块较远或光线较暗，可以降低红色阈值：

```powershell
python ".\recognize_printed_marker_yolov8.py" --mode image --source ".\Cache_19fc14d1bc378a86.png" --model ".\runs_yolov8_cls\zoumaguangbei_cls\weights\best.pt" --min-red-area 40 --red-s-min 60 --red-v-min 40 --save-debug
```

输出目录示例：

```text
runs_printed_marker\Cache_19fc14d1bc378a86
├─ input_with_overlay.jpg
├─ red_mask.jpg
├─ rectified_12x17.jpg
├─ crop_12x12.jpg
└─ result.json
```

webcam 实时识别：

```powershell
python ".\recognize_printed_marker_yolov8.py" --mode webcam --camera 0 --model ".\runs_yolov8_cls\zoumaguangbei_cls\weights\best.pt" --camera-width 1280 --camera-height 720 --device cpu --imgsz 224 --output-width 720 --show
```

webcam 窗口快捷键：

```text
q / ESC：退出
s：保存当前帧、红色 mask、矫正图、裁切图和 result.json
```

## 8. 注意

本任务是图像分类，不是目标检测，因此不会生成 YOLO 检测任务使用的 `labels/*.txt` 或 `data.yaml`。
