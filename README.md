# InsuLens：ROS 2 + YOLOv10 绝缘子巡检系统

InsuLens 是一个面向课程实践、算法原型和现场数据预筛的绝缘子视觉巡检项目。它在
Gazebo Classic 中自动生成并标注绝缘子仿真数据，以 YOLOv10s 完成仿真预训练，
再使用 CPLID 真实输电场景数据进行仿真到真实场景微调；运行阶段通过 ROS 2 接收
Gazebo 相机、真实 ROS 相机、单张图片、图片目录或视频，并输出检测结果、缺片告警、
带框图像、推理耗时和可复核的缺陷证据。

项目不依赖现成的绝缘子专用权重。仿真预训练默认从 `yolov10s.yaml` 网络结构
开始；只有在显式传入 `.pt` 模型时，训练节点才会使用该权重继续微调。

## 目录

- [系统能力](#系统能力)
- [系统架构](#系统架构)
- [环境与安装](#环境与安装)
- [完整工作流](#完整工作流)
- [运行与接入方式](#运行与接入方式)
- [ROS-2-接口与数据格式](#ros-2-接口与数据格式)
- [参数说明](#参数说明)
- [输出、模型与可追溯性](#输出模型与可追溯性)
- [目录结构](#目录结构)
- [结果、边界与数据说明](#结果边界与数据说明)

## 系统能力

| 功能 | 已实现行为 |
|---|---|
| 仿真建模 | 提供杆塔、绝缘子和巡检载体模型，以及巡检与数据生成两套 Gazebo 世界。 |
| 仿真数据生成 | 随机改变绝缘子距离、水平/垂直位置、滚转/俯仰/偏航；用相机参数将已知三维包围盒投影为二维 YOLO 标签。 |
| 数据扩充 | 在保持标签不变的前提下加入色相、饱和度、亮度、模糊和高斯噪声扰动，并生成少量空标签负样本。 |
| 仿真训练 | 使用 YOLOv10s 训练单类别 `insulator` 模型；支持 epoch、输入尺寸、批大小、设备、工作线程、早停和导出路径配置。 |
| 真实场景迁移 | 下载 CPLID，将 VOC 标注转换为双类别 YOLO 数据集，分层划分训练/验证集，并从仿真权重迁移训练。 |
| 缺陷识别 | 真实场景权重识别 `insulator` 与 `missing_disc` 两类；仅将 `missing_disc` 作为缺陷告警类别。 |
| 多源推理 | 统一检测节点可订阅任意 ROS 2 图像话题；内置媒体源节点可将单图、图片目录或视频发布成图像话题。 |
| 仿真巡检 | 自动巡检节点沿线路方向往返发布速度指令；可启动 Gazebo、检测节点和可选带框图窗口。 |
| 结果交付 | 发布 JSON 检测结果、带框图像、单帧耗时、缺陷告警；按冷却时间保存缺陷 JPG 与对应 JSON。 |
| 运行观测 | 可选终端监视器累计检测消息、带框图像、目标数、缺陷事件、平均推理耗时和异常消息数。 |
| 运行诊断 | 诊断脚本列出 ROS 2 节点，并检查关键话题是否存在及其发布者、订阅者信息。 |

## 系统架构

```text
仿真训练链路
Gazebo 世界 + 相机
  -> 随机姿态与自动标注
  -> datasets/insulator_sim
  -> YOLOv10s 仿真训练
  -> models/insulator_yolov10s.pt

真实场景训练链路
CPLID 原始图像与 VOC 标注
  -> 分层划分与 YOLO 双类别转换
  -> datasets/cplid_yolo
  -> 从仿真权重微调
  -> models/insulator_defect_yolov10s.pt

在线巡检链路
Gazebo 相机 / ROS 2 相机 / 图片 / 目录 / 视频
  -> /InsuLens/*/image_raw
  -> YOLOv10 检测节点
  -> 检测 JSON、带框图、耗时、缺片告警、JPG + JSON 证据
  -> 可选终端监视器与话题诊断
```

ROS 2 软件包职责如下。

| 软件包 | 职责 | 可执行入口 |
|---|---|---|
| `InsuLens_gazebo` | Gazebo 世界、模型、自动标注数据生成和自动往返巡检 | `generate_dataset`、`patrol` |
| `InsuLens_perception` | YOLOv10 训练、ROS 2 检测、真实媒体发布和终端监视 | `train_yolov10`、`detector`、`image_source`、`inspection_monitor` |
| `InsuLens_bringup` | 完整仿真巡检与真实图片/视频推理启动编排 | `inspection_sim.launch.py`、`real_image_demo.launch.py` |

## 环境与安装

### 运行环境

- Ubuntu 22.04
- ROS 2 Humble
- Gazebo Classic 11
- 系统 Python 3.10
- NVIDIA GPU（推荐）或 CPU

ROS 2 Humble 的 Python 绑定依赖系统 Python 3.10。`setup_env.sh` 创建带
`--system-site-packages` 的 `.venv`，以复用 ROS 2 的 Python 包；不要在 ROS
节点中切换到 Conda Python 3.12。

安装基础 ROS/Gazebo 依赖：

```bash
sudo apt update
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-cv-bridge \
  ros-humble-rqt-image-view python3.10-venv xvfb
```

Python 依赖由 `requirements.txt` 管理，包括 Ultralytics、NumPy、OpenCV 和
PyYAML。GPU 训练时，环境脚本会在系统中没有 PyTorch 时安装 CUDA 12.1 对应的
PyTorch 与 Torchvision；如需使用其他 CUDA 版本，请先自行安装匹配的 PyTorch。

### 初始化与编译

```bash
cd /root/InsuLens
chmod +x scripts/*.sh
./scripts/setup_env.sh
./scripts/build.sh

source /opt/ros/humble/setup.bash
source .venv/bin/activate
source install/setup.bash
```

`build.sh` 使用 `colcon build --symlink-install` 构建三个 ament Python 软件包。
之后每次重新打开终端，都应重新执行上述三个 `source` 命令。

## 完整工作流

### 1. 自动生成仿真训练集

默认生成 1200 张图像：

```bash
./scripts/generate_dataset.sh 1200
```

脚本会启动 `generate_dataset.launch.py`。有桌面环境时打开 Gazebo；无 `DISPLAY`
且已安装 `xvfb-run` 时，自动使用无界面模式。生成结束后，启动文件会关闭 Gazebo。

输出结构：

```text
datasets/insulator_sim/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

数据生成器固定每 5 张取 1 张作为验证集，其余作为训练集。默认随机种子为 42，
负样本比例为 0.08；负样本会生成空标签文件。自动标注使用相机 `CameraInfo` 中的
内参，将已知 0.8 × 0.8 × 2.4 米的绝缘子三维包围盒投影至图像平面；无法形成有效
目标框或目标框过小的帧会被跳过并重新采样。

如需直接指定输出目录、样本数、种子或窗口开关，可使用：

```bash
ros2 launch InsuLens_gazebo generate_dataset.launch.py \
  output_dir:=/data/insulator_sim num_samples:=2000 seed:=42 gui:=false
```

### 2. 训练仿真域绝缘子模型

默认训练 60 个 epoch：

```bash
./scripts/train.sh 60
```

训练使用 `yolov10s.yaml`，输入尺寸 640、批大小 16、设备 `0`、工作线程 8、
早停耐心值 15，并启用余弦学习率、水平翻转、平移、缩放、轻量 MixUp 和最后
10 个 epoch 关闭 Mosaic。训练结束后会将最佳权重复制到：

```text
runs/insulator_yolov10s/weights/best.pt
models/insulator_yolov10s.pt
```

也可直接调用训练入口以覆盖全部参数：

```bash
ros2 run InsuLens_perception train_yolov10 --help
```

可配置参数为 `--data`、`--model`、`--epochs`、`--imgsz`、`--batch`、`--device`、
`--workers`、`--project`、`--name`、`--patience` 和 `--export`。当 `--model` 以
`.pt` 结尾时，Ultralytics 按预训练/迁移模式加载该权重；使用 `.yaml` 时从网络
结构开始训练。

### 3. 下载并转换 CPLID 真实场景数据

```bash
./scripts/download_cplid.sh
```

脚本下载 CPLID 源仓库压缩包，解压至 `datasets/raw/cplid/`，随后执行
`prepare_cplid.py`。转换程序具有以下行为：

- 读取正常绝缘子与缺陷绝缘子图像及其 VOC XML 标注；
- 将完整绝缘子映射为类别 0：`insulator`；
- 将上游 `defect` 标注映射为类别 1：`missing_disc`；
- 对正常与缺陷两组图像分别以固定随机种子分层切分，默认验证比例为 20%；
- 复制图像、写出 YOLO 标签、清除同一输出目录下不再属于当前划分的旧文件；
- 写出 `data.yaml` 与 `dataset_metadata.json`，后者保存来源、提交版本、引用信息、
  数据限制、类别映射、样本计数和随机种子。

默认划分为 678 张训练图和 170 张验证图。对应目录为：

```text
datasets/cplid_yolo/
├── data.yaml
├── dataset_metadata.json
├── images/{train,val}/
└── labels/{train,val}/
```

需要使用本地 CPLID 副本或调整划分时，可直接调用：

```bash
python scripts/prepare_cplid.py \
  --source /path/to/InsulatorDataSet-master \
  --output /data/cplid_yolo --val-ratio 0.2 --seed 42
```

### 4. 训练真实场景缺片模型

```bash
./scripts/train_real_defect.sh 50
```

该脚本会先检查 `datasets/cplid_yolo/data.yaml` 是否存在，然后从
`models/insulator_yolov10s.pt` 初始化，使用 768 像素输入、批大小 8、设备 `0`、
工作线程 8，训练 50 个 epoch。输出目录与最终运行权重为：

```text
runs/insulator_defect_yolov10s/weights/best.pt
models/insulator_defect_yolov10s.pt
```

真实域模型有两个类别：`insulator` 与 `missing_disc`。较大的输入尺寸用于尽量保留
远距离航拍图像中的小缺片纹理。

## 运行与接入方式

### Gazebo 自动巡检

最简方式：

```bash
./scripts/run_sim.sh
```

脚本默认加载 `models/insulator_yolov10s.pt`。有图形环境时启动 Gazebo 与
`rqt_image_view`；无图形环境时通过 Xvfb 以无界面方式启动，并关闭图像窗口。
可以传入其他仿真权重路径：

```bash
./scripts/run_sim.sh /path/to/insulator_yolov10s.pt
```

完整启动文件支持下列参数：

```bash
ros2 launch InsuLens_bringup inspection_sim.launch.py \
  model_path:=/root/InsuLens/models/insulator_yolov10s.pt \
  device:=cuda:0 gui:=true visualize:=true monitor:=true
```

- `model_path`：仿真域检测权重路径；
- `device`：Ultralytics 推理设备，例如 `cuda:0` 或 `cpu`；
- `gui`：是否显示 Gazebo 图形界面；
- `visualize`：是否启动 `rqt_image_view` 显示 `/InsuLens/detection_image`；
- `monitor`：是否启动终端巡检监视器。

仿真启动文件会设置 `GAZEBO_MODEL_PATH`，依次启动 Gazebo、自动往返巡检节点、
YOLOv10 检测节点，以及可选图像窗口和监视器。巡检节点默认速度为 0.65，单程持续
58 秒，并在每个单程结束后反向。单独启动时可覆盖其参数：

```bash
ros2 run InsuLens_gazebo patrol --ros-args -p speed:=0.8 -p leg_duration:=45.0
```

### 图片、目录或视频检测

```bash
ros2 launch InsuLens_bringup real_image_demo.launch.py \
  source:=/path/to/image_or_directory_or_video \
  model_path:=/root/InsuLens/models/insulator_defect_yolov10s.pt \
  device:=cuda:0 visualize:=true monitor:=true
```

`real_image_demo.launch.py` 启动媒体源节点和检测节点。媒体源支持 `.jpg`、`.jpeg`、
`.png`、`.bmp`、`.tif`、`.tiff` 单图或图片目录；其他可由 OpenCV 打开的文件按视频
处理。图片与视频默认以 5 FPS 发布，并在末尾循环。

直接启动媒体源时，可设置来源、发布话题、帧编号、发布频率和循环开关：

```bash
ros2 run InsuLens_perception image_source --ros-args \
  -p source:=/path/to/media -p topic:=/InsuLens/real_camera/image_raw \
  -p frame_id:=real_camera_optical_frame -p fps:=5.0 -p loop:=true
```

### 接入现有 ROS 2 相机

检测器可订阅任意 `sensor_msgs/Image` 话题。只启动检测器并指定相机话题：

```bash
ros2 launch InsuLens_perception detector.launch.py \
  model_path:=/root/InsuLens/models/insulator_defect_yolov10s.pt \
  device:=cuda:0 image_topic:=/your_camera/image_raw
```

当请求 `cuda:*` 但 PyTorch 检测不到可用 CUDA 时，检测器会记录警告并自动回退到
CPU。若模型文件不存在，节点会终止并提示先训练或提供正确权重路径。

### 终端监视与运行诊断

在任意完整启动命令中加入 `monitor:=true`，即可启动 `inspection_monitor`。它订阅
检测 JSON、缺陷告警和带框图像，每 5 秒输出累计检测消息数、带框图像数、检测目标
总数、缺陷事件数、有效检测消息的平均推理耗时及无法解析的消息数。收到缺陷告警时，
会额外输出本帧缺陷目标数量。

也可以在已运行的系统中单独启动，并调整统计周期或订阅话题：

```bash
ros2 run InsuLens_perception inspection_monitor --ros-args \
  -p status_period_sec:=2.0 \
  -p detections_topic:=/InsuLens/detections \
  -p alerts_topic:=/InsuLens/defect_alerts \
  -p annotated_topic:=/InsuLens/detection_image
```

诊断脚本检查 `/InsuLens/camera/image_raw`、`/InsuLens/detection_image`、
`/InsuLens/detections`、`/InsuLens/defect_alerts` 和 `/InsuLens/inference_ms`：

```bash
./scripts/diagnose_runtime.sh
```

该脚本要求 ROS 2 图已经可访问；若没有运行中的节点或终端未加载 ROS 2 环境，会给出
错误信息。若需要测量实际图像发布频率，请在另一个终端执行：

```bash
ros2 topic hz /InsuLens/detection_image
```

## ROS 2 接口与数据格式

### 节点与话题

| 节点 | 输入 | 输出/作用 |
|---|---|---|
| `insulator_dataset_generator` | Gazebo 相机图像、相机内参、`/gazebo/set_entity_state` 服务 | 随机移动训练绝缘子，保存图片和 YOLO 标签。 |
| `inspection_patrol` | 无 | 向 `/InsuLens/cmd_vel` 发布往返巡检 `Twist`。 |
| `real_image_source` | 本地图片、目录或视频 | 向 `/InsuLens/real_camera/image_raw` 发布 `Image`。 |
| `yolov10_insulator_detector` | 配置的图像话题 | 发布检测 JSON、带框图、告警、耗时和缺陷证据。 |
| `inspection_monitor` | 检测 JSON、告警、带框图 | 在终端输出累计运行状态。 |

| 话题 | 类型 | 发布方 | 含义 |
|---|---|---|---|
| `/InsuLens/camera/image_raw` | `sensor_msgs/Image` | Gazebo 相机 | 仿真巡检图像。 |
| `/InsuLens/camera/camera_info` | `sensor_msgs/CameraInfo` | Gazebo 相机 | 仿真数据生成使用的相机内参。 |
| `/InsuLens/real_camera/image_raw` | `sensor_msgs/Image` | `real_image_source` | 真实媒体源图像。 |
| `/InsuLens/detection_image` | `sensor_msgs/Image` | 检测器 | 已绘制检测框与类别的图像。 |
| `/InsuLens/detections` | `std_msgs/String` | 检测器 | 每帧完整检测 JSON。 |
| `/InsuLens/defect_alerts` | `std_msgs/String` | 检测器 | 仅检测到缺陷类别时发布的 JSON。 |
| `/InsuLens/inference_ms` | `std_msgs/Float32` | 检测器 | 单帧端到端模型预测耗时，单位为毫秒。 |
| `/InsuLens/cmd_vel` | `geometry_msgs/Twist` | 自动巡检节点 | Gazebo 巡检运动指令。 |

### 检测 JSON

`/InsuLens/detections` 的消息体是 UTF-8 JSON，结构如下；`detections` 可以为空列表。
`/InsuLens/defect_alerts` 使用相同结构，但只保留 `is_defect: true` 的检测项。

```json
{
  "stamp": {"sec": 0, "nanosec": 0},
  "frame_id": "camera_frame",
  "inference_ms": 10.7,
  "defect_detected": true,
  "detections": [
    {
      "class_id": 1,
      "class_name": "missing_disc",
      "is_defect": true,
      "confidence": 0.914,
      "bbox_xyxy": [100.0, 80.0, 160.0, 140.0]
    }
  ]
}
```

### 缺陷证据

只要当前帧包含 `defect_classes` 中的类别，检测器就会发布一次告警。若
`save_defect_evidence` 为真，且距离上次保存已超过 `evidence_cooldown_sec`，
检测器会在 `evidence_dir` 中写入一对同名文件：

```text
inspection_results/
├── defect_YYYYMMDD_HHMMSS_mmm.jpg
└── defect_YYYYMMDD_HHMMSS_mmm.json
```

JPG 是带框图像，JSON 是该缺陷事件的告警载荷。证据保存冷却只限制磁盘写入，
不会抑制 `/InsuLens/defect_alerts` 的发布。

## 参数说明

检测器的默认参数位于
`src/InsuLens_perception/config/detector.yaml`；启动文件可通过 ROS 参数覆盖。

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `model_path` | `/root/InsuLens/models/insulator_defect_yolov10s.pt` | YOLOv10 权重文件路径。 |
| `image_topic` | `/InsuLens/camera/image_raw` | 输入图像话题。 |
| `annotated_topic` | `/InsuLens/detection_image` | 带框图像输出话题。 |
| `detections_topic` | `/InsuLens/detections` | 检测 JSON 输出话题。 |
| `confidence` | `0.35` | 置信度阈值。 |
| `iou` | `0.55` | 非极大值抑制的 IoU 阈值。 |
| `image_size` | `768` | YOLO 推理输入尺寸。 |
| `device` | `cuda:0` | 推理设备；也可设为 `cpu`。 |
| `frame_stride` | `1` | 每隔多少输入帧执行一次检测。 |
| `defect_classes` | `[missing_disc]` | 触发告警和证据保存的类别集合。 |
| `save_defect_evidence` | `true` | 是否保存缺陷证据。 |
| `evidence_dir` | `/root/InsuLens/inspection_results` | 证据目录。 |
| `evidence_cooldown_sec` | `2.0` | 同类证据的最小保存间隔，单位为秒。 |

示例：降低检测频率、仅发布结果而不落盘：

```bash
ros2 run InsuLens_perception detector --ros-args \
  -p image_topic:=/your_camera/image_raw -p frame_stride:=3 \
  -p save_defect_evidence:=false -p device:=cpu
```

## 输出、模型与可追溯性

| 产物 | 默认位置 | 生成方式 |
|---|---|---|
| 仿真训练数据 | `datasets/insulator_sim/` | `generate_dataset.sh` 或数据生成启动文件。 |
| CPLID YOLO 数据与元数据 | `datasets/cplid_yolo/` | `download_cplid.sh` / `prepare_cplid.py`。 |
| 仿真训练记录 | `runs/insulator_yolov10s/` | `train.sh`。 |
| 真实场景训练记录 | `runs/insulator_defect_yolov10s/` | `train_real_defect.sh`。 |
| 仿真域权重 | `models/insulator_yolov10s.pt` | 仿真训练最佳权重副本。 |
| 真实域缺陷权重 | `models/insulator_defect_yolov10s.pt` | 真实场景训练最佳权重副本。 |
| 模型卡 | `models/README.md` | 记录结构、类别、数据、指标、文件大小与 SHA-256。 |
| 缺陷事件证据 | `inspection_results/` | 检测器根据告警和冷却策略写入。 |

`.gitignore` 排除了数据集、训练目录、构建目录、运行日志、虚拟环境、模型二进制、
ONNX/TensorRT 文件、ROS bag、数据库和巡检证据。发布模型时应使用 Git LFS，并在
`models/README.md` 中更新权重来源、许可证和 SHA-256。

## 目录结构

```text
InsuLens/
├── src/
│   ├── InsuLens_gazebo/
│   │   ├── InsuLens_gazebo/dataset_generator.py   # 自动标注与仿真数据保存
│   │   ├── InsuLens_gazebo/patrol.py              # 自动往返巡检
│   │   ├── launch/generate_dataset.launch.py
│   │   ├── worlds/                              # 巡检与数据生成世界
│   │   └── models/                              # 杆塔、绝缘子、巡检载体
│   ├── InsuLens_perception/
│   │   ├── InsuLens_perception/detector_node.py   # YOLOv10 ROS 2 检测器
│   │   ├── InsuLens_perception/image_source.py    # 单图/目录/视频发布器
│   │   ├── InsuLens_perception/inspection_monitor.py
│   │   ├── InsuLens_perception/train.py           # 通用训练入口
│   │   ├── config/detector.yaml
│   │   └── launch/detector.launch.py
│   └── InsuLens_bringup/
│       └── launch/                             # 完整仿真与真实媒体启动文件
├── scripts/
│   ├── setup_env.sh
│   ├── build.sh
│   ├── generate_dataset.sh
│   ├── train.sh
│   ├── download_cplid.sh
│   ├── prepare_cplid.py
│   ├── train_real_defect.sh
│   ├── run_sim.sh
│   └── diagnose_runtime.sh
├── models/README.md                             # 模型卡
├── deliverables/                                # 汇报材料与演讲稿
└── requirements.txt
```

## 结果、边界与数据说明

### 当前固定划分结果

真实场景缺陷模型的验证集包含 170 张图、269 个绝缘子框和 50 个缺片框。模型卡中
记录的最优权重结果如下：

| 类别 | 精确率 | 召回率 | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| 全部类别平均 | 0.791 | 0.844 | 0.882 | 0.530 |
| `insulator` | 0.668 | 0.729 | 0.785 | 0.484 |
| `missing_disc` | 0.914 | 0.960 | 0.980 | 0.575 |

仿真域模型卡记录的留出仿真集结果为 mAP50 0.976、mAP50-95 0.647。上述均为仓库
已记录的数据划分结果，不是独立电力线路的现场验收指标。

### 能力边界

- 当前缺陷模型只定义 `missing_disc`（缺片）一种故障，不能用于宣称识别裂纹、
  污损、闪络、锈蚀等未训练类别；
- CPLID 的正常绝缘子图像来自真实无人机拍摄，但缺陷图像使用合成缺损绝缘子叠加到
  输电场景背景中得到；
- 训练集和验证集同源，尚未完成跨线路、跨地区、跨相机或跨天气的外部验证；
- 尚未完成现场飞行验收，阈值选择、漏检和误报的业务成本也未量化；
- 当前仓库记录的是桌面 GPU 推理结果，未验证 Jetson 等边缘端的功耗、延迟和稳定性；
- 因此系统适合课程演示、算法原型和现场数据预筛，不能替代电力安全专业复核或
  现场作业决策。

### CPLID 使用说明

CPLID（Chinese Power Line Insulator Dataset）的上游仓库未声明明确的数据许可证。
项目不将原始图像纳入 Git 历史；使用时应引用原始论文，不应重新分发图像。数据集
引用信息见 [models/README.md](models/README.md) 与转换生成的
`datasets/cplid_yolo/dataset_metadata.json`。

项目源代码采用 MIT License。
