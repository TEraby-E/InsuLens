# InsuLens：ROS 2 + YOLOv10 绝缘子巡检系统

> **Web 巡检交付（图片、视频、报告）**：项目现提供独立于 ROS 的 `FastAPI` Web
> 应用。它可上传图片或视频，自动读取任意兼容检测模型的类别元数据，执行逐帧跟踪、小目标
> 切片推理融合，并导出 JSON/CSV 巡检报告。以下「Web 巡检系统」章节是从零复现该交付的完整流程。

## Web 巡检系统

### 1. 模型驱动的动态分类接口

Web API 不再固定类别数量、标签名称或类别顺序。加载 Ultralytics 目标检测权重后，后端读取
模型的 `names` 元数据并生成 `class_schema`；前端的类别卡片、颜色、统计、报告文本和导出文件
都由该 schema 动态渲染。仓库内已有模型可直接验证 1 类、2 类和 6 类三种情况：

| 示例权重 | 模型输出类别 |
| --- | --- |
| `insulator_yolov10s.pt` | `insulator` |
| `insulator_defect_yolov10s.pt` | `insulator`, `missing_disc` |
| `insulator_six_class_yolov10s.pt` | `normal`, `broken`, `crack`, `pollution`, `missing`, `flashover` |

Web 后端的 `InsulatorDetector` 会以指定模型进行全图推理；对大分辨率
画面再做重叠切片推理，以类感知 NMS 融合结果，专门降低小绝缘子漏检。视频逐帧检测后由
`IoUTracker` 维护类别一致的 Track ID。统计使用任务期间观察到的唯一轨迹，避免将同一个目标
在每一帧重复累计。界面实时展示处理 FPS、总数、动态类别数量、每帧检测数据和平均置信度。

接口默认原样保留模型标签，不进行绝缘子业务类别重映射。确有业务需要时，可通过环境变量配置
类别别名、中文显示名或一个低置信度推断类；这些都是可选元数据，不再是模型接入门槛。模型未
加载、没有 `names` 元数据，或仅为整图分类任务时，上传巡检才返回 `503`。

### 2. 环境配置

```bash
git clone <your-repository-url> InsuLens
cd InsuLens
python3 -m venv .venv
source .venv/bin/activate                    # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
# GPU 请先按 PyTorch 官网安装与 CUDA 匹配的 torch/torchvision，再安装本项目依赖。
pip install -r requirements.txt
pip install -e src/insulens_perception
```

运行视频需要 OpenCV 能使用 FFmpeg 或系统的视频编码器；Linux 可安装 `ffmpeg`。验证环境：

```bash
python -c "import cv2, fastapi, ultralytics; print('runtime ready')"
```

### 3. 绝缘子五分类数据准备与训练示例

本节是项目自带绝缘子模型的训练流程，不是 Web 前端接口的固定 schema；其他数据集与类别数量
训练出的目标检测权重同样可以接入前端。

使用 YOLO 数据集布局，所有标签坐标为归一化的 `class x_center y_center width height`：

```text
datasets/insulator_six_class/
  images/{train,val}/xxx.jpg
  labels/{train,val}/xxx.txt
  data.yaml
  sources.jsonl
  dataset_metadata.json
```

`data.yaml` 必须与该训练任务的标签 ID 一致；Web API 会在加载权重后自动读取这份类别顺序：

```yaml
path: datasets/insulator_six_class
train: images/train
val: images/val
names: [normal, broken, crack, pollution, missing]
```

构建器严格复制 CPLID 原图字节并重映射其已有标注，不添加颜色通道、线条、污层、电弧或
其他像素级渲染。CPLID 上游固定为
[`InsulatorDataSet@1f6349f`](https://github.com/InsulatorData/InsulatorDataSet/tree/1f6349f619237344d49905090ecf2704505394a4)，
其可用标注仅映射为 `normal` 与 `missing`。`broken`、`crack`、`pollution` 必须从另行授权、
可追溯且已标注的真实数据导入；在三类任一缺失时训练入口会主动拒绝训练。`flashover`
始终不进入训练图片或标签：

```bash
PYTHONPATH=src/insulens_perception python -m insulens_perception.six_class_dataset \
  --source datasets/cplid_yolo --output datasets/insulator_six_class \
  --per-split 120 --seed 42
```

导入网络数据时，只能使用许可清楚且提供标注的公开数据集；将来源、许可证、类别映射和划分
写入数据清单，不能把无标注网络图片直接加入监督训练。每类应有独立真实训练/验证/测试样本，
并按线路、拍摄批次或视频片段划分。使用 `P2 + Coordinate Attention` 小目标版本训练：

```bash
python -m insulens_perception.train \
  --data datasets/insulator_six_class/data.yaml \
  --model yolov10s.pt --small-object-model \
  --imgsz 960 --epochs 120 --batch 8 --device 0 \
  --name insulator_five_class_yolov10s \
  --export models/insulator_five_class_yolov10s.pt
```

训练后应在独立真实测试集报告五个模型类别的 AP、mAP50-95、召回率和混淆矩阵；另以独立
闪络案例验证排除式阈值的误报率与漏报率。仅当验收集指标满足现场阈值时才更新 Web 权重。

### 4. 剪枝、轻量化与部署

绝缘子巡检是跨帧重复查找高相似度目标的任务，优化策略为：

1. 采用 P2 检测头、切片推理与检测融合，优先提升小目标召回；
2. Web 视频使用 IoU 跟踪，稳定 ID 并基于轨迹计数，减少重复业务判定；
3. 以基线模型为起点对卷积输出通道执行结构化 L2 剪枝，进行恢复微调；
4. 导出 ONNX（或 TensorRT/OpenVINO）并可进行经代表性校准集验证的 INT8 量化；
5. 对比剪枝前后的模型大小、每帧延迟、吞吐和五类 mAP/召回，不以单一模型体积作为上线标准。

```bash
python -m insulens_perception.optimize_model \
  --weights models/insulator_defect_yolov10s.pt \
  --data datasets/insulator_fault/data.yaml \
  --sparsity 0.20 --epochs 25 --imgsz 960 --format onnx --int8
```

每一个剪枝比例都必须重新执行 `model.val(data=...)`；若 `Missing`、`Damage` 或
`闪络烧痕` 召回下降超过业务阈值，应降低 `--sparsity` 或保留基线模型。`optimize_model.py`
输出状态字典和部署工件；生产部署须绑定导出工件、类别映射和相应版本的验证报告。

### 5. 启动 Web 服务

```bash
# 指向任意兼容的 Ultralytics 目标检测 .pt；类别数与标签名称不限。
export INSULENS_WEB_MODEL="$PWD/models/insulator_defect_yolov10s.pt"
python -m insulens_perception.web_app
# 浏览器打开 http://127.0.0.1:8080
```

可选环境变量：`PORT=8080` 修改端口；`INSULENS_RESULT_DIR=/data/inspections` 修改报告和
标注媒体输出目录。ROS 环境仍可按后续原有章节启动，不影响 Web 模式。

模型标签无需修改即可显示。需要业务别名或中文显示名时传入 JSON 对象：

```bash
export INSULENS_CLASS_ALIASES='{"missing_disc":"missing"}'
export INSULENS_CLASS_LABELS='{"insulator":"绝缘子","missing":"缺片"}'
```

低置信度推断类默认关闭，避免对无关模型做错误重分类。需要恢复特定业务规则时显式设置：

```bash
export INSULENS_INFERRED_CLASS=flashover
export INSULENS_INFERRED_CLASS_THRESHOLD=0.30
export INSULENS_INFERENCE_CANDIDATE_CONFIDENCE=0.05
```

### 6. 图片、视频检测与报告导出

1. 打开 Web 页面，在「上传巡检素材」选择 `jpg/png/bmp` 图片或 `mp4/avi/mov/mkv` 视频；
2. 点击「开始巡检」。视频会逐帧执行全图/切片检测、NMS 融合和 Track ID 关联；
3. 完成后查看标注图片/视频、FPS、平均置信度及类别统计；
4. 点击「导出 JSON 报告」或「导出 CSV 报告」。

每个任务存入 `inspection_results/web/inspection_<UTC时间>_<随机ID>/`。CSV 包含检测数量、
模型实际类别及数量、平均置信度与 FPS。JSON 额外保存动态 `class_schema`、后端类型、帧数、
耗时和输出媒体名；视频 JSON 还保存逐帧 `track_id`、边界框、类别和置信度，便于审计与复核。

### 7. 端到端复现和测试

```bash
# 核心跟踪、报告和既有小目标单元测试
PYTHONPATH=src/insulens_perception pytest -q src/insulens_perception/test

# 启动服务后另开终端，使用 API 上传复现
curl -F "upload=@path/to/inspection.mp4" http://127.0.0.1:8080/api/inspect
curl http://127.0.0.1:8080/api/health
```

API 成功响应含 `job_id`、`class_schema`、`detection_total`、`category_counts`、`fps`、
`average_confidence`、`output_media`、`report_json`、`report_csv` 与 `download_base`。
下载接口为 `/api/jobs/{job_id}/{artifact}`。先在合成或已标注素材上验证全流程，再以独立
现场测试集确认精度与吞吐后部署。

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
| `insulens_gazebo` | Gazebo 世界、模型、自动标注数据生成和自动往返巡检 | `generate_dataset`、`patrol` |
| `insulens_perception` | YOLOv10 训练、ROS 2 检测、真实媒体发布和终端监视 | `train_yolov10`、`detector`、`image_source`、`inspection_monitor` |
| `insulens_bringup` | 完整仿真巡检与真实图片/视频推理启动编排 | `inspection_sim.launch.py`、`real_image_demo.launch.py` |

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
ros2 launch insulens_gazebo generate_dataset.launch.py \
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
ros2 run insulens_perception train_yolov10 --help
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
ros2 launch insulens_bringup inspection_sim.launch.py \
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
ros2 run insulens_gazebo patrol --ros-args -p speed:=0.8 -p leg_duration:=45.0
```

### 图片、目录或视频检测

```bash
ros2 launch insulens_bringup real_image_demo.launch.py \
  source:=/path/to/image_or_directory_or_video \
  model_path:=/root/InsuLens/models/insulator_defect_yolov10s.pt \
  device:=cuda:0 visualize:=true monitor:=true
```

`real_image_demo.launch.py` 启动媒体源节点和检测节点。媒体源支持 `.jpg`、`.jpeg`、
`.png`、`.bmp`、`.tif`、`.tiff` 单图或图片目录；其他可由 OpenCV 打开的文件按视频
处理。图片与视频默认以 5 FPS 发布，并在末尾循环。

直接启动媒体源时，可设置来源、发布话题、帧编号、发布频率和循环开关：

```bash
ros2 run insulens_perception image_source --ros-args \
  -p source:=/path/to/media -p topic:=/InsuLens/real_camera/image_raw \
  -p frame_id:=real_camera_optical_frame -p fps:=5.0 -p loop:=true
```

### 接入现有 ROS 2 相机

检测器可订阅任意 `sensor_msgs/Image` 话题。只启动检测器并指定相机话题：

```bash
ros2 launch insulens_perception detector.launch.py \
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
ros2 run insulens_perception inspection_monitor --ros-args \
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
`src/insulens_perception/config/detector.yaml`；启动文件可通过 ROS 参数覆盖。

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `model_path` | `auto` | 自动查找工作空间 `models/insulator_defect_yolov10s.pt`；也可指定权重绝对路径。 |
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
| `evidence_dir` | `inspection_results` | 证据目录；相对路径基于节点启动时的工作目录。 |
| `evidence_cooldown_sec` | `2.0` | 同类证据的最小保存间隔，单位为秒。 |

示例：降低检测频率、仅发布结果而不落盘：

```bash
ros2 run insulens_perception detector --ros-args \
  -p image_topic:=/your_camera/image_raw -p frame_stride:=3 \
  -p save_defect_evidence:=false -p device:=cpu
```

该命令会自动查找工作空间 `models/insulator_defect_yolov10s.pt`。如果模型位于其他
位置，可设置环境变量 `INSULENS_MODEL_PATH`，或增加
`-p model_path:=/absolute/path/to/model.pt`。

## 第二阶段：小目标检测、坐标注意力与 Web 看板

YOLOv10 是 **anchor-free** 检测器。本项目不会生成或替换 anchor；而是通过 GT
尺度聚类决定 P2 检测头、TAL 实验候选和增强范围。

### 1. 标注尺度分析

```bash
source install/setup.bash
ros2 run insulens_perception analyze_small_objects \
  --data datasets/cplid_yolo/data.yaml --imgsz 768 \
  --output reports/small_object
```

命令会生成 `small_object_scale_report.json` 和 Markdown 附录，包含类别统计、K-means++
尺度簇、P2 建议以及 TAL `topk=6/10/13` 候选。结果仅用于 anchor-free YOLOv10 的
结构和超参数决策，不输出 anchor 文件。

### 2. P2 + Coordinate Attention 训练和消融

基线继续使用现有的 `yolov10s.yaml`。要训练项目内版本控制的四尺度 P2 + CA 模型，使用：

```bash
ros2 run insulens_perception train_yolov10 \
  --data datasets/cplid_yolo/data.yaml --small-object-model \
  --tal-topk 6 --imgsz 768 --batch 8 --device 0 \
  --project runs --name insulator_p2_ca_topk6
```

每次训练会将模型选择和 TAL 候选写入 `small_object_experiment.json`。建议完成
baseline、`+P2`、`+P2+CA` 与 `topk=6/10/13` 的对照，记录 mAP@0.5、mAP@0.5:0.95、
AP_small 和 FPS。P2 会增加显存和延迟；低算力部署可直接继续使用基线模型。

### 3. 重叠切片推理

`tiled_inference` 默认关闭，保证当前实时推理链路不变。对无人机高分辨率图像可开启：

```bash
ros2 run insulens_perception detector --ros-args \
  -p tiled_inference.enabled:=true \
  -p tiled_inference.tile_size:=1024 \
  -p tiled_inference.overlap:=0.20 \
  -p tiled_inference.fusion_iou:=0.55
```

节点会将每个切片的检测框恢复到原图坐标，并按类别执行加权框融合；随后仍发布既有
`/insulens/detections`、`/insulens/defect_alerts`、`/insulens/detection_image` 和证据文件。

### 4. rosbridge Web 看板

安装并启动 `rosbridge_suite` 后，分别启动 ROS 检测演示、WebSocket 桥和静态看板：

```bash
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
ros2 launch insulens_perception web_dashboard.launch.py port:=8080
```

浏览器打开 `http://localhost:8080`，填写 rosbridge 地址（默认 `ws://localhost:9090`）后连接。
看板会显示连接状态、检测帧数、目标数、推理延迟、缺陷告警、最新检测表格和
`sensor_msgs/Image` 格式的带框图像。rosbridge 或浏览器不可用时，不影响原有 ROS2
检测节点、证据保存或 `rqt_image_view` 演示。

## 输出、模型与可追溯性

### 可视化训练成果

仓库中的训练曲线、PR 曲线、混淆矩阵和验证集预测图可以汇总为一个自包含 HTML
报告，适合演示、答辩或直接分享：

```bash
cd /root/InsuLens
source .venv/bin/activate
python scripts/generate_training_report.py
```

报告输出到 `reports/training_report.html`。报告中的图片已嵌入 HTML，无需连同
`runs/` 目录一起复制；点击图表可以放大查看。

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
│   ├── insulens_gazebo/
│   │   ├── insulens_gazebo/dataset_generator.py   # 自动标注与仿真数据保存
│   │   ├── insulens_gazebo/patrol.py              # 自动往返巡检
│   │   ├── launch/generate_dataset.launch.py
│   │   ├── worlds/                              # 巡检与数据生成世界
│   │   └── models/                              # 杆塔、绝缘子、巡检载体
│   ├── insulens_perception/
│   │   ├── insulens_perception/detector_node.py   # YOLOv10 ROS 2 检测器
│   │   ├── insulens_perception/image_source.py    # 单图/目录/视频发布器
│   │   ├── insulens_perception/inspection_monitor.py
│   │   ├── insulens_perception/train.py           # 通用训练入口
│   │   ├── config/detector.yaml
│   │   └── launch/detector.launch.py
│   └── insulens_bringup/
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
