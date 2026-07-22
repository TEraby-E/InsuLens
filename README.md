# Doggo：ROS 2 + YOLOv10 绝缘子巡检仿真

Doggo 使用 Gazebo 自动生成绝缘子数据完成第一阶段训练，再使用 CPLID
真实 UAV 输电场景进行 sim-to-real 微调，在 ROS 2 Humble 中实时检测绝缘子
及缺片故障。整个流程不依赖现成的绝缘子专用权重。

## 系统流程

```text
Gazebo 自动标注 -> YOLOv10s 仿真预训练 -> CPLID 真实场景微调
         -> insulator + missing_disc -> ROS 2 检测/告警/证据保存
```

最终模型包含两个类别：`insulator` 和 `missing_disc`。

## 当前交付能力

- 在杆塔、导线、山地/天空等真实输电背景中定位绝缘子串；
- 定位绝缘子串中的缺片区域，并把它作为独立缺陷事件告警；
- 接收 ROS 2 相机话题、单张图片、图片目录或视频；
- 发布检测框 JSON、带框图像、推理耗时和缺片告警；
- 自动保存缺陷带框 JPG 与同名 JSON，便于巡检结果复核；
- 使用仿真专用权重运行 Gazebo 无人机巡检，并可重复生成带自动标注的训练集。

## 能力边界

当前缺陷模型只定义了 `missing_disc`（缺片）一种故障，不能用于宣称识别
裂纹、污损、闪络、锈蚀等未训练类别。验证集与训练集均来自 CPLID，尚未在
独立线路或不同相机数据上完成生产级泛化验证；CPLID 的缺陷目标也属于合成
缺片样本。因此本模型适合课程演示、算法原型和现场数据预筛，投入实际巡检前
仍需使用目标线路采集并复核的样本继续微调与验收。

## 环境

- Ubuntu 22.04
- ROS 2 Humble
- Gazebo Classic 11
- Python 3.10
- NVIDIA GPU（推荐）或 CPU

ROS 2 Humble 必须使用系统 Python 3.10。本项目会创建带
`--system-site-packages` 的 `.venv`，不要在 ROS 节点中使用 Conda Python 3.12。

缺少基础 ROS/Gazebo 依赖时安装：

```bash
sudo apt update
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-cv-bridge \
  ros-humble-rqt-image-view python3.10-venv xvfb
```

## 安装和编译

```bash
cd /root/doggo
chmod +x scripts/*.sh
./scripts/setup_env.sh
./scripts/build.sh
source /opt/ros/humble/setup.bash
source .venv/bin/activate
source install/setup.bash
```

## 1. 生成仿真训练集

生成 1200 张图片：

```bash
./scripts/generate_dataset.sh 1200
```

输出目录：

```text
datasets/insulator_sim/
├── data.yaml
├── images/{train,val}/
└── labels/{train,val}/
```

Gazebo 中的绝缘子会改变距离、平移和三轴姿态。程序使用相机内参把已知
三维包围盒投影到图像，自动产生 YOLO 标注，并加入少量负样本、颜色、亮度、
模糊和噪声扰动。

## 2. 训练 YOLOv10s

默认训练 60 个 epoch：

```bash
./scripts/train.sh 60
```

权重输出：

```text
runs/insulator_yolov10s/weights/best.pt
models/insulator_yolov10s.pt
```

默认使用 Ultralytics 内置的 `yolov10s.yaml` 网络结构从仿真数据训练，
不依赖外部绝缘子权重，也不需要从 GitHub 下载预训练权重。如网络环境允许，
也可把训练参数 `--model` 改为 `yolov10s.pt` 进行预训练微调。

## 3. 准备真实输电场景数据

```bash
./scripts/download_cplid.sh
```

脚本下载 Chinese Power Line Insulator Dataset（CPLID），将 VOC 标注转换成
YOLO 双类别数据集，并按固定随机种子分成 678 张训练图和 170 张验证图。
原始 `defect` 类在本项目中明确映射为 `missing_disc`。

CPLID 的 600 张正常绝缘子图像来自真实 UAV 拍摄；248 张缺陷图像是数据集
作者把缺损绝缘子合成到输电场景背景中得到的。上游仓库没有声明明确的数据
许可证，因此原图不进入本项目 Git 历史，使用时应引用原论文且不要重新分发。

## 4. 训练真实场景缺陷模型

```bash
./scripts/train_real_defect.sh 50
```

该阶段从仿真权重迁移已有的 `insulator` 特征，把检测头扩展为双类别，并用
768 像素输入保留小缺片目标。最终权重输出到：

```text
models/insulator_defect_yolov10s.pt
```

固定种子验证集包含 170 张图、269 个绝缘子框和 50 个缺片框。最终最优权重
的验证结果如下：

| 类别 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| 全部类别平均 | 0.791 | 0.844 | 0.882 | 0.530 |
| `insulator` | 0.668 | 0.729 | 0.785 | 0.484 |
| `missing_disc` | 0.914 | 0.960 | 0.980 | 0.575 |

这些是 CPLID 内部固定划分的结果，不等同于独立线路现场验收指标。

## 5. 启动巡检仿真

```bash
./scripts/run_sim.sh
```

有桌面显示时脚本会打开 Gazebo 和检测图像窗口；无 `DISPLAY` 时会自动通过
Xvfb 运行无界面仿真，检测结果仍通过 ROS 2 话题发布。

也可以手动启动：

```bash
ros2 launch doggo_bringup inspection_sim.launch.py \
  model_path:=/root/doggo/models/insulator_yolov10s.pt \
  device:=cuda:0 visualize:=true
```

Gazebo 默认使用仿真专用权重，真实媒体启动文件默认使用真实缺陷权重。两者
共用同一个 ROS 2 检测节点，但分别保留各自域内的最佳效果。

## 6. 检测真实图片或视频

```bash
ros2 launch doggo_bringup real_image_demo.launch.py \
  source:=/path/to/real/images_or_video visualize:=true
```

`source` 可以是单张图片、图片目录或视频文件。接真实相机时也可以只启动
`doggo_perception/detector.launch.py`，并把 `image_topic` 指向相机话题。

主要话题：

| Topic | Type | 内容 |
|---|---|---|
| `/doggo/camera/image_raw` | `sensor_msgs/Image` | Gazebo 相机图像 |
| `/doggo/detection_image` | `sensor_msgs/Image` | YOLOv10 检测框图像 |
| `/doggo/detections` | `std_msgs/String` | JSON 检测结果 |
| `/doggo/defect_alerts` | `std_msgs/String` | 只在发现缺片时发布的告警 |
| `/doggo/inference_ms` | `std_msgs/Float32` | 单帧推理耗时 |
| `/doggo/cmd_vel` | `geometry_msgs/Twist` | 自动巡检运动指令 |

检测 JSON 包含时间戳、类别、置信度、`is_defect` 和 `xyxy` 像素坐标。
缺陷证据默认保存到 `inspection_results/`，每个事件包括带框 JPG 和 JSON。

## 参数

检测参数位于 `src/doggo_perception/config/detector.yaml`。常用参数包括：

- `confidence`：置信度阈值，默认 `0.35`；
- `image_size`：推理分辨率，默认 `768`；
- `device`：`cuda:0` 或 `cpu`；
- `frame_stride`：每隔多少帧执行一次检测。
- `defect_classes`：触发告警的类别，默认 `[missing_disc]`；
- `evidence_cooldown_sec`：缺陷证据保存间隔，默认 2 秒。

## 仓库说明

生成的数据集、训练结果和权重不会进入普通 Git 历史。发布模型时应使用
Git LFS，并在 `models/README.md` 中记录权重来源、许可证和 SHA-256。

项目采用 MIT License。
