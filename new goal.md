# 电力巡检 ROS2 + YOLOv10 小目标检测优化与系统架构规划方案

## 0. 关键修正说明（先看这个）

YOLOv10 沿用了 YOLOv8 的 **anchor-free** 范式：检测头对每个特征点直接回归框的偏移/尺寸（DFL 回归），标签分配靠 **Task-Aligned Assigner（TAL）** 动态选正样本，不存在预设的 anchor 形状。所以原计划里"K-means++ 重新聚类生成锚框"这一步，在 YOLOv10 上没有替换对象。

调整后的对应关系：

| 原目标（anchor-based思路） | YOLOv10（anchor-free）里的对应做法 |
|---|---|
| K-means++聚类生成更合适的anchor尺寸 | K-means++聚类分析GT框的尺度/宽高比分布 → 用结果指导**是否加P2检测头**、**TAL的topk等超参**、**多尺度训练范围** |
| 锚框覆盖小目标不足 → 直接换anchor | 特征点在浅层分辨率不够 → **加P2（stride=4）检测头**，让小目标有更多候选特征点参与分配 |
| — | **TAL超参调优**（topk/alpha/beta），增加小目标候选正样本数 |

下面方案里 K-means++ 的代码保留，但产出物是"尺度分布报告"而不是"anchor文件"。

---

## 1. 总体目标

1. 用坐标注意力（Coordinate Attention, CA）增强 YOLOv10 backbone/neck 在浅层的小目标特征表达
2. 用 K-means++ 对训练集 GT 框做尺度聚类分析，数据驱动地决定 P2 头和 TAL 超参
3. 补齐前处理（大图切片/小目标数据增强）和后处理（跨帧去重、置信度融合）
4. 用 ROS2 把整个感知流程包装成可演示的节点化系统（不是Web前后端分离，而是ROS2话题/节点架构，可选叠加一个轻量web看板）

假设：你在用 ultralytics 的 YOLOv10 训练接口（`pip install ultralytics`，yaml 配置模型结构），且课设需要在 ROS2 环境下跑通感知节点。如果你是拿官方 THU-MIG 仓库单独训练、再导出到ROS2侧推理，下面第2节的yaml/代码位置需要对应调整，思路不变。

---

## 2. 算法层面改进

### 2.1 GT框尺度分布分析（K-means++，供决策用）

目的：搞清楚"缺片"这类小目标和"绝缘子串"这类大目标的尺寸差异有多大，决定要不要加P2头、TAL的topk设多少合适。

```python
import numpy as np

def iou_dist(boxes, centroids):
    """boxes: (N,2) w,h   centroids: (k,2) w,h   返回 (N,k) 的 1-IoU 距离"""
    n, k = boxes.shape[0], centroids.shape[0]
    box_area = boxes[:, 0] * boxes[:, 1]
    cent_area = centroids[:, 0] * centroids[:, 1]
    inter_w = np.minimum(boxes[:, 0:1], centroids[:, 0].reshape(1, k))
    inter_h = np.minimum(boxes[:, 1:2], centroids[:, 1].reshape(1, k))
    inter = inter_w * inter_h
    union = box_area.reshape(n, 1) + cent_area.reshape(1, k) - inter
    return 1 - inter / (union + 1e-9)

def kmeans_pp_init(boxes, k, seed=42):
    rng = np.random.default_rng(seed)
    centroids = [boxes[rng.integers(len(boxes))]]
    for _ in range(1, k):
        d = iou_dist(boxes, np.array(centroids)).min(axis=1)
        probs = d ** 2
        probs /= probs.sum()
        centroids.append(boxes[rng.choice(len(boxes), p=probs)])
    return np.array(centroids)

def kmeans_iou(boxes, k, max_iter=300, seed=42):
    centroids = kmeans_pp_init(boxes, k, seed)
    for _ in range(max_iter):
        assign = iou_dist(boxes, centroids).argmin(axis=1)
        new_c = np.array([
            np.median(boxes[assign == i], axis=0) if np.any(assign == i) else centroids[i]
            for i in range(k)
        ])
        if np.allclose(new_c, centroids, atol=1e-6):
            break
        centroids = new_c
    return centroids, assign

# 用法：boxes 从标注文件里提取所有GT框的(w,h)（像素或归一化都行，保持一致即可）
# centroids, assign = kmeans_iou(boxes, k=6)
# 重点看：绝缘子类 vs 缺片类 是否分别聚成独立簇、缺片类的中位尺寸相对输入分辨率占比多大
```

**怎么用这个结果**：
- 如果缺片类的GT框中位边长 < 输入图像的 1/64（即小于stride=16对应的一个网格感受野量级），说明P3(stride=8)甚至P4都覆盖不到足够细粒度的特征点，**加P2检测头基本是必须的**，不是可选项。
- 把统计出的宽高比分布记录下来，训练时数据增强的 `scale`、`mosaic` 参数按这个范围设置，避免增强后把小目标缩没了。

### 2.2 新增P2小目标检测头

ultralytics 官方仓库里有 `yolov8-p2.yaml` 的模板（P2/stride4 + P3 + P4 + P5 四尺度输出），可以照着改一份 `yolov10-p2.yaml`：

- backbone 在 stride=4 的那一层（通常是第2个下采样后）额外引出一个分支
- neck部分让这个P2特征也参与 PAN 的上采样融合
- head部分从3个检测头（P3/P4/P5）变成4个（P2/P3/P4/P5）

代价：P2分辨率最高，计算量和显存占用明显上升（尤其训练时），课设时间紧的话建议先在小分辨率（如640）+ 小batch上验证有没有实质提升，再决定要不要保留。

### 2.3 坐标注意力模块（CA）

YOLOv10 backbone里已经有 PSA（Partial Self-Attention）模块，但PSA一般只放在SPPF之后的深层，做全局上下文建模；CA是轻量级的，可以插到浅层（P2/P3附近）做**位置敏感**的通道注意力，两者互补、不冲突。

```python
import torch
import torch.nn as nn

class h_swish(nn.Module):
    def forward(self, x):
        return x * torch.nn.functional.relu6(x + 3, inplace=True) / 6

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, 1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, 1)
        self.conv_w = nn.Conv2d(mip, oup, 1)

    def forward(self, x):
        identity = x
        n, c, h, w = x.shape
        x_h = self.pool_h(x)                       # n,c,h,1
        x_w = self.pool_w(x).permute(0, 1, 3, 2)    # n,c,w,1
        y = self.act(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_h * a_w
```

接入 ultralytics 代码库的三步：
1. 把 `CoordAtt` 类加到 `ultralytics/nn/modules/conv.py`（或新建 `attention.py`）
2. 在 `ultralytics/nn/modules/__init__.py` 里导出，`ultralytics/nn/tasks.py` 的 `parse_model` 函数里把 `CoordAtt` 加入可识别模块列表
3. 在 `yolov10-p2.yaml` 里，在backbone的P2、P3输出层之后各插一层 `[-1, 1, CoordAtt, [c1, c1]]`（c1保持通道数不变）

消融实验建议对照组：baseline / +P2头 / +P2头+CA(仅backbone) / +P2头+CA(backbone+neck)，看 mAP@0.5、mAP@0.5:0.95、以及**小目标专项指标 AP_small**（COCO格式评估自带，一定要单独看这个，总mAP可能被大目标"绝缘子串"的高精度掩盖掉小目标的提升）。

### 2.4 TAL 分配器调优

位置：`ultralytics/utils/tal.py` 的 `TaskAlignedAssigner`，默认 `topk=10, alpha=0.5, beta=6.0`。小目标候选特征点少，topk过大可能引入低质量正样本，过小可能欠拟合。建议做一组小范围实验（topk=6/10/13），配合2.1的尺度分布决定起始值。

### 2.5 前处理

- **大图切片推理**（类似 SAHI 思路）：无人机原图分辨率通常远大于640，直接resize会让缺片这种小目标丢失大量像素信息。做法：原图按重叠窗口切成多个子图分别推理，检测框映射回原图坐标后跨窗口做NMS合并。
- **小目标定向数据增强**：Mosaic/MixUp基础上，针对缺片类做 copy-paste（把标注好的缺片小图粘贴到不同背景的绝缘子图上），扩充稀缺样本。

### 2.6 后处理

- 用 Soft-NMS 或加权NMS替代硬NMS，减少小目标因重叠被误删
- **跨帧去重**：无人机连续帧会重复拍到同一个缺陷，需要基于世界坐标（见3.4）+ 缺陷类型做关联，避免同一缺陷被记成多条记录

---

## 3. ROS2 系统架构

### 3.1 整体数据流

```
相机驱动节点 (/drone/camera/image_raw)
        │
        ▼
yolo_detector_node  ──订阅image──> YOLOv10推理(TensorRT) ──发布──> /inspection/detections
        │                                                              │
        ▼(如有MAVROS)                                                  ▼
pose_sync_node (/mavros/local_position/pose) ──时间同步──> geo_localization_node
                                                                        │
                                                                        ▼
                                                          /inspection/detections_geo
                                                                        │
                                                                        ▼
                                                          defect_aggregator_node（跨帧去重+置信度融合）
                                                                        │
                                                          ┌─────────────┴─────────────┐
                                                          ▼                           ▼
                                                  /inspection/defect_reports    写入SQLite/rosbag
                                                          │
                                              ┌───────────┴───────────┐
                                              ▼                       ▼
                                        rviz2 Marker可视化      可选：rosbridge → Web看板
```

### 3.2 功能包划分

```
inspection_ws/src/
├── inspection_msgs/            # 自定义消息
├── inspection_perception/      # yolo_detector_node, geo_localization_node
├── inspection_postproc/        # defect_aggregator_node
├── inspection_viz/             # rviz2 marker publisher / web dashboard bridge
└── inspection_bringup/         # launch文件、参数yaml
```

### 3.3 自定义消息设计

```
# inspection_msgs/msg/DefectDetection.msg
std_msgs/Header header
string defect_type              # "insulator_missing_disc" / "insulator_normal" ...
float32 confidence
sensor_msgs/RegionOfInterest bbox
geometry_msgs/Point world_position   # 若无位姿信息则全0，由geo_localization_node填充
int32 track_id                   # 跨帧关联用，未关联时为-1

# inspection_msgs/msg/DefectDetectionArray.msg
std_msgs/Header header
DefectDetection[] detections

# inspection_msgs/msg/DefectReport.msg  （聚合后，一条=一个物理缺陷）
string report_id
string defect_type
float32 confidence
geometry_msgs/Point location
uint32 observation_count         # 被观测到的帧数，用于置信度融合
string tower_id                  # 可选，关联杆塔编号
```

也可以直接复用 `vision_msgs/Detection2DArray` 减少自定义工作量，缺点是塞不下 `world_position`、`tower_id` 这些业务字段，课设建议还是自定义更省事。

### 3.4 关键节点实现要点

**yolo_detector_node**（感知核心）
- 订阅 `sensor_msgs/Image`（建议用 `image_transport` 的压缩传输，节省带宽）
- 模型加载：ultralytics导出的 `.engine`（TensorRT）优先，Jetson等嵌入式平台上比原生pt快数倍
- 推理结果转成 `DefectDetectionArray` 发布，`header.stamp` 必须沿用输入图像的时间戳（后面时间同步要用）

```python
# 伪代码示意
class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector_node')
        self.model = YOLO('yolov10_ca_p2.engine')
        self.sub = self.create_subscription(Image, '/drone/camera/image_raw', self.cb, 10)
        self.pub = self.create_publisher(DefectDetectionArray, '/inspection/detections', 10)

    def cb(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        results = self.model.predict(frame, verbose=False)[0]
        out = DefectDetectionArray(header=msg.header)
        for box in results.boxes:
            det = DefectDetection()
            det.header = msg.header
            det.defect_type = self.model.names[int(box.cls)]
            det.confidence = float(box.conf)
            det.bbox = self.to_roi(box.xywh[0])
            out.detections.append(det)
        self.pub.publish(out)
```

**geo_localization_node**（可选，若有MAVROS/GPS）
- 用 `message_filters.ApproximateTimeSynchronizer` 同步 `/inspection/detections` 和 `/mavros/local_position/pose`
- 结合相机内参 + 云台角度（若有）+ 无人机位姿，把像素框反投影成世界坐标（简化版可以只用无人机水平位置近似，精确版需要地面高程/杆塔已知高度做交会）
- 用 `tf2` 管理 `camera_optical_frame → base_link → map` 的静态/动态变换

**defect_aggregator_node**（后处理）
- 维护一个滑动窗口内的检测记录，按 `world_position` 距离阈值 + `defect_type` 做关联（简单版：欧氏距离<1m视为同一缺陷）
- 多帧观测做置信度融合（如取最大值或加权平均），达到一定 `observation_count` 才正式生成 `DefectReport`，减少单帧误检
- 落库：课设量级用SQLite足够，字段对齐 `DefectReport`

### 3.5 可视化方案

优先做 **rviz2**：`yolo_detector_node`/`defect_aggregator_node` 额外发布 `visualization_msgs/MarkerArray`，在rviz2里实时看到检测框和缺陷点标记，这是最"原生"、工作量最小、演示效果也直观的方式，课设优先做这个。

如果还想要一个网页看板（给不装ROS2环境的老师看统计数据），加一层 `rosbridge_suite`：`ros2 launch rosbridge_server rosbridge_websocket_launch.xml` 起一个websocket桥，前端用 `roslibjs` 订阅 `/inspection/defect_reports` 直接渲染，不需要额外写REST API后端。这是ROS2生态下比自己搭FastAPI更省事的路线。

### 3.6 部署与性能

- 模型导出：`yolo export model=best.pt format=engine device=0 half=True` 生成TensorRT引擎，FP16基本必开（嵌入式平台显存/算力有限）
- 如果是Jetson系列做机载计算，注意JetPack版本对应的TensorRT/CUDA版本兼容性，ultralytics导出环境最好和部署环境JetPack版本一致，避免engine不兼容
- 图像话题较大时用 `image_transport` 的 `compressed` 传输类型，减少节点间/机载-地面站间带宽压力

---

## 4. 实验与验证计划

| 实验 | 对比项 | 关键指标 |
|---|---|---|
| baseline | 原YOLOv10（3头，无CA） | mAP@0.5, mAP@0.5:0.95, AP_small |
| +P2头 | 4头 vs 3头 | AP_small 提升幅度、推理耗时增量 |
| +CA | CA插入backbone / neck / 两者都插 | mAP、AP_small、参数量增量 |
| +TAL调优 | topk=6/10/13 | AP_small、正样本分配质量（可视化assign结果） |
| 全量融合 | P2+CA+TAL调优 vs baseline | 综合提升 + FPS（嵌入式平台上实测） |

尺度分布分析（2.1节）的输出作为附录放进报告，用来解释"为什么加P2头/为什么topk这么设"，比单纯调参更有说服力。

---

## 5. 时间规划（按2.5周估算，供参考）

| 阶段 | 内容 | 预计时长 |
|---|---|---|
| 1 | GT尺度分布分析 + 数据/切片增强脚本 | 1-2天 |
| 2 | P2检测头yaml改造 + 跑通baseline对比 | 2天 |
| 3 | CA模块实现、接入、消融实验 | 2-3天 |
| 4 | TAL超参小范围调优 | 1天 |
| 5 | ROS2消息包+yolo_detector_node跑通 | 2天 |
| 6 | geo_localization_node（若做）+ defect_aggregator_node | 2-3天 |
| 7 | rviz2可视化 + 可选web看板 | 1-2天 |
| 8 | 联调、TensorRT部署测试、报告撰写 | 2天 |

时间紧张的话，第6阶段的geo定位可以砍掉或简化成"只输出像素坐标+仅用于跨帧近似去重"，不做真实世界坐标反投影，工作量能省不少。

---

## 6. 风险与备选方案

- **P2头训练不稳定/收敛慢**：先只跑几轮小实验看loss曲线，不行就退回3头+仅CA的方案，一样能在报告里讲清楚小目标优化的思路
- **CA提升不明显**：准备CBAM或SE模块做平行对比，哪个有效写哪个，都是加分的消融实验素材
- **没有MAVROS/GPS**：geo_localization_node整体跳过，defect_aggregator_node改成基于图像特征相似度（而不是世界坐标）做跨帧关联，同样能讲通去重逻辑
- **Jetson等嵌入式设备算力不够跑P2头+CA**：优先保证NMS-free的YOLOv10原生实时性优势，小目标优化模块做成可开关的配置项，报告里给出精度-速度权衡表