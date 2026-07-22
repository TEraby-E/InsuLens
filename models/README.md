# 模型卡

模型二进制文件不会被普通 Git 规则跟踪。如需发布，请使用仓库的 Git LFS 配置，
并在本文件中保留校验和与数据来源信息。

## `insulator_yolov10s.pt`

面向仿真域的专用权重，用于 Gazebo 演示，并作为真实场景迁移学习的初始权重。

| 字段 | 内容 |
|---|---|
| 网络结构 | YOLOv10s |
| 类别 | `insulator`（绝缘子） |
| 训练数据 | 500 张自动标注的 Gazebo 图像 |
| 训练设置 | 30 个 epoch、640 像素输入，使用 `yolov10s.yaml` 训练 |
| 留出仿真集指标 | mAP50 0.976，mAP50-95 0.647 |
| 文件大小 | 16,508,475 字节 |
| SHA-256 | `80f6581bc0d7523e009ada94a308ed214c4cb119faed8163a078fb44c60bc084` |

## `insulator_defect_yolov10s.pt`

基于 `insulator_yolov10s.pt` 初始化的真实场景权重；检测头扩展为两个类别后，
在 CPLID 上完成微调。

| 字段 | 内容 |
|---|---|
| 网络结构 | YOLOv10s |
| 类别 | `insulator`（绝缘子）、`missing_disc`（缺片） |
| 训练数据 | CPLID：678 张训练图 / 170 张验证图 |
| 验证实例 | 269 个 `insulator`、50 个 `missing_disc` |
| 训练设置 | 50 个 epoch、768 像素输入、批大小 8 |
| 全类别平均 | P 0.791，R 0.844，mAP50 0.882，mAP50-95 0.530 |
| `insulator` | P 0.668, R 0.729, mAP50 0.785, mAP50-95 0.484 |
| `missing_disc` | P 0.914, R 0.960, mAP50 0.980, mAP50-95 0.575 |
| 文件大小 | 16,530,171 字节 |
| SHA-256 | `a61f1bea6896f721a23c0697258eab66d3be60aaca998e1477895fe6d8c273f6` |

指标来自 `scripts/prepare_cplid.py` 以固定随机种子生成的 CPLID 划分，
不属于独立现场基准。CPLID 的正常绝缘子图像来自真实无人机拍摄，其缺陷图像则是
将合成缺损绝缘子叠加到输电场景背景中得到的。上游数据集未声明明确许可证，
仅应在研究或课程作业中署名使用，不应重新分发其图像。

数据集引用：

> X. Tao et al., "Detection of Power Line Insulator Defects Using Aerial Images
> Analyzed With Convolutional Neural Networks," IEEE Transactions on Systems,
> Man, and Cybernetics: Systems, 2018.
