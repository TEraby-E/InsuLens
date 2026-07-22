#!/usr/bin/env bash
set -euo pipefail

TOPICS=(
  /doggo/camera/image_raw
  /doggo/detection_image
  /doggo/detections
  /doggo/defect_alerts
  /doggo/inference_ms
)

if ! ros2 node list >/dev/null 2>&1; then
  echo "无法连接 ROS 2 图：请先 source ROS 2 与工作空间环境，并启动巡检系统。" >&2
  exit 1
fi

echo "=== Doggo ROS 2 运行诊断 ==="
echo "活动节点："
ros2 node list
echo
echo "关键话题："
for topic in "${TOPICS[@]}"; do
  if ros2 topic info "${topic}" >/dev/null 2>&1; then
    echo "[正常] ${topic}"
    ros2 topic info "${topic}" | sed 's/^/  /'
  else
    echo "[缺失] ${topic}"
  fi
done

echo
echo "提示：可另开终端运行 'ros2 topic hz /doggo/detection_image' 检查实际发布频率。"
