"""Summarize Doggo inspection traffic for terminal-based runtime observation."""

import json

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


class InspectionMonitor(Node):
    """Report detector throughput, latency, alerts, and malformed messages."""

    def __init__(self) -> None:
        super().__init__("inspection_monitor")
        self.declare_parameter("detections_topic", "/doggo/detections")
        self.declare_parameter("alerts_topic", "/doggo/defect_alerts")
        self.declare_parameter("annotated_topic", "/doggo/detection_image")
        self.declare_parameter("status_period_sec", 5.0)

        self.detection_messages = 0
        self.valid_detection_messages = 0
        self.annotated_images = 0
        self.detected_objects = 0
        self.alert_messages = 0
        self.invalid_messages = 0
        self.total_inference_ms = 0.0

        self.create_subscription(
            String,
            self.get_parameter("detections_topic").value,
            self._on_detections,
            10,
        )
        self.create_subscription(
            String,
            self.get_parameter("alerts_topic").value,
            self._on_alert,
            10,
        )
        self.create_subscription(
            Image,
            self.get_parameter("annotated_topic").value,
            self._on_annotated_image,
            10,
        )
        status_period = max(
            0.5, float(self.get_parameter("status_period_sec").value)
        )
        self.create_timer(status_period, self._report_status)
        self.get_logger().info(
            "巡检监视器已启动：将汇总检测消息、带框图像、缺陷告警和推理耗时。"
        )

    def _on_detections(self, message: String) -> None:
        self.detection_messages += 1
        try:
            payload = json.loads(message.data)
            detections = payload["detections"]
            inference_ms = float(payload["inference_ms"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self.invalid_messages += 1
            self.get_logger().warning(f"无法解析检测消息：{exc}")
            return
        if not isinstance(detections, list):
            self.invalid_messages += 1
            self.get_logger().warning("检测消息中的 detections 不是列表。")
            return
        self.detected_objects += len(detections)
        self.total_inference_ms += inference_ms
        self.valid_detection_messages += 1

    def _on_alert(self, message: String) -> None:
        self.alert_messages += 1
        try:
            payload = json.loads(message.data)
            defect_count = len(payload.get("detections", []))
        except (json.JSONDecodeError, TypeError):
            defect_count = 0
        self.get_logger().warning(f"发现缺陷事件：本帧包含 {defect_count} 个缺陷目标。")

    def _on_annotated_image(self, _message: Image) -> None:
        self.annotated_images += 1

    def _report_status(self) -> None:
        average_latency = (
            self.total_inference_ms / self.valid_detection_messages
            if self.valid_detection_messages
            else 0.0
        )
        self.get_logger().info(
            "巡检状态 | 检测消息=%d | 带框图像=%d | 目标总数=%d | "
            "缺陷事件=%d | 平均推理=%.1f ms | 异常消息=%d"
            % (
                self.detection_messages,
                self.annotated_images,
                self.detected_objects,
                self.alert_messages,
                average_latency,
                self.invalid_messages,
            )
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = InspectionMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
