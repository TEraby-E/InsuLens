"""Publish a real image, image directory, or video as a ROS 2 image stream."""

from pathlib import Path
from typing import List, Optional

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class ImageSource(Node):
    """Turn common real-world media into a repeatable ROS camera topic."""

    def __init__(self) -> None:
        super().__init__("real_image_source")
        self.declare_parameter(
            "source", "/root/doggo/datasets/cplid_yolo/images/val"
        )
        self.declare_parameter("topic", "/doggo/real_camera/image_raw")
        self.declare_parameter("frame_id", "real_camera_optical_frame")
        self.declare_parameter("fps", 5.0)
        self.declare_parameter("loop", True)

        self.source = Path(self.get_parameter("source").value).expanduser()
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.loop = bool(self.get_parameter("loop").value)
        fps = max(0.1, float(self.get_parameter("fps").value))
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(
            Image, self.get_parameter("topic").value, 10
        )
        self.images: List[Path] = []
        self.index = 0
        self.video: Optional[cv2.VideoCapture] = None

        if self.source.is_dir():
            self.images = sorted(
                path
                for path in self.source.iterdir()
                if path.suffix.lower() in IMAGE_SUFFIXES
            )
        elif self.source.is_file() and self.source.suffix.lower() in IMAGE_SUFFIXES:
            self.images = [self.source]
        elif self.source.is_file():
            self.video = cv2.VideoCapture(str(self.source))
            if not self.video.isOpened():
                raise RuntimeError(f"Unable to open video: {self.source}")
        else:
            raise FileNotFoundError(f"Image/video source not found: {self.source}")
        if not self.images and self.video is None:
            raise RuntimeError(f"No supported images found in: {self.source}")

        self.create_timer(1.0 / fps, self._publish_frame)
        self.get_logger().info(f"Publishing real media from {self.source}")

    def _next_frame(self):
        if self.images:
            if self.index >= len(self.images):
                if not self.loop:
                    return None
                self.index = 0
            frame = cv2.imread(str(self.images[self.index]))
            self.index += 1
            return frame

        ok, frame = self.video.read()
        if ok:
            return frame
        if not self.loop:
            return None
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = self.video.read()
        return frame if ok else None

    def _publish_frame(self) -> None:
        frame = self._next_frame()
        if frame is None:
            return
        message = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.frame_id
        self.publisher.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = ImageSource()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except KeyboardInterrupt:
                pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

