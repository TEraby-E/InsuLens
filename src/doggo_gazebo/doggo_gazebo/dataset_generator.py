"""Generate a YOLO dataset from the Gazebo RGB camera.

The target pose is randomized through Gazebo's SetEntityState service.  Its
known 3-D extent is projected with CameraInfo, so every saved simulation frame
receives a deterministic YOLO bounding-box label without hand annotation.
"""

from pathlib import Path
import math
import random
from typing import Optional, Tuple

import cv2
from cv_bridge import CvBridge
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


class DatasetGenerator(Node):
    """Move one simulated insulator and save images with projected labels."""

    def __init__(self) -> None:
        super().__init__("insulator_dataset_generator")
        self.declare_parameter("output_dir", "/root/doggo/datasets/insulator_sim")
        self.declare_parameter("num_samples", 1200)
        self.declare_parameter("seed", 42)
        self.declare_parameter("entity_name", "insulator_training")
        self.declare_parameter("image_topic", "/doggo/camera/image_raw")
        self.declare_parameter("camera_info_topic", "/doggo/camera/camera_info")
        self.declare_parameter("settle_frames", 3)
        self.declare_parameter("negative_fraction", 0.08)
        self.declare_parameter("augment", True)

        self.output_dir = Path(
            self.get_parameter("output_dir").get_parameter_value().string_value
        ).expanduser()
        self.num_samples = self.get_parameter("num_samples").value
        self.entity_name = self.get_parameter("entity_name").value
        self.settle_frames_param = self.get_parameter("settle_frames").value
        self.negative_fraction = self.get_parameter("negative_fraction").value
        self.use_augmentation = self.get_parameter("augment").value
        self.rng = random.Random(self.get_parameter("seed").value)
        self.np_rng = np.random.default_rng(self.get_parameter("seed").value)

        for split in ("train", "val"):
            (self.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        self._write_data_yaml()

        self.bridge = CvBridge()
        self.camera_info: Optional[CameraInfo] = None
        self.target_pose = None
        self.is_negative = False
        self.index = 0
        self.settling = -1
        self.request_in_flight = False
        self.finished = False

        image_topic = self.get_parameter("image_topic").value
        info_topic = self.get_parameter("camera_info_topic").value
        self.create_subscription(
            Image, image_topic, self._on_image, 10
        )
        self.create_subscription(
            CameraInfo, info_topic, self._on_camera_info, 10
        )
        self.set_state = self.create_client(
            SetEntityState, "/gazebo/set_entity_state"
        )
        self.create_timer(0.25, self._start_if_ready)
        self.get_logger().info(
            f"Waiting for Gazebo camera; output={self.output_dir}, "
            f"samples={self.num_samples}"
        )

    def _write_data_yaml(self) -> None:
        content = (
            f"path: {self.output_dir.resolve()}\n"
            "train: images/train\n"
            "val: images/val\n"
            "names:\n"
            "  0: insulator\n"
        )
        (self.output_dir / "data.yaml").write_text(content, encoding="utf-8")

    def _on_camera_info(self, message: CameraInfo) -> None:
        if self.camera_info is None:
            self.get_logger().info(
                f"Camera ready: {message.width}x{message.height}, fx={message.k[0]:.2f}"
            )
        self.camera_info = message

    def _start_if_ready(self) -> None:
        if self.finished or self.request_in_flight or self.settling >= 0:
            return
        if self.camera_info is None:
            return
        if not self.set_state.service_is_ready():
            self.set_state.wait_for_service(timeout_sec=0.05)
            return
        self._randomize_target()

    def _randomize_target(self) -> None:
        self.is_negative = self.rng.random() < self.negative_fraction
        if self.is_negative:
            x, y, z = -4.0, 0.0, 2.0
        else:
            x = self.rng.uniform(3.0, 10.0)
            half_width = x * math.tan(1.0472 / 2.0) * 0.62
            y = self.rng.uniform(-half_width, half_width)
            z = 2.0 + self.rng.uniform(-0.75, 0.75)

        roll = self.rng.uniform(-0.32, 0.32)
        pitch = self.rng.uniform(-0.32, 0.32)
        yaw = self.rng.uniform(-math.pi, math.pi)
        qx, qy, qz, qw = quaternion_from_euler(roll, pitch, yaw)

        state = EntityState()
        state.name = self.entity_name
        state.reference_frame = "world"
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = z
        state.pose.orientation.x = qx
        state.pose.orientation.y = qy
        state.pose.orientation.z = qz
        state.pose.orientation.w = qw
        self.target_pose = state.pose

        request = SetEntityState.Request()
        request.state = state
        self.request_in_flight = True
        if self.index == 0:
            self.get_logger().info("Gazebo state service ready; starting capture")
        future = self.set_state.call_async(request)
        future.add_done_callback(self._on_pose_set)

    def _on_pose_set(self, future) -> None:
        self.request_in_flight = False
        try:
            result = future.result()
        except Exception as exc:  # pragma: no cover - ROS transport failure
            self.get_logger().error(f"SetEntityState failed: {exc}")
            self.settling = -1
            return
        if not result.success:
            self.get_logger().error("Gazebo rejected the target pose")
            self.settling = -1
            return
        self.settling = self.settle_frames_param

    def _on_image(self, message: Image) -> None:
        if self.finished or self.settling < 0:
            return
        if self.settling > 0:
            self.settling -= 1
            return

        frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        frame = self._augment(frame) if self.use_augmentation else frame
        bbox = None if self.is_negative else self._project_bbox(frame.shape)
        if not self.is_negative and bbox is None:
            self.settling = -1
            self._randomize_target()
            return

        split = "val" if self.index % 5 == 0 else "train"
        stem = f"sim_{self.index:06d}"
        image_path = self.output_dir / "images" / split / f"{stem}.jpg"
        label_path = self.output_dir / "labels" / split / f"{stem}.txt"
        cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 94])
        label = "" if bbox is None else "0 {:.6f} {:.6f} {:.6f} {:.6f}\n".format(*bbox)
        label_path.write_text(label, encoding="utf-8")

        self.index += 1
        if self.index % 50 == 0 or self.index == self.num_samples:
            self.get_logger().info(f"Generated {self.index}/{self.num_samples}")
        if self.index >= self.num_samples:
            self.finished = True
            self.get_logger().info(
                f"Dataset complete: {self.output_dir / 'data.yaml'}"
            )
            return
        self.settling = -1
        self._randomize_target()

    def _project_bbox(
        self, image_shape: Tuple[int, ...]
    ) -> Optional[Tuple[float, float, float, float]]:
        """Project the known 0.8 x 0.8 x 2.4 m target box into the image."""
        if self.camera_info is None or self.target_pose is None:
            return None
        height, width = image_shape[:2]
        fx, fy = self.camera_info.k[0], self.camera_info.k[4]
        cx, cy = self.camera_info.k[2], self.camera_info.k[5]
        pose = self.target_pose
        rotation = rotation_from_quaternion(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        center = np.array(
            [pose.position.x, pose.position.y, pose.position.z], dtype=np.float64
        )
        projected = []
        for lx in (-0.42, 0.42):
            for ly in (-0.42, 0.42):
                for lz in (-1.24, 1.24):
                    world = rotation @ np.array([lx, ly, lz]) + center
                    relative = world - np.array([0.0, 0.0, 2.0])
                    # Gazebo camera: +X forward, +Y left, +Z up.
                    optical_x = -relative[1]
                    optical_y = -relative[2]
                    optical_z = relative[0]
                    if optical_z <= 0.1:
                        continue
                    projected.append(
                        (
                            fx * optical_x / optical_z + cx,
                            fy * optical_y / optical_z + cy,
                        )
                    )
        if len(projected) < 4:
            return None
        points = np.asarray(projected)
        xmin = float(np.clip(points[:, 0].min(), 0, width - 1))
        xmax = float(np.clip(points[:, 0].max(), 0, width - 1))
        ymin = float(np.clip(points[:, 1].min(), 0, height - 1))
        ymax = float(np.clip(points[:, 1].max(), 0, height - 1))
        if xmax - xmin < 12 or ymax - ymin < 12:
            return None
        return (
            ((xmin + xmax) / 2.0) / width,
            ((ymin + ymax) / 2.0) / height,
            (xmax - xmin) / width,
            (ymax - ymin) / height,
        )

    def _augment(self, frame: np.ndarray) -> np.ndarray:
        """Apply inexpensive sensor/domain randomization while preserving labels."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + self.rng.uniform(-5.0, 5.0)) % 180
        hsv[..., 1] *= self.rng.uniform(0.75, 1.25)
        hsv[..., 2] *= self.rng.uniform(0.65, 1.30)
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        output = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        if self.rng.random() < 0.18:
            output = cv2.GaussianBlur(output, (3, 3), 0)
        if self.rng.random() < 0.30:
            noise = self.np_rng.normal(0, self.rng.uniform(1.0, 5.0), output.shape)
            output = np.clip(output.astype(np.float32) + noise, 0, 255).astype(
                np.uint8
            )
        return output

def quaternion_from_euler(roll: float, pitch: float, yaw: float):
    """Return an xyzw quaternion for the supplied Euler angles."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def rotation_from_quaternion(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Convert an xyzw quaternion to a 3-by-3 rotation matrix."""
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DatasetGenerator()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
