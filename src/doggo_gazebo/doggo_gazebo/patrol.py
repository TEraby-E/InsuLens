"""Simple repeatable patrol motion for the simulated camera carrier."""

from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node


class PatrolNode(Node):
    """Drive forward and backward along the transmission line."""

    def __init__(self) -> None:
        super().__init__("inspection_patrol")
        self.declare_parameter("speed", 0.65)
        self.declare_parameter("leg_duration", 58.0)
        self.speed = float(self.get_parameter("speed").value)
        self.leg_duration = float(self.get_parameter("leg_duration").value)
        self.publisher = self.create_publisher(Twist, "/doggo/cmd_vel", 10)
        self.start_ns = self.get_clock().now().nanoseconds
        self.create_timer(0.1, self._publish_command)
        self.get_logger().info("Automatic transmission-line patrol started")

    def _publish_command(self) -> None:
        elapsed = (self.get_clock().now().nanoseconds - self.start_ns) / 1e9
        direction = 1.0 if int(elapsed / self.leg_duration) % 2 == 0 else -1.0
        command = Twist()
        command.linear.x = direction * self.speed
        self.publisher.publish(command)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PatrolNode()
    try:
        rclpy.spin(node)
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
