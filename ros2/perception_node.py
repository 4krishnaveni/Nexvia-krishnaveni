"""ROS2 node: /camera/image_raw -> Perception -> /perception/image_annotated, /perception/brake, /perception/objects.
Run from repo root:
  PYTHONPATH=src python3 ros2/perception_node.py --ros-args -p weights:=runs/kitti_yolov8s/weights/best.pt"""
import rclpy, json
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge
from pipeline import Perception

class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception")
        self.declare_parameter("weights", "runs/kitti_yolov8s/weights/best.pt")
        self.declare_parameter("engine", "")
        eng = self.get_parameter("engine").value or None
        self.p = Perception(self.get_parameter("weights").value, engine=eng)
        self.bridge = CvBridge()
        self.create_subscription(Image, "/camera/image_raw", self.cb, 1)
        self.pub_img = self.create_publisher(Image, "/perception/image_annotated", 1)
        self.pub_brake = self.create_publisher(Bool, "/perception/brake", 1)
        self.pub_info = self.create_publisher(String, "/perception/objects", 1)

    def cb(self, msg):
        vis, info = self.p(self.bridge.imgmsg_to_cv2(msg, "bgr8"))
        self.pub_img.publish(self.bridge.cv2_to_imgmsg(vis, "bgr8"))
        self.pub_brake.publish(Bool(data=bool(info["brake"])))
        self.pub_info.publish(String(data=json.dumps(
            [{k: o[k] for k in ("id", "cls", "d_midas", "in_path")} for o in info["objects"]])))

def main():
    rclpy.init(); rclpy.spin(PerceptionNode()); rclpy.shutdown()
if __name__ == "__main__": main()
