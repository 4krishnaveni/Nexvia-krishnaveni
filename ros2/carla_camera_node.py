"""CARLA -> ROS2. Spawns ego car + RGB camera + a stopped obstacle ahead, publishes /camera/image_raw,
and brakes when /perception/brake is True (closed loop for the demo video).
Start CARLA first (./CarlaUE4.sh -quality-level=Low), then: python3 ros2/carla_camera_node.py"""
import carla, numpy as np, rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge

W, H = 1242, 375

class CarlaCam(Node):
    def __init__(self):
        super().__init__("carla_camera")
        self.pub = self.create_publisher(Image, "/camera/image_raw", 10)
        self.create_subscription(Bool, "/perception/brake", lambda m: setattr(self, "brake", m.data), 10)
        self.bridge, self.brake = CvBridge(), False
        client = carla.Client("localhost", 2000); client.set_timeout(15)
        self.world = client.get_world(); lib = self.world.get_blueprint_library()
        sp = self.world.get_map().get_spawn_points()[0]
        self.ego = self.world.spawn_actor(lib.filter("vehicle.tesla.model3")[0], sp)
        wp = self.world.get_map().get_waypoint(sp.location).next(45.0)[0]
        self.obst = self.world.spawn_actor(lib.filter("vehicle.audi.a2")[0], wp.transform)
        self.obst.apply_control(carla.VehicleControl(hand_brake=True))
        cam = lib.find("sensor.camera.rgb")
        for k, v in dict(image_size_x=W, image_size_y=H, fov=82, sensor_tick=0.05).items(): cam.set_attribute(k, str(v))
        self.cam = self.world.spawn_actor(cam, carla.Transform(carla.Location(x=1.5, z=1.6)), attach_to=self.ego)
        self.cam.listen(self.on_image)
        self.create_timer(0.05, self.drive)

    def on_image(self, img):
        a = np.frombuffer(img.raw_data, np.uint8).reshape(img.height, img.width, 4)[:, :, :3]
        self.pub.publish(self.bridge.cv2_to_imgmsg(np.ascontiguousarray(a), "bgr8"))

    def drive(self):
        self.ego.apply_control(carla.VehicleControl(throttle=0.0 if self.brake else 0.45, brake=1.0 if self.brake else 0.0))

    def cleanup(self):
        for a in (self.cam, self.obst, self.ego): a.destroy()

def main():
    rclpy.init(); n = CarlaCam()
    try: rclpy.spin(n)
    finally: n.cleanup(); rclpy.shutdown()
if __name__ == "__main__": main()
