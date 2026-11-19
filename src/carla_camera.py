
import carla
import time
import os
import queue
import cv2
import numpy as np

OUTPUT_DIR = "results/carla_camera"
os.makedirs(OUTPUT_DIR, exist_ok=True)

client = carla.Client("127.0.0.1", 2000)
client.set_timeout(30.0)

world = client.get_world()
blueprints = world.get_blueprint_library()

camera_bp = blueprints.find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", "640")
camera_bp.set_attribute("image_size_y", "360")
camera_bp.set_attribute("fov", "90")
camera_bp.set_attribute("enable_postprocess_effects", "false")

vehicle = None
camera = None
image_queue = queue.Queue()

try:
    # Spawn a vehicle
    vehicle_bp = blueprints.filter("vehicle.*")[0]
    spawn_points = world.get_map().get_spawn_points()

    vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[0])

    if vehicle is None:
        raise RuntimeError("Could not spawn vehicle at first spawn point")

    vehicle.set_autopilot(True)

    # Attach camera in front of vehicle
    camera_transform = carla.Transform(
        carla.Location(x=-6.0, z=2.8),
        carla.Rotation(pitch=-10.0)
    )

    camera = world.spawn_actor(
        camera_bp,
        camera_transform,
        attach_to=vehicle
    )

    camera.listen(image_queue.put)

    print("CARLA virtual camera started!")
    print("Waiting for camera frame...")

    frame = image_queue.get(timeout=30)

    image = np.frombuffer(frame.raw_data, dtype=np.uint8)
    image = image.reshape((frame.height, frame.width, 4))
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    output_path = os.path.join(OUTPUT_DIR, "carla_camera_frame.png")
    cv2.imwrite(output_path, image)

    print("Image saved:", output_path)
    print("Image dimensions:", image.shape)

finally:
    if camera is not None:
        camera.stop()
        camera.destroy()

    if vehicle is not None:
        vehicle.destroy()

    print("CARLA camera cleanup complete.")
