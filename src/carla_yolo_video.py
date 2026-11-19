
import carla
import cv2
import numpy as np
import queue
import os
from ultralytics import YOLO

# Configuration
MODEL_PATH = "runs/detect/runs/smoke/weights/best.pt"
OUTPUT_PATH = "results/carla_yolo/carla_yolo_live.mp4"

WIDTH = 640
HEIGHT = 360
MAX_FRAMES = 60
CONFIDENCE = 0.15
VIDEO_FPS = 10

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Load trained YOLO model
print("Loading YOLO model...")
model = YOLO(MODEL_PATH)

# Connect to CARLA
client = carla.Client("127.0.0.1", 2000)
client.set_timeout(30.0)
world = client.get_world()
blueprints = world.get_blueprint_library()

vehicle = None
camera = None
writer = None
image_queue = queue.Queue()

try:
    # Find an available vehicle spawn point
    vehicle_bp = blueprints.filter("vehicle.*")[0]
    spawn_points = world.get_map().get_spawn_points()

    for point in spawn_points:
        vehicle = world.try_spawn_actor(vehicle_bp, point)
        if vehicle is not None:
            break

    if vehicle is None:
        raise RuntimeError("No available vehicle spawn point")

    vehicle.set_autopilot(True)

    # Configure RGB camera
    camera_bp = blueprints.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(WIDTH))
    camera_bp.set_attribute("image_size_y", str(HEIGHT))
    camera_bp.set_attribute("fov", "90")
    camera_bp.set_attribute("enable_postprocess_effects", "false")

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

    # Prepare output video
    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        VIDEO_FPS,
        (WIDTH, HEIGHT)
    )

    if not writer.isOpened():
        raise RuntimeError("Could not create output video")

    print("CARLA camera running...")
    print("YOLO detection started...")

    total_detections = 0

    for frame_number in range(MAX_FRAMES):
        frame = image_queue.get(timeout=30)

        image = np.frombuffer(
            frame.raw_data, dtype=np.uint8
        ).reshape((HEIGHT, WIDTH, 4))

        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        # Run trained YOLO model
        result = model.predict(
            image,
            imgsz=640,
            conf=CONFIDENCE,
            verbose=False
        )[0]

        total_detections += len(result.boxes)

        # Draw bounding boxes
        annotated = result.plot()
        writer.write(annotated)

        if (frame_number + 1) % 10 == 0:
            print(
                f"Processed {frame_number + 1}/{MAX_FRAMES} "
                f"frames | Detections: {total_detections}"
            )

    print("\nCARLA + YOLO processing completed!")
    print("Total frames:", MAX_FRAMES)
    print("Total detections:", total_detections)
    print("Average detections/frame:",
          round(total_detections / MAX_FRAMES, 2))
    print("Video saved:", OUTPUT_PATH)

finally:
    if camera is not None:
        camera.stop()
        camera.destroy()

    if vehicle is not None:
        vehicle.destroy()

    if writer is not None:
        writer.release()

    print("Cleanup completed.")
