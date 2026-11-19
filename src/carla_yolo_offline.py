
import cv2
import os
import time
from ultralytics import YOLO

IMAGE_PATH = "results/carla_camera/carla_camera_frame.png"
MODEL_PATH = "runs/detect/runs/smoke/weights/best.pt"
OUTPUT_PATH = "results/carla_yolo/carla_yolo_offline_demo.mp4"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

print("Loading YOLO model...")
model = YOLO(MODEL_PATH)

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(f"Cannot read image: {IMAGE_PATH}")

print("Running YOLO detection...")
start = time.time()

result = model.predict(
    image,
    imgsz=640,
    conf=0.15,
    verbose=False
)[0]

annotated = result.plot()

height, width = annotated.shape[:2]
fps = 10
duration_seconds = 5

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not writer.isOpened():
    raise RuntimeError("Could not create output video")

for _ in range(fps * duration_seconds):
    writer.write(annotated)

writer.release()

elapsed = time.time() - start

print("\nOffline YOLO demonstration completed!")
print("Image dimensions:", width, "x", height)
print("Detections:", len(result.boxes))
print("Processing time:", round(elapsed, 2), "seconds")
print("Video saved:", OUTPUT_PATH)
