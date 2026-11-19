
import cv2
from pathlib import Path
from ultralytics import YOLO
from lane_segmentation import RoadSegmenter

weights = "runs/detect/runs/smoke/weights/best.pt"
image_dir = Path("data/Dataset/Video16/images")

frames = [194, 236, 258, 277, 289]

model = YOLO(weights)
segmenter = RoadSegmenter()

for frame_no in frames:
    filename = f"{frame_no:010d}.png"
    image = cv2.imread(str(image_dir / filename))

    if image is None:
        print(f"Missing: {filename}")
        continue

    mask = segmenter(image)
    result = model.predict(image, conf=0.35, verbose=False)[0]

    print(f"\n--- Frame {frame_no} ---")

    if result.boxes is None:
        continue

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        cx = max(0, min(mask.shape[1] - 1, (x1+x2)//2))
        cy = max(0, min(mask.shape[0] - 1, y2-1))

        # Inspect a 21x21 pixel area around bottom center
        y0, y1p = max(0, cy-10), min(mask.shape[0], cy+11)
        x0, x1p = max(0, cx-10), min(mask.shape[1], cx+11)

        patch = mask[y0:y1p, x0:x1p]
        local_coverage = 100 * (patch > 0).mean()

        print(
            f"{model.names[int(box.cls[0])]} "
            f"confidence={float(box.conf[0]):.3f}, "
            f"box=({x1},{y1},{x2},{y2}), "
            f"bottom_center=({cx},{cy}), "
            f"local_road={local_coverage:.1f}%"
        )
