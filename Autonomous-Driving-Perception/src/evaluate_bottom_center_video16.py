
import cv2
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from lane_segmentation import RoadSegmenter

weights = "runs/detect/runs/smoke/weights/best.pt"
image_dir = Path("data/Dataset/Video16/images")

model = YOLO(weights)
segmenter = RoadSegmenter()

records = []
images = sorted(image_dir.glob("*.png"))

for index, path in enumerate(images):
    image = cv2.imread(str(path))

    if image is None:
        continue

    mask = segmenter(image)
    result = model.predict(image, conf=0.35, verbose=False)[0]

    if result.boxes is None:
        continue

    height, width = mask.shape

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        cx = max(0, min(width - 1, (x1 + x2) // 2))
        cy = max(0, min(height - 1, y2 - 1))
        # Measure road coverage in the bottom 20% of the detection box
        box_top = max(0, y1)
        box_bottom = min(height, y2)
        box_left = max(0, x1)
        box_right = min(width, x2)
        box_height = box_bottom - box_top
        box_width = box_right - box_left
        if box_height > 0 and box_width > 0:
            lower_start = max(box_top, box_bottom -
                              max(1, int(box_height * 0.20)))
            lower_region = mask[lower_start:box_bottom, box_left:box_right]
            lower_road_coverage = float((lower_region > 0).mean() * 100)
        else:
            lower_road_coverage = 0.0

        touches_bottom = y2 >= height - 2
        touches_left = x1 <= 1
        touches_right = x2 >= width - 2

        records.append({
            "frame": index,
            "image": path.name,
            "class": model.names[int(box.cls[0])],
            "confidence": float(box.conf[0]),
            "bottom_center_on_road": bool(mask[cy, cx] > 0),
            "touches_bottom": touches_bottom,
            "touches_left": touches_left,
            "touches_right": touches_right,
            "lower_box_road_coverage": lower_road_coverage,
        })

    if (index + 1) % 50 == 0:
        print(f"Processed {index + 1}/{len(images)} frames")

df = pd.DataFrame(records)

Path("results").mkdir(exist_ok=True)
df.to_csv("results/bottom_center_video16.csv", index=False)

print("\nEvaluation complete")
print("Frames:", len(images))
print("Detections:", len(df))

if not df.empty:
    print("\nBottom-center results:")
    print(df["bottom_center_on_road"].value_counts())

    print("\nBoundary contact:")
    print("Bottom:", df["touches_bottom"].sum())
    print("Left:", df["touches_left"].sum())
    print("Right:", df["touches_right"].sum())

    print("\nResults by class:")
    print(df.groupby("class")["bottom_center_on_road"].value_counts())

print("\nSaved: results/bottom_center_video16.csv")
