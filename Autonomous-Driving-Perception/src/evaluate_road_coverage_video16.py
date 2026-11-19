import cv2
import glob
import csv
from pathlib import Path
from ultralytics import YOLO
from lane_segmentation import RoadSegmenter

image_dir = "data/Dataset/Video16/images"
output_file = "results/road_coverage_video16.csv"

files = sorted(glob.glob(image_dir + "/*.png"))

if not files:
    raise FileNotFoundError("No Video16 images found: " + image_dir)

Path("results").mkdir(exist_ok=True)

print("Video16 frames found:", len(files))
print("Loading SegFormer and YOLO...")

segmenter = RoadSegmenter()
model = YOLO("runs/detect/runs/smoke/weights/best.pt")

frame_count = 0
detection_count = 0
on_road_count = 0
off_road_count = 0
coverage_values = []

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "frame", "image", "class",
        "road_coverage_percent", "likely_on_road"
    ])

    for frame_no, path in enumerate(files):
        img = cv2.imread(path)

        if img is None:
            continue

        mask = segmenter(img)
        prediction = model.predict(
            img, conf=0.35, verbose=False
        )[0]

        frame_count += 1

        if prediction.boxes is not None:
            for box, cls in zip(
                prediction.boxes.xyxy.cpu().numpy(),
                prediction.boxes.cls.int().tolist()
            ):
                x1, y1, x2, y2 = map(int, box)

                xa = max(0, x1 + int((x2-x1)*0.2))
                xb = min(img.shape[1], x2 - int((x2-x1)*0.2))
                ya = max(0, y1 + int((y2-y1)*0.7))
                yb = min(img.shape[0], y2)

                if xb <= xa or yb <= ya:
                    coverage = 0.0
                else:
                    region = mask[ya:yb, xa:xb]
                    coverage = float((region > 0).mean() * 100)

                likely_on_road = coverage >= 50

                detection_count += 1
                coverage_values.append(coverage)

                if likely_on_road:
                    on_road_count += 1
                else:
                    off_road_count += 1

                writer.writerow([
                    frame_no,
                    Path(path).name,
                    prediction.names[cls],
                    round(coverage, 1),
                    likely_on_road
                ])

        if (frame_no + 1) % 25 == 0:
            f.flush()
            print(
                "Processed:", frame_no + 1, "/", len(files),
                "| Detections:", detection_count
            )

    f.flush()

print("\nVIDEO16 EVALUATION COMPLETE")
print("Frames processed:", frame_count)
print("Total detections:", detection_count)
print("Likely on-road:", on_road_count)
print("Likely off-road:", off_road_count)

if coverage_values:
    import numpy as np
    print("Average road coverage:", round(np.mean(coverage_values), 2), "%")
    print("Median road coverage:", round(np.median(coverage_values), 2), "%")

print("CSV saved:", output_file)
