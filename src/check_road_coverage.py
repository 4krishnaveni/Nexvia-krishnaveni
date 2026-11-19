import cv2
import glob
import csv
from ultralytics import YOLO
from lane_segmentation import RoadSegmenter

files = sorted(glob.glob("data/demo_sample/*.png"))

segmenter = RoadSegmenter()
model = YOLO("runs/detect/runs/smoke/weights/best.pt")

results = []

for frame_no, path in enumerate(files):
    img = cv2.imread(path)
    if img is None:
        continue

    mask = segmenter(img)
    prediction = model.predict(img, conf=0.35, verbose=False)[0]

    for box, cls in zip(
        prediction.boxes.xyxy.cpu().numpy(),
        prediction.boxes.cls.int().tolist()
    ):
        x1, y1, x2, y2 = map(int, box)

        xa = max(0, x1 + int((x2-x1)*0.2))
        xb = min(img.shape[1], x2 - int((x2-x1)*0.2))
        ya = max(0, y1 + int((y2-y1)*0.7))
        yb = min(img.shape[0], y2)

        coverage = 0.0
        if xb > xa and yb > ya:
            coverage = (mask[ya:yb, xa:xb] > 0).mean() * 100

        results.append([
            frame_no,
            prediction.names[cls],
            round(coverage, 1),
            coverage >= 50
        ])

with open("results/road_coverage_test.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "class", "road_coverage_percent", "likely_on_road"])
    writer.writerows(results)

print("Frames processed:", len(files))
print("Total detections:", len(results))
print("Likely on-road:", sum(r[3] for r in results))
print("Likely off-road:", sum(not r[3] for r in results))
print("CSV saved: results/road_coverage_test.csv")
