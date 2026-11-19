import cv2
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from lane_segmentation import RoadSegmenter, overlay_mask

csv = pd.read_csv("results/road_coverage_video16.csv")
segmenter = RoadSegmenter()
model = YOLO("runs/detect/runs/smoke/weights/best.pt")

Path("results/road_debug").mkdir(parents=True, exist_ok=True)

for frame_no in [1, 96, 12, 286]:
    rows = csv[csv["frame"] == frame_no]
    if rows.empty:
        continue

    path = "data/Dataset/Video16/images/" + rows.iloc[0]["image"]
    img = cv2.imread(path)
    mask = segmenter(img)
    result = model.predict(img, conf=0.35, verbose=False)[0]

    out = overlay_mask(img, mask)

    for box, cls in zip(
        result.boxes.xyxy.cpu().numpy(),
        result.boxes.cls.int().tolist()
    ):
        x1, y1, x2, y2 = map(int, box)
        xa = max(0, x1 + int((x2-x1)*0.2))
        xb = min(img.shape[1], x2 - int((x2-x1)*0.2))
        ya = max(0, y1 + int((y2-y1)*0.7))
        yb = min(img.shape[0], y2)

        coverage = 0
        if xb > xa and yb > ya:
            coverage = (mask[ya:yb, xa:xb] > 0).mean() * 100

        color = (0, 255, 0) if coverage >= 50 else (0, 0, 255)
        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        cv2.putText(
            out, f"{result.names[cls]} {coverage:.1f}%",
            (x1, max(20,y1-7)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    output = f"results/road_debug/frame_{frame_no}.png"
    cv2.imwrite(output, out)
    print("Saved:", output)

print("Debug images created.")
