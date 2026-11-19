
import cv2
from pathlib import Path
from lane_segmentation import RoadSegmenter

segmenter = RoadSegmenter()

frames = [
    "0000000001.png",
    "0000000012.png",
    "0000000096.png",
    "0000000286.png",
]

image_dir = Path("data/Dataset/Video16/images")

for filename in frames:
    image = cv2.imread(str(image_dir / filename))

    if image is None:
        print(f"{filename}: Image not found")
        continue

    mask = segmenter(image)
    road_percent = 100 * (mask > 0).mean()

    print(
        f"{filename}: "
        f"Road pixels = {(mask > 0).sum()}, "
        f"Road coverage = {road_percent:.2f}%"
    )
