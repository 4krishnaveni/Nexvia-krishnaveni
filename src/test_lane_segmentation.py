
import cv2
from pathlib import Path
from lane_segmentation import RoadSegmenter, overlay_mask

image_path = Path("data/Dataset/Video17/images/0000000000.png")
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

img = cv2.imread(str(image_path))

if img is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

print("Loading SegFormer model...")
segmenter = RoadSegmenter()

print("Segmenting road...")
mask = segmenter(img)
overlay = overlay_mask(img, mask)

cv2.imwrite(str(output_dir / "road_mask.png"), mask)
cv2.imwrite(str(output_dir / "road_segmentation.png"), overlay)

print("Road pixels:", int((mask > 0).sum()))
print("Total pixels:", mask.size)
print("Road percentage:", round(100 * (mask > 0).mean(), 2), "%")

print("Saved results/road_mask.png")
print("Saved results/road_segmentation.png")
