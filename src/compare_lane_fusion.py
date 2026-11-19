
import cv2
import csv
from pathlib import Path

from lane_segmentation import RoadSegmenter
from lane_classical import detect_lanes
from lane_fusion import detect_lanes_with_road_mask

frame_indices = [0, 25, 50, 75, 100, 125, 150, 200, 250, 300, 339]

image_dir = Path("data/Dataset/Video17/images")
output_dir = Path("results/lane_fusion_comparison")
output_dir.mkdir(parents=True, exist_ok=True)

frames = sorted(image_dir.glob("*.png"))

print("Loading SegFormer model...")
segmenter = RoadSegmenter()

results = []

for index in frame_indices:
    print(f"\nProcessing frame {index}...")

    img = cv2.imread(str(frames[index]))

    if img is None:
        print("Skipping unreadable image")
        continue

    # Classical detection
    classical_overlay, _, classical_found = detect_lanes(img)

    # SegFormer road segmentation
    road_mask = segmenter(img)

    # Combined fusion detection
    fusion_overlay, corridor, fusion_found = (
        detect_lanes_with_road_mask(img, road_mask)
    )

    road_percentage = round(
        100 * (road_mask > 0).mean(), 2
    )

    # Save side-by-side comparison
    comparison = cv2.hconcat([
        img,
        classical_overlay,
        fusion_overlay
    ])

    output_path = output_dir / f"fusion_frame_{index}.jpg"
    cv2.imwrite(str(output_path), comparison)

    results.append({
        "frame": index,
        "classical_found": classical_found,
        "fusion_found": fusion_found,
        "road_percentage": road_percentage,
        "image": str(output_path)
    })

    print("Classical:", classical_found)
    print("Fusion:", fusion_found)
    print("Road area:", road_percentage, "%")

# Save CSV
csv_path = output_dir / "fusion_results.csv"

with open(csv_path, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

# Calculate detection-success rates
total = len(results)

classical_count = sum(r["classical_found"] for r in results)
fusion_count = sum(r["fusion_found"] for r in results)

print("\n========== FUSION RESULTS ==========")
print("Frames processed:", total)
print(f"Classical: {classical_count}/{total}")
print(f"Fusion: {fusion_count}/{total}")

print(
    "Classical success rate:",
    round(classical_count / total * 100, 2), "%"
)

print(
    "Fusion success rate:",
    round(fusion_count / total * 100, 2), "%"
)

print("CSV saved:", csv_path)
print("Images saved:", output_dir)
