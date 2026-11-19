
import cv2
import csv
from pathlib import Path

from lane_classical import detect_lanes
from lane_segmentation import RoadSegmenter, overlay_mask

# Select 11 frames from Video17
frame_indices = [0, 25, 50, 75, 100, 125, 150, 200, 250, 300, 339]

image_dir = Path("data/Dataset/Video17/images")
output_dir = Path("results/lane_comparison")
output_dir.mkdir(parents=True, exist_ok=True)

frames = sorted(image_dir.glob("*.png"))

if not frames:
    raise FileNotFoundError(f"No images found in {image_dir}")

if max(frame_indices) >= len(frames):
    raise IndexError("A selected frame index exceeds the available frames.")

print("Loading SegFormer model...")
segmenter = RoadSegmenter()

results = []

for index in frame_indices:
    image_path = frames[index]
    print(f"\nProcessing frame {index}: {image_path.name}")

    img = cv2.imread(str(image_path))

    if img is None:
        print("Could not read image. Skipping.")
        continue

    # Classical lane detection
    classical_overlay, classical_mask, lanes_found = detect_lanes(img)

    # SegFormer road segmentation
    road_mask = segmenter(img)
    segformer_overlay = overlay_mask(img, road_mask)

    road_percentage = round(
        100 * (road_mask > 0).mean(), 2
    )

    # Create side-by-side comparison
    comparison = cv2.hconcat([
        img,
        classical_overlay,
        segformer_overlay
    ])

    output_path = output_dir / f"comparison_frame_{index}.jpg"
    cv2.imwrite(str(output_path), comparison)

    results.append({
        "frame": index,
        "classical_both_lanes_found": lanes_found,
        "segformer_road_percentage": road_percentage,
        "comparison_image": str(output_path)
    })

    print("Classical: Both lanes found =", lanes_found)
    print("SegFormer: Road percentage =", road_percentage, "%")
    print("Saved:", output_path)

# Save results as CSV
csv_path = output_dir / "comparison_results.csv"

with open(csv_path, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

classical_success = sum(
    row["classical_both_lanes_found"] for row in results
)

print("\n========== FINAL COMPARISON ==========")
print("Frames processed:", len(results))
print(
    "Classical both-lane detection:",
    f"{classical_success}/{len(results)}"
)
print(
    "Classical detection rate:",
    round(100 * classical_success / len(results), 2), "%"
)
print("SegFormer road percentages:")

for row in results:
    print(
        f"Frame {row['frame']}: "
        f"{row['segformer_road_percentage']}%"
    )

print("\nCSV saved:", csv_path)
print("Comparison images saved:", output_dir)
