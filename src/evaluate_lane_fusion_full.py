
import cv2
import csv
import time
from pathlib import Path

from lane_segmentation import RoadSegmenter
from lane_classical import detect_lanes
from lane_fusion import detect_lanes_with_road_mask

videos = ["Video17", "Video18"]

output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

csv_path = output_dir / "lane_fusion_full_test.csv"

# Collect all test frames
all_frames = []

for video in videos:
    image_dir = Path("data/Dataset") / video / "images"
    frames = sorted(image_dir.glob("*.png"))

    print(f"{video}: {len(frames)} frames")

    for index, frame_path in enumerate(frames):
        all_frames.append((video, index, frame_path))

total = len(all_frames)

if total == 0:
    raise FileNotFoundError("No test images found.")

print("\nTotal frames:", total)

# Load model once
print("\nLoading SegFormer...")
segmenter = RoadSegmenter()

classical_success = 0
fusion_success = 0
processed = 0
start_time = time.time()

# Save results continuously
with open(csv_path, "w", newline="") as file:

    fields = [
        "video",
        "frame_index",
        "image",
        "classical_found",
        "fusion_found",
        "road_percentage"
    ]

    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()

    for video, index, image_path in all_frames:

        image = cv2.imread(str(image_path))

        if image is None:
            print("Skipping unreadable:", image_path)
            continue

        # Classical detection
        _, _, classical_found = detect_lanes(image)

        # SegFormer road segmentation
        road_mask = segmenter(image)

        # Combined lane fusion
        _, _, fusion_found = detect_lanes_with_road_mask(
            image, road_mask
        )

        road_percentage = round(
            100 * (road_mask > 0).mean(), 2
        )

        classical_success += int(classical_found)
        fusion_success += int(fusion_found)
        processed += 1

        writer.writerow({
            "video": video,
            "frame_index": index,
            "image": str(image_path),
            "classical_found": classical_found,
            "fusion_found": fusion_found,
            "road_percentage": road_percentage
        })

        # Save progress after every frame
        file.flush()

        if processed % 10 == 0 or processed == total:
            elapsed = round(time.time() - start_time, 1)
            print(
                f"Progress: {processed}/{total} | "
                f"Classical: {classical_success} | "
                f"Fusion: {fusion_success} | "
                f"Elapsed: {elapsed}s"
            )

print("\n========== FULL TEST RESULTS ==========")
print("Frames processed:", processed)
print("Classical successes:", classical_success)
print("Fusion successes:", fusion_success)

if processed:
    print(
        "Classical success rate:",
        round(classical_success / processed * 100, 2), "%"
    )

    print(
        "Fusion success rate:",
        round(fusion_success / processed * 100, 2), "%"
    )

print("\nCSV saved:", csv_path)
print("Evaluation completed.")
