
"""Combined road-aware lane detection using SegFormer and classical CV."""

import cv2
import numpy as np

from lane_segmentation import RoadSegmenter, overlay_mask


def detect_lanes_with_road_mask(bgr, road_mask):
    """
    Detect possible lane boundaries using SegFormer's road mask.

    Returns:
        overlay: visualization of road and lane boundaries
        corridor_mask: estimated area between both boundaries
        found: whether both boundaries were detected
    """

    h, w = bgr.shape[:2]

    if road_mask.shape[:2] != (h, w):
        road_mask = cv2.resize(
            road_mask, (w, h),
            interpolation=cv2.INTER_NEAREST
        )

    # Restrict edge detection to the predicted road area
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)

    road_edges = cv2.bitwise_and(edges, road_mask)

    # Focus on the lower road region
    roi = np.zeros_like(road_edges)
    polygon = np.array([[
        (int(0.02 * w), h - 1),
        (int(0.40 * w), int(0.58 * h)),
        (int(0.60 * w), int(0.58 * h)),
        (int(0.98 * w), h - 1)
    ]], dtype=np.int32)

    cv2.fillPoly(roi, polygon, 255)
    road_edges = cv2.bitwise_and(road_edges, roi)

    # Detect line candidates
    segments = cv2.HoughLinesP(
        road_edges,
        1,
        np.pi / 180,
        25,
        minLineLength=30,
        maxLineGap=80
    )

    left_lines = []
    right_lines = []

    if segments is not None:
        for x1, y1, x2, y2 in segments.reshape(-1, 4):
            if x2 == x1:
                continue

            slope = (y2 - y1) / (x2 - x1)

            if abs(slope) < 0.3:
                continue

            if slope < 0:
                left_lines.append((x1, y1, x2, y2))
            else:
                right_lines.append((x1, y1, x2, y2))

    def fit_line(lines):
        if not lines:
            return None

        ys = np.array(
            [[line[1], line[3]] for line in lines]
        ).flatten()

        xs = np.array(
            [[line[0], line[2]] for line in lines]
        ).flatten()

        return np.polyfit(ys, xs, 1)

    left_fit = fit_line(left_lines)
    right_fit = fit_line(right_lines)

    # Show road area in blue
    overlay = overlay_mask(bgr, road_mask, colour=(255, 0, 0))

    corridor_mask = np.zeros((h, w), dtype=np.uint8)
    y_bottom = h - 1
    y_top = int(0.62 * h)

    def point_on_line(fit, y):
        x = int(np.clip(np.polyval(fit, y), 0, w - 1))
        return (x, y)

    if left_fit is not None:
        cv2.line(
            overlay,
            point_on_line(left_fit, y_bottom),
            point_on_line(left_fit, y_top),
            (0, 255, 0),
            4
        )

    if right_fit is not None:
        cv2.line(
            overlay,
            point_on_line(right_fit, y_bottom),
            point_on_line(right_fit, y_top),
            (0, 255, 0),
            4
        )

    found = left_fit is not None and right_fit is not None

    if found:
        points = np.array([[
            point_on_line(left_fit, y_bottom),
            point_on_line(left_fit, y_top),
            point_on_line(right_fit, y_top),
            point_on_line(right_fit, y_bottom)
        ]], dtype=np.int32)

        cv2.fillPoly(corridor_mask, points, 255)

        tint = overlay.copy()
        cv2.fillPoly(tint, points, (0, 255, 0))
        overlay = cv2.addWeighted(tint, 0.2, overlay, 0.8, 0)

    return overlay, corridor_mask, found


if __name__ == "__main__":
    from pathlib import Path

    image_path = Path(
        "data/Dataset/Video17/images/0000000000.png"
    )

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(image_path)

    print("Loading SegFormer...")
    segmenter = RoadSegmenter()

    print("Generating road mask...")
    road_mask = segmenter(image)

    print("Detecting lanes inside road area...")
    result, corridor, found = detect_lanes_with_road_mask(
        image, road_mask
    )

    cv2.imwrite("results/fusion_frame0.png", result)
    cv2.imwrite("results/fusion_corridor_frame0.png", corridor)

    print("Both lane boundaries found:", found)
    print("Saved results/fusion_frame0.png")
    print("Saved results/fusion_corridor_frame0.png")
