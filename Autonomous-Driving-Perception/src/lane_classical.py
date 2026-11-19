
"""Classical lane detection with position and geometry validation."""
import cv2
import numpy as np


def detect_lanes(bgr):
    """Returns (overlay_bgr, corridor_mask uint8, found: bool)."""
    h, w = bgr.shape[:2]

    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    white = cv2.inRange(hls, (0, 190, 0), (255, 255, 255))
    yellow = cv2.inRange(hls, (15, 30, 115), (35, 204, 255))
    colour = cv2.dilate(cv2.bitwise_or(white, yellow), None, iterations=2)

    gray = cv2.GaussianBlur(
        cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0
    )
    edges = cv2.bitwise_and(cv2.Canny(gray, 50, 150), colour)

    roi = np.zeros_like(edges)
    poly = np.array([[
        (int(0.02*w), h-1),
        (int(0.40*w), int(0.58*h)),
        (int(0.60*w), int(0.58*h)),
        (int(0.98*w), h-1)
    ]], np.int32)
    cv2.fillPoly(roi, poly, 255)
    edges = cv2.bitwise_and(edges, roi)

    segs = cv2.HoughLinesP(
        edges, 1, np.pi / 180, 25,
        minLineLength=30, maxLineGap=80
    )

    left, right = [], []

    if segs is not None:
        for x1, y1, x2, y2 in segs.reshape(-1, 4):
            dx = x2 - x1
            dy = y2 - y1

            if abs(dx) < 1:
                continue

            slope = dy / dx
            if abs(slope) < 0.3:
                continue

            # Extrapolate the segment to the bottom of the image.
            a = dx / (dy + 1e-6)
            x_bottom = x1 + a * ((h - 1) - y1)

            # Reject lines outside plausible image regions.
            if slope < 0 and 0.08*w < x_bottom < 0.55*w:
                left.append((x1, y1, x2, y2))

            elif slope > 0 and 0.45*w < x_bottom < 0.92*w:
                right.append((x1, y1, x2, y2))

    def fit(lines):
        if not lines:
            return None

        ys = np.array([[line[1], line[3]] for line in lines]).ravel()
        xs = np.array([[line[0], line[2]] for line in lines]).ravel()

        return np.polyfit(ys, xs, 1)

    L, R = fit(left), fit(right)

    overlay = bgr.copy()
    mask = np.zeros((h, w), np.uint8)

    if L is None or R is None:
        return overlay, mask, False

    y_bottom = h - 1
    y_top = int(0.62 * h)

    lx_bottom = float(np.polyval(L, y_bottom))
    rx_bottom = float(np.polyval(R, y_bottom))
    lx_top = float(np.polyval(L, y_top))
    rx_top = float(np.polyval(R, y_top))

    bottom_width = rx_bottom - lx_bottom
    top_width = rx_top - lx_top

    # Reject crossed, implausibly narrow or excessively wide lanes.
    if not (0.20*w < bottom_width < 0.75*w):
        return overlay, mask, False

    if not (0.05*w < top_width < 0.55*w):
        return overlay, mask, False

    if not (0 <= lx_bottom < rx_bottom < w):
        return overlay, mask, False

    if not (0 <= lx_top < rx_top < w):
        return overlay, mask, False

    pts = np.array([[
        (int(lx_bottom), y_bottom),
        (int(lx_top), y_top),
        (int(rx_top), y_top),
        (int(rx_bottom), y_bottom)
    ]], np.int32)

    cv2.fillPoly(mask, pts, 255)

    cv2.line(overlay, tuple(pts[0][0]), tuple(pts[0][1]), (0, 255, 0), 4)
    cv2.line(overlay, tuple(pts[0][3]), tuple(pts[0][2]), (0, 255, 0), 4)

    tint = overlay.copy()
    cv2.fillPoly(tint, pts, (0, 255, 0))
    overlay = cv2.addWeighted(tint, 0.25, overlay, 0.75, 0)

    return overlay, mask, True


if __name__ == "__main__":
    import sys

    img = cv2.imread(sys.argv[1])
    if img is None:
        raise FileNotFoundError(sys.argv[1])

    overlay, mask, found = detect_lanes(img)
    cv2.imwrite("results/lane_classical.png", overlay)
    print("Valid lane corridor:", found)
