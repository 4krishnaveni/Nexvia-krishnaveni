"""Step 3a: classical lane detection (colour mask + Canny + ROI + Hough + line fit)."""
import cv2
import numpy as np


def detect_lanes(bgr):
    """Returns (overlay_bgr, corridor_mask uint8, found: bool)."""
    h, w = bgr.shape[:2]
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    white = cv2.inRange(hls, (0, 190, 0), (255, 255, 255))
    yellow = cv2.inRange(hls, (15, 30, 115), (35, 204, 255))
    colour = cv2.dilate(cv2.bitwise_or(white, yellow), None, iterations=2)
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.bitwise_and(cv2.Canny(gray, 50, 150), colour)
    roi = np.zeros_like(edges)
    poly = np.array(
        [[(0.02*w, h), (0.40*w, 0.58*h), (0.60*w, 0.58*h), (0.98*w, h)]], np.int32)
    cv2.fillPoly(roi, poly, 255)
    edges = cv2.bitwise_and(edges, roi)
    segs = cv2.HoughLinesP(edges, 1, np.pi / 180, 25,
                           minLineLength=30, maxLineGap=80)

    left, right = [], []
    if segs is not None:
        for x1, y1, x2, y2 in segs.reshape(-1, 4):
            if y1 == y2:
                continue
            m = (y2 - y1) / (x2 - x1 + 1e-6)
            if abs(m) < 0.3:
                continue
            (left if m < 0 else right).append((x1, y1, x2, y2))

    def fit(lines):   # x = a*y + b, robust to near-vertical lines
        if not lines:
            return None
        ys = np.array([[l[1], l[3]] for l in lines]).ravel()
        xs = np.array([[l[0], l[2]] for l in lines]).ravel()
        return np.polyfit(ys, xs, 1)

    L, R = fit(left), fit(right)
    overlay, mask = bgr.copy(), np.zeros((h, w), np.uint8)
    y_top = int(0.62 * h)
    for f in (L, R):
        if f is not None:
            cv2.line(overlay, (int(np.polyval(f, h)), h),
                     (int(np.polyval(f, y_top)), y_top), (0, 255, 0), 4)
    found = L is not None and R is not None
    if found:
        pts = np.array([[(np.polyval(L, h), h), (np.polyval(L, y_top), y_top),
                         (np.polyval(R, y_top), y_top), (np.polyval(R, h), h)]], np.int32)
        cv2.fillPoly(mask, pts, 255)
        tint = overlay.copy()
        cv2.fillPoly(tint, pts, (0, 255, 0))
        overlay = cv2.addWeighted(tint, 0.25, overlay, 0.75, 0)
    return overlay, mask, found


if __name__ == "__main__":
    import sys
    img = cv2.imread(sys.argv[1])
    o, m, ok = detect_lanes(img)
    cv2.imwrite("results/lane_classical.png", o)
    print("both lanes found:", ok)
