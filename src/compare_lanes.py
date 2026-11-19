"""Step 3c: classical vs segmentation on N frames -> side-by-side images, speed, agreement.
Usage: python src/compare_lanes.py data/Dataset/Video16/images --n 100
"""
import argparse, glob, time, cv2, json, numpy as np
from lane_classical import detect_lanes
from lane_segmentation import RoadSegmenter, overlay_mask

ap = argparse.ArgumentParser(); ap.add_argument("folder"); ap.add_argument("--n", type=int, default=100)
a = ap.parse_args()
allf = sorted(glob.glob(a.folder + "/*.png"))
files = allf[:: max(1, len(allf) // a.n)][: a.n]
seg = RoadSegmenter(); seg(cv2.imread(files[0]))  # warm-up
t_c, t_s, found, inside = [], [], 0, []
for i, f in enumerate(files):
    img = cv2.imread(f)
    t = time.perf_counter(); o_c, corridor, ok = detect_lanes(img); t_c.append(time.perf_counter() - t)
    t = time.perf_counter(); road = seg(img); t_s.append(time.perf_counter() - t)
    found += ok
    if ok and corridor.sum() > 0:   # proxy metric: share of ego corridor lying on predicted road
        inside.append(float(((corridor > 0) & (road > 0)).sum() / (corridor > 0).sum()))
    if i % 10 == 0:
        cv2.imwrite(f"results/lane_cmp_{i:03d}.jpg", np.vstack([o_c, overlay_mask(img, road)]))
res = {"frames": len(files), "classical_ms": 1000 * np.mean(t_c), "segformer_ms": 1000 * np.mean(t_s),
       "classical_both_lanes_found_pct": 100 * found / len(files),
       "corridor_inside_road_mean": float(np.mean(inside)) if inside else None}
print(json.dumps(res, indent=2)); json.dump(res, open("results/lane_comparison.json", "w"), indent=2)
