"""Step 2b: detection + tracking on a frame folder / video (ByteTrack or BoT-SORT).
Usage: python src/track.py --weights runs/kitti_yolov8s/weights/best.pt \
          --source data/Dataset/Video16/images --out results/track_video16.mp4
"""
import argparse, glob, cv2, collections
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True); ap.add_argument("--source", required=True)
ap.add_argument("--out", default="results/tracked.mp4")
ap.add_argument("--tracker", default="bytetrack.yaml")   # or botsort.yaml
ap.add_argument("--imgsz", type=int, default=1280); ap.add_argument("--fps", type=int, default=10)
a = ap.parse_args()

model = YOLO(a.weights)
frames = sorted(glob.glob(a.source + "/*.png")) if not a.source.endswith((".mp4", ".avi")) else None
cap = None if frames else cv2.VideoCapture(a.source)
writer, ids_seen, n = None, collections.defaultdict(set), 0
while True:
    if frames:
        if n >= len(frames): break
        img = cv2.imread(frames[n])
    else:
        ok, img = cap.read()
        if not ok: break
    r = model.track(img, persist=True, tracker=a.tracker, imgsz=a.imgsz, verbose=False)[0]
    if r.boxes.id is not None:
        for i, c in zip(r.boxes.id.int().tolist(), r.boxes.cls.int().tolist()):
            ids_seen[r.names[c]].add(i)
    vis = r.plot()
    if writer is None:
        h, w = vis.shape[:2]
        writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))
    writer.write(vis); n += 1
writer.release()
print(f"{n} frames. unique track IDs per class:", {k: len(v) for k, v in ids_seen.items()})
