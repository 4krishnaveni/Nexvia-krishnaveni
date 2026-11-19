"""Integrated perception stack: YOLO + ByteTrack + lanes + MiDaS distance + brake logic.
Usage: python src/pipeline.py --weights best.pt --source data/Dataset/Video16/images --out results/demo.mp4
`Perception` is reused by the ROS2 node."""
import argparse
import glob
import time
import csv
import cv2
from ultralytics import YOLO
from collections import deque
from depth import MiDaS, pinhole_distance
from lane_segmentation import RoadSegmenter

BRAKE_DIST_M = 12.0


class Perception:
    def __init__(self, weights, tracker="bytetrack.yaml", imgsz=1280, use_depth=True, use_road=True, engine=None):
        self.det = YOLO(engine or weights, task="detect")
        self.tracker, self.imgsz = tracker, imgsz

        self.depth = MiDaS() if use_depth else None
        self.road_segmenter = RoadSegmenter() if use_road else None

        self.road_history = {}
        self.history_size = 5

    def smooth_road_status(self, track_id, class_name, raw_status):
        if track_id < 0:
            return raw_status

        key = (class_name, track_id)

        if key not in self.road_history:
            self.road_history[key] = deque(maxlen=self.history_size)

        history = self.road_history[key]
        history.append(raw_status)

        # Majority vote across available observations
        counts = {
            status: history.count(status)
            for status in set(history)
        }

        smoothed_status = history[-1]
        best_count = 0

        for status in reversed(history):
            count = history.count(status)

            if count > best_count:
                best_count = count
                smoothed_status = status

        return smoothed_status

    def __call__(self, frame):
        t0 = time.perf_counter()

        r = self.det.track(
            frame,
            persist=True,
            tracker=self.tracker,
            imgsz=self.imgsz,
            verbose=False,
            conf=0.35
        )[0]

        vis = frame.copy()
        H, W = frame.shape[:2]

        # Generate road mask once per frame
        road_mask = self.road_segmenter(frame) if self.road_segmenter else None

        # MiDaS depth estimation
        inv = self.depth.inv_depth(frame) if self.depth else None

        objs = []
        nearest = None

        if r.boxes is not None and len(r.boxes):
            ids = (
                r.boxes.id.int().tolist()
                if r.boxes.id is not None
                else [-1] * len(r.boxes)
            )

            for b, c, i in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.cls.int().tolist(),
                ids
            ):
                name = r.names[c]
                x1, y1, x2, y2 = map(int, b)

                # Distance estimates (experimental)
                d_ph = pinhole_distance(b, name, W)
                d_mi = (
                    self.depth.metres(self.depth.box_inv(inv, b))
                    if self.depth else None
                )
                d = d_mi if d_mi is not None else d_ph

                # Lower 20% of detection box
                left = max(0, min(W, x1))
                right = max(0, min(W, x2))
                top = max(0, min(H, y1))
                bottom = max(0, min(H, y2))

                box_h = bottom - top
                box_w = right - left

                if road_mask is not None and box_h > 0 and box_w > 0:
                    lower_start = max(
                        top,
                        bottom - max(1, int(box_h * 0.20))
                    )
                    lower_region = road_mask[
                        lower_start:bottom, left:right
                    ]
                    coverage = (
                        float((lower_region > 0).mean() * 100)
                        if lower_region.size else None
                    )
                else:
                    coverage = None

                # Three-state experimental road assessment
                if coverage is None:
                    road_status = "UNKNOWN"
                elif coverage >= 40:
                    road_status = "ON_ROAD"
                elif coverage < 20:
                    road_status = "OFF_ROAD"
                else:
                    road_status = "UNKNOWN"

                raw_road_status = road_status

                smoothed_status = self.smooth_road_status(
                    i, name, raw_road_status
                )

                # Keep brake logic disabled during validation
                in_path = smoothed_status == "ON_ROAD"

                objs.append(dict(
                    id=i,
                    cls=name,
                    box=b.tolist(),
                    d_midas=d_mi,
                    d_pinhole=d_ph,
                    road_coverage=coverage,
                    road_status=raw_road_status,
                    smoothed_road_status=smoothed_status,
                    in_path=in_path
                ))

                if smoothed_status == "ON_ROAD":
                    col = (0, 200, 0)
                elif smoothed_status == "OFF_ROAD":
                    col = (0, 0, 255)
                else:
                    col = (0, 200, 255)

                cv2.rectangle(
                    vis,
                    (x1, y1),
                    (x2, y2),
                    col,
                    2
                )

                distance_text = f"{d:.1f}m" if d is not None else "N/A"
                coverage_text = (
                    f"{coverage:.0f}%"
                    if coverage is not None else "N/A"
                )

                label = (
                    f"{name}#{i} {distance_text} "
                    f"{smoothed_status} {coverage_text}"
                )

                cv2.putText(
                    vis,
                    label,
                    (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    col,
                    2
                )

                # Do not use experimental road status for braking
                # or claim that distance is safety-validated.

        ms = 1000 * (time.perf_counter() - t0)

        cv2.putText(
            vis,
            f"{1000 / max(ms, 0.001):.1f} FPS",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            vis,
            "BRAKING DISABLED - EXPERIMENTAL PERCEPTION",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2
        )

        return vis, dict(
            objects=objs,
            brake=False,
            nearest=None,
            ms=ms
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="results/demo.mp4")
    ap.add_argument("--no-depth", action="store_true")
    a = ap.parse_args()
    p = Perception(a.weights, use_depth=not a.no_depth)
    files = sorted(glob.glob(a.source + "/*.png"))
    w = None
    log = open("results/pipeline_log.csv", "w", newline="")
    cw = csv.writer(log)
    cw.writerow([
        "frame",
        "id",
        "cls",
        "d_midas",
        "d_pinhole",
        "road_coverage",
        "road_status",
        "smoothed_road_status",
        "in_path",
        "brake",
        "ms"
    ])
    for n, f in enumerate(files):
        vis, info = p(cv2.imread(f))
        if w is None:
            w = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(
                *"mp4v"), 10, (vis.shape[1], vis.shape[0]))
        w.write(vis)
        for o in info["objects"]:
            cw.writerow([
                n,
                o["id"],
                o["cls"],
                o["d_midas"],
                o["d_pinhole"],
                o["road_coverage"],
                o["road_status"],
                o["smoothed_road_status"],
                o["in_path"],
                info["brake"],
                round(info["ms"], 1)
            ])
    w.release()
    print("saved", a.out)
