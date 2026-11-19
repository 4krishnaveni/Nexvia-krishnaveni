"""Step 4: MiDaS monocular depth + distance estimation.
MiDaS gives RELATIVE inverse depth (no metres). We convert to metres with a scale k calibrated
against the pinhole estimate d = f*H_real/h_pixels on cars (KITTI f ~ 721.5 px at 1242 px width).
For a rigorous accuracy number, validate against LiDAR depth from the official KITTI data."""
import json, pathlib, cv2, numpy as np, torch

REAL_HEIGHT_M = {"car": 1.5, "van": 1.9, "truck": 3.0, "pedestrian": 1.7, "person_sitting": 1.2,
                 "cyclist": 1.7, "tram": 3.5, "misc": 1.5}
F_KITTI, W_KITTI = 721.5, 1242.0
CALIB = pathlib.Path("results/depth_calib.json")

class MiDaS:
    def __init__(self, variant="MiDaS_small", device=None):
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("intel-isl/MiDaS", variant).to(self.dev).eval()
        t = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.tf = t.small_transform if variant == "MiDaS_small" else t.dpt_transform
        self.k = json.load(open(CALIB))["k"] if CALIB.exists() else None

    @torch.no_grad()
    def inv_depth(self, bgr):
        x = self.tf(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).to(self.dev)
        p = self.model(x).unsqueeze(1)
        p = torch.nn.functional.interpolate(p, size=bgr.shape[:2], mode="bicubic").squeeze()
        return p.cpu().numpy()

    @staticmethod
    def box_inv(inv, box):
        x1, y1, x2, y2 = [int(v) for v in box]
        cx = (x1 + x2) // 2; bw = max(2, (x2 - x1) // 4)
        y0 = int(y1 + 0.6 * (y2 - y1))
        ys = slice(y0, max(y0 + 1, y2))                       # lower part of the box (touches the road)
        return float(np.median(inv[ys, max(0, cx - bw): cx + bw]))

    def metres(self, inv_value):
        return None if (self.k is None or inv_value <= 1e-6) else self.k / inv_value

def pinhole_distance(box, cls_name, img_w):
    H = REAL_HEIGHT_M.get(cls_name.lower())
    h_px = box[3] - box[1]
    if H is None or h_px < 4: return None
    return (F_KITTI * img_w / W_KITTI) * H / h_px

if __name__ == "__main__":      # calibrate k:  python src/depth.py best.pt data/Dataset/Video16/images
    import sys, glob
    from ultralytics import YOLO
    det, files = YOLO(sys.argv[1]), sorted(glob.glob(sys.argv[2] + "/*.png"))[::5][:150]
    md, ks = MiDaS(), []
    for f in files:
        img = cv2.imread(f); inv = md.inv_depth(img)
        r = det.predict(img, verbose=False, conf=0.5)[0]
        for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.int().tolist()):
            if r.names[c].lower() != "car": continue
            d = pinhole_distance(b, "car", img.shape[1])
            if d and 5 < d < 40: ks.append(d * md.box_inv(inv, b))
    k = float(np.median(ks)); CALIB.parent.mkdir(exist_ok=True)
    json.dump({"k": k, "n_samples": len(ks), "iqr": [float(np.percentile(ks, 25)), float(np.percentile(ks, 75))]},
              open(CALIB, "w")); print("k =", k, "from", len(ks), "cars")
