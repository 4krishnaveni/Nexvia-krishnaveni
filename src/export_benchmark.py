"""Step 6: export to ONNX + TensorRT and benchmark PyTorch vs ONNX vs TensorRT (FP16).
Needs an NVIDIA GPU + `pip install tensorrt`. Build the .engine on the GPU you will run it on.
Usage: python src/export_benchmark.py --weights runs/kitti_yolov8s/weights/best.pt --images data/Dataset/Video16/images"""
import argparse, glob, time, cv2, numpy as np, pandas as pd, wandb
from ultralytics import YOLO

ap = argparse.ArgumentParser(); ap.add_argument("--weights", required=True); ap.add_argument("--images", required=True)
ap.add_argument("--imgsz", type=int, default=1280); ap.add_argument("--n", type=int, default=200)
a = ap.parse_args()
wandb.init(project="av-perception", name="export-benchmark", job_type="benchmark")

YOLO(a.weights).export(format="onnx", imgsz=a.imgsz, simplify=True)
YOLO(a.weights).export(format="engine", imgsz=a.imgsz, half=True)           # TensorRT FP16
imgs = [cv2.imread(f) for f in sorted(glob.glob(a.images + "/*.png"))[: a.n]]
base = a.weights.rsplit(".", 1)[0]

def bench(path, label, half=False):
    m = YOLO(path, task="detect")
    for im in imgs[:20]: m.predict(im, imgsz=a.imgsz, half=half, verbose=False)   # warm-up
    t = []
    for im in imgs:
        s = time.perf_counter(); m.predict(im, imgsz=a.imgsz, half=half, verbose=False); t.append(1000 * (time.perf_counter() - s))
    return dict(model=label, mean_ms=np.mean(t), p95_ms=np.percentile(t, 95), fps=1000 / np.mean(t))

rows = [bench(a.weights, "PyTorch FP32"), bench(a.weights, "PyTorch FP16", True),
        bench(base + ".onnx", "ONNX"), bench(base + ".engine", "TensorRT FP16", True)]
df = pd.DataFrame(rows).round(2); print(df.to_string(index=False)); df.to_csv("results/benchmark.csv", index=False)
wandb.log({"benchmark": wandb.Table(dataframe=df)})
# Also prove accuracy didn't drop after optimisation:
# YOLO(base + ".engine", task="detect").val(data="splits/data.yaml", split="test", imgsz=a.imgsz)
