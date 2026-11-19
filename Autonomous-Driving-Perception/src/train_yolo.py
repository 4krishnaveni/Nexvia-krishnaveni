"""Step 2a: fine-tune YOLO. W&B logging is automatic once `wandb login` is done
and `yolo settings wandb=True` is set.
Usage: python src/train_yolo.py --model yolov8s.pt --epochs 50
"""
import argparse
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="yolov8s.pt")
ap.add_argument("--data", default="splits/data.yaml")
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--imgsz", type=int, default=1280)   # KITTI is 1242x375: wide images, small objects
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--name", default="kitti_yolov8s")
a = ap.parse_args()

model = YOLO(a.model)
model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, name=a.name,
            project="runs", patience=15, cos_lr=True, close_mosaic=10, plots=True)
metrics = model.val(data=a.data, split="test", imgsz=a.imgsz, name=a.name + "_test")
print("TEST mAP50:", metrics.box.map50, " mAP50-95:", metrics.box.map)
print("per-class mAP50-95:", dict(zip(metrics.names.values(), metrics.box.maps)))
