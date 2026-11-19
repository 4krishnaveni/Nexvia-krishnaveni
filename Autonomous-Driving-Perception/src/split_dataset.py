"""Split KITTI-YOLO by VIDEO (not by frame) so consecutive near-identical frames
don't leak between train and val. Writes splits/{train,val,test}.txt + splits/data.yaml.

Usage: python src/split_dataset.py data/Dataset
"""
import re, sys, yaml, pathlib

root = pathlib.Path(sys.argv[1]).resolve()
out = pathlib.Path("splits"); out.mkdir(exist_ok=True)

def vid_num(p): return int(re.findall(r"\d+", p.name)[0])
videos = sorted([p for p in root.glob("Video*") if p.is_dir()], key=vid_num)
assert videos, f"No Video* folders under {root}"

n = len(videos)
parts = {"train": videos[: n - 5], "val": videos[n - 5 : n - 2], "test": videos[n - 2 :]}
for name, vids in parts.items():
    imgs = [str(i) for v in vids for i in sorted((v / "images").glob("*.png"))]
    (out / f"{name}.txt").write_text("\n".join(imgs))
    print(f"{name}: {len(vids)} videos, {len(imgs)} frames -> {[v.name for v in vids]}")

orig = yaml.safe_load(open(root / "dataset.yaml")) if (root / "dataset.yaml").exists() \
    else yaml.safe_load(open(root.parent / "dataset.yaml"))
cfg = {"path": str(out.resolve()), "train": "train.txt", "val": "val.txt", "test": "test.txt",
       "names": orig["names"]}
yaml.safe_dump(cfg, open(out / "data.yaml", "w"))
print("classes:", orig["names"])
