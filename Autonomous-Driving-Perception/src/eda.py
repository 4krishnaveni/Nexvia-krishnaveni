"""Step 1: explore labels + class balance.
Usage: python src/eda.py data/Dataset data/Dataset/dataset.yaml
"""
import sys, pathlib, yaml
import pandas as pd, matplotlib.pyplot as plt

root, ycfg = pathlib.Path(sys.argv[1]), sys.argv[2]
names = yaml.safe_load(open(ycfg))["names"]
names = names if isinstance(names, dict) else dict(enumerate(names))
rows, empty, total = [], 0, 0
for lf in root.glob("Video*/labels/*.txt"):
    total += 1
    lines = [l.split() for l in lf.read_text().splitlines() if l.strip()]
    empty += (len(lines) == 0)
    for p in lines:
        c, x, y, w, h = int(p[0]), *map(float, p[1:5])
        rows.append((lf.parent.parent.name, lf.stem, c, x, y, w, h))
df = pd.DataFrame(rows, columns=["video", "frame", "cls", "x", "y", "w", "h"])
df["name"] = df.cls.map(names)

pathlib.Path("results").mkdir(exist_ok=True)
counts = df.name.value_counts()
print(f"{total} label files, {empty} empty, {len(df)} boxes\n{counts}\n")
print("median box size (normalised) per class:\n", df.groupby("name")[["w", "h"]].median())
counts.to_csv("results/class_counts.csv")

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
counts.plot.bar(ax=ax[0], title="Class balance (boxes)"); ax[0].set_yscale("log")
df.groupby("video").size().plot.bar(ax=ax[1], title="Boxes per video")
ax[2].hist2d(df.x, df.y, bins=40); ax[2].set_title("Box centre heatmap"); ax[2].invert_yaxis()
plt.tight_layout(); plt.savefig("results/eda.png", dpi=150)
