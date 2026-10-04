"""Create the required fold-0 class distribution and sample image sheet."""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from dataset import CLASS_NAMES, check_split, load_split


def make_eda(labels_dir="data/labels", images_dir="data/images", out_dir="eda"):
    train, val, test = load_split(labels_dir, 0)
    stats = check_split(train, val, test, images_dir)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.26
    x = range(len(CLASS_NAMES))
    for offset, (split, df, color) in enumerate((("Train", train, "#3b82f6"), ("Val", val, "#f59e0b"), ("Test", test, "#10b981"))):
        counts = df.Label.value_counts().reindex(range(9), fill_value=0)
        pos = [i + (offset-1)*width for i in x]
        ax.bar(pos, counts, width, label=f"{split} (n={len(df)})", color=color)
    ax.set_xticks(list(x), CLASS_NAMES, rotation=35, ha="right")
    ax.set_ylabel("Number of images"); ax.set_title("DeepWeeds fold 0: class distribution")
    ax.legend(); ax.grid(axis="y", alpha=.2); fig.tight_layout()
    fig.savefig(out/"class_distribution.png", dpi=180); plt.close(fig)
    # Select deterministic first three filenames per class, cycling through splits.
    merged = pd.concat([train.assign(Split="train"), val.assign(Split="val"), test.assign(Split="test")])
    fig, axes = plt.subplots(9, 3, figsize=(9, 23))
    for label in range(9):
        rows = merged[merged.Label == label].head(3)
        for col, ax in enumerate(axes[label]):
            ax.axis("off")
            if col < len(rows):
                row = rows.iloc[col]
                with Image.open(Path(images_dir)/row.Filename) as image:
                    ax.imshow(image.convert("RGB"))
                ax.set_title(f"{CLASS_NAMES[label]}\n{row.Filename}", fontsize=8)
    fig.tight_layout(); fig.savefig(out/"sample_images.png", dpi=140); plt.close(fig)
    pd.DataFrame({"class": CLASS_NAMES,
                  "train": train.Label.value_counts().reindex(range(9), fill_value=0).to_numpy(),
                  "val": val.Label.value_counts().reindex(range(9), fill_value=0).to_numpy(),
                  "test": test.Label.value_counts().reindex(range(9), fill_value=0).to_numpy()}).to_csv(out/"class_counts.csv", index=False)
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-dir", default="data/labels"); parser.add_argument("--images-dir", default="data/images")
    parser.add_argument("--out-dir", default="eda"); args = parser.parse_args()
    make_eda(args.labels_dir, args.images_dir, args.out_dir)
