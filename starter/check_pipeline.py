"""Run the required initial-loss and tiny-batch overfit checks on real images."""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from dataset import build_transforms, load_split, make_loader
from model import build_model
from train import set_seed


def check(images_dir="data/images", labels_dir="data/labels", out_dir="eda", steps=80):
    set_seed(11)
    train_df, _, _ = load_split(labels_dir, 0)
    loader = make_loader(train_df.iloc[:8], images_dir, build_transforms(True, 64, "basic"),
                         batch_size=8, train=False, num_workers=0, seed=11)
    x, y, _ = next(iter(loader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = build_model("mobilenetv3_large_100", pretrained=False, init="scratch").to(device)
    x, y = x.to(device), y.to(device)
    net.eval()
    with torch.inference_mode(): initial = float(F.cross_entropy(net(x), y))
    net.train(); opt = torch.optim.AdamW(net.parameters(), lr=0.003, weight_decay=0.0)
    curve = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(net(x), y); loss.backward(); opt.step()
        curve.append(float(loss.detach()))
    result = {"initial_ce": initial, "uniform_9_class_reference": float(np.log(9)),
              "final_tiny_batch_ce": curve[-1], "steps": steps, "samples": int(len(y)),
              "device": str(device), "overfit_threshold_0.1": curve[-1] < 0.1}
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out/"pipeline_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    # Plot the fixed augmented batch after undoing ImageNet normalization.
    images = (x[:8].detach().cpu()*torch.tensor([.229,.224,.225]).view(1,3,1,1)
              + torch.tensor([.485,.456,.406]).view(1,3,1,1)).clamp(0,1)
    fig, axes = plt.subplots(2, 4, figsize=(10,5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].permute(1,2,0).numpy()); ax.set_title(f"label={int(y[i])}"); ax.axis("off")
    fig.tight_layout(); fig.savefig(out/"augmented_batch.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6,4)); ax.plot(range(1,steps+1), curve)
    ax.set(xlabel="Optimization step", ylabel="Cross-entropy", title="Tiny-batch overfit check")
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out/"tiny_batch_overfit.png", dpi=160); plt.close(fig)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__": check()
