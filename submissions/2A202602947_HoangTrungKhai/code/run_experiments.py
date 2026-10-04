"""Run the controlled backbone or training sweep on a GPU runtime.

Examples:
  python starter/run_experiments.py --stage backbones
  python starter/run_experiments.py --stage training --backbone convnext_tiny
Runs are deliberately single-seed screens. Final multi-seed runs belong after
all choices have been made using validation only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from train import Config, run


BACKBONES = [
    ("B01", "resnet50"),
    ("B02", "resnext50_32x4d"),
    ("B03", "convnext_tiny"),
    ("B04", "deit_small_patch16_224"),
    ("B05", "mobilenetv3_large_100"),
]


def training_configs(backbone: str):
    # Each T run changes one factor against T00 (except T07, the declared combination).
    base = dict(exp_id="T00", backbone=backbone)
    return [
        base,
        {"exp_id": "T01_frozen", "backbone": backbone, "init": "frozen"},
        {"exp_id": "T02_color", "backbone": backbone, "aug": "color"},
        {"exp_id": "T03_label_smoothing", "backbone": backbone, "loss": "ls", "label_smoothing": 0.1},
        {"exp_id": "T04_focal", "backbone": backbone, "loss": "focal", "focal_gamma": 2.0},
        {"exp_id": "T05_balanced_sampler", "backbone": backbone, "sampler": "balanced"},
        {"exp_id": "T06_mixup", "backbone": backbone, "mix": "mixup", "mix_alpha": 0.4},
        {"exp_id": "T07_combo", "backbone": backbone, "aug": "color", "loss": "ls",
         "label_smoothing": 0.1, "ema_decay": 0.999},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("backbones", "training"), required=True)
    parser.add_argument("--backbone", default="resnet50", help="backbone pivot chosen from validation results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    configs = ([{"exp_id": exp, "backbone": name} for exp, name in BACKBONES]
               if args.stage == "backbones" else training_configs(args.backbone))
    rows = []
    for values in configs:
        cfg = Config(**values, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size)
        row = run(cfg); rows.append(row)
        print(json.dumps(row, indent=2, default=str), flush=True)
        pd.DataFrame(rows).to_csv(Path("runs") / f"{args.stage}_summary.csv", index=False)


if __name__ == "__main__":
    main()
