"""Run the validation-selected recipe and T00 baseline on test for >=3 seeds.

Run only after `run_experiments.py --stage training` has completed. Recipe selection
uses training_summary.csv validation macro-F1 only; test is touched only by final runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from train import Config, run
from run_experiments import training_configs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="convnext_tiny")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--images-dir", default="../data/images")
    ap.add_argument("--labels-dir", default="../data/labels")
    ap.add_argument("--summary", default="runs/training_summary.csv")
    ap.add_argument("--inference-summary", default="runs/inference_summary.csv")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--pred-dir", default="predictions")
    ap.add_argument("--curves-dir", default="curves")
    args = ap.parse_args()
    if len(set(args.seeds)) < 3:
        ap.error("Nộp kết quả chung kết cần ít nhất 3 seed khác nhau")
    summary = pd.read_csv(args.summary)
    required = {"exp_id", "macro_f1_val"}
    if not required.issubset(summary.columns):
        ap.error(f"{args.summary} thiếu cột {sorted(required - set(summary.columns))}")
    candidates = summary[summary.exp_id.isin([x["exp_id"] for x in training_configs(args.backbone)])]
    if candidates.empty:
        ap.error("Không thấy kết quả ablation cho backbone này; chạy stage training trước")
    chosen_id = str(candidates.sort_values("macro_f1_val", ascending=False).iloc[0].exp_id)
    chosen = next(x for x in training_configs(args.backbone) if x["exp_id"] == chosen_id)
    inf_path=Path(args.inference_summary)
    if not inf_path.is_file():
        ap.error("Chạy inference_eval.py trên checkpoint ablation tốt nhất trước final stage")
    inf=pd.read_csv(inf_path).sort_values(["macro_f1_val","ece_val","p95_ms"],
                                          ascending=[False,True,True],na_position="last")
    inference_method=str(inf.iloc[0]["method"])
    print(f"Validation-only selection: recipe={chosen_id} {chosen}; inference={inference_method}", flush=True)

    for seed in args.seeds:
        for exp_id, recipe in (("T00", {"exp_id": "T00", "backbone": args.backbone}),
                               ("F01", {**chosen, "exp_id": "F01", "inference_method": inference_method})):
            cfg = Config(**recipe, seed=seed, epochs=args.epochs, batch_size=args.batch_size,
                         images_dir=args.images_dir, labels_dir=args.labels_dir,
                         out_dir=args.out_dir, pred_dir=args.pred_dir,
                         save_test_predictions=True)
            cfg.curves_dir = args.curves_dir
            result = run(cfg)
            print(json.dumps({"final_id": exp_id, **result}, default=str), flush=True)


if __name__ == "__main__":
    main()
