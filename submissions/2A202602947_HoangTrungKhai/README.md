# Lab Day 2 — Hoang Trung Khai (2A202602947)

## Reproduce

This submission uses the official DeepWeeds fold 0. From the repository root, place the dataset at `data/images/` and the four CSV files at `data/labels/` as described in the repository README. The data files are intentionally excluded from this submission.

Install `torch`, `torchvision`, `timm`, `pandas`, `numpy`, `Pillow`, `scikit-learn`, and `matplotlib`. The implementation was prepared with Python 3.10, PyTorch 2.13.0, torchvision 0.28.0, and timm 1.0.30. A CUDA GPU is recommended; this workspace currently has no available CUDA device.

From the repository root:

```bash
python3 submissions/2A202602947_HoangTrungKhai/code/eda.py
python3 submissions/2A202602947_HoangTrungKhai/code/check_pipeline.py
python3 submissions/2A202602947_HoangTrungKhai/code/run_experiments.py --stage backbones
```

After selecting a backbone using validation macro-F1 and measured latency, run the controlled training comparisons:

```bash
python3 submissions/2A202602947_HoangTrungKhai/code/run_experiments.py --stage training --backbone resnet50
```

The copied results include five backbone screening runs (B01–B05), seed 0, with epoch logs, validation curves, validation predictions, configs and validation logits under `runs/`, `curves/` and `predictions/`. The best validation macro-F1 is ConvNeXt-Tiny (B03): 0.9694 (top-1 0.9763). This is a one-seed screening result, not a final result. Backbone latency was not measured. The large `.pt` checkpoints are omitted from this submission folder. Final experiments still require controlled training comparisons and at least three seeds; test predictions are not present. Do not use test results to make configuration choices.

Colab/Kaggle notebook link: **add the shareable notebook URL after uploading/running this code**.

## Current status

Fold 0 and the pipeline have been checked, and the five backbone screening runs are copied into this package. EDA artifacts are in `eda/`; training logs/configs/validation logits are in `runs/`; curves are in `curves/`; validation predictions are in `predictions/`. No test predictions or three-seed final comparison exist yet, so this package is not complete for final rubric scoring. The notebook URL also remains to be added after upload.
