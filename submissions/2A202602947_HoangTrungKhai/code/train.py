"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

PSEUDO-CODE: chỉ có khung (cấu hình và quy ước đặt tên file); bạn tự hoàn thiện mọi hàm có
`raise NotImplementedError` và các bước TODO trong `run()`. Dùng MỘT hàm `run(cfg)` cho mọi cấu hình
(RUBRIC mục H): đổi thí nghiệm chỉ bằng cách đổi `Config`.

Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
Chỉ số dùng để chọn checkpoint (macro-F1 val) phải tính bằng eval.compute_metrics của repo gốc,
để cùng định nghĩa với lúc chấm:
    sys.path.insert(0, "<thư mục chứa eval.py>");  from eval import compute_metrics
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import copy
import importlib.util
import json
import random
import sys
import time
import types
import typing

import numpy as np
import pandas as pd

# Ghi file dự đoán đúng định dạng bằng hàm có sẵn trong eval.py (repo gốc):
#     from eval import save_predictions, compute_metrics
# Log theo epoch (history.csv) và config.json bạn tự ghi bằng pandas/json.


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug ...
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def _load_eval_module():
    """Load the repository's unchanged eval.py from a parent directory.

    This also works when train.py is run from a copied submissions/<student>/code
    directory, where the repository root is not automatically on sys.path.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "eval.py"
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location("deepweeds_eval", candidate)
            if spec is None or spec.loader is None:
                raise ImportError(f"Không thể nạp {candidate}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError("Không tìm thấy eval.py trong các thư mục cha của train.py")


def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên.

    TODO: random, numpy, torch (CPU và CUDA); cân nhắc cudnn.deterministic/benchmark và
    seed cho worker của DataLoader. Ghi lại trong báo cáo mức độ tái lập bạn đạt được.
    """
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model, cfg: Config):
    """AdamW với 3 nhóm tham số (xem model.param_groups). TODO."""
    import torch
    from model import param_groups
    groups = param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    if not groups: raise ValueError("Không có tham số train được")
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về ~0 (slide trang 55). TODO.

    Cập nhật theo bước (iteration) hoặc theo epoch đều được; ghi rõ bạn chọn gì.
    Gợi ý kiểm tra: vẽ đường LR theo bước để thấy đúng hình warmup + cosine.
    """
    import math
    import torch
    total = max(1, cfg.epochs * steps_per_epoch)
    warmup = min(total - 1, int(cfg.warmup_epochs * steps_per_epoch))
    def factor(step):
        if warmup and step < warmup: return max(1e-8, (step + 1) / warmup)
        progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W  (slide trang 56).

    TODO:
      - __init__(model, decay): sao chép trọng số
      - update(model): sau mỗi bước tối ưu
      - copy_to(model) hoặc dùng bản sao riêng để đánh giá bằng trọng số EMA
      - lưu ý BatchNorm: buffer (running_mean/var) cũng phải được xử lý hợp lý
    """

    def __init__(self, model, decay: float):
        self.decay = float(decay)
        if not 0 < self.decay < 1: raise ValueError("EMA decay phải thuộc (0,1)")
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model) -> None:
        import torch
        state = model.state_dict()
        for k, value in state.items():
            if torch.is_floating_point(value): self.shadow[k].mul_(self.decay).add_(value.detach(), alpha=1-self.decay)
            else: self.shadow[k].copy_(value)

    def copy_to(self, model): model.load_state_dict(self.shadow, strict=True)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    """Một epoch huấn luyện. Trả về dict, ví dụ {"train_loss": ..., "lr": ...}.

    TODO:
      - model.train() (nếu init == "frozen": giữ phần backbone ở eval, xem model.freeze_backbone)
      - nếu cfg.mix: mix_batch rồi mixed_loss (losses.py)
      - AMP (autocast + GradScaler), clip gradient nếu cần, optimizer.step(), scheduler.step()
      - nếu có EMA: ema.update(model)
    """
    import torch
    import model as model_lib
    import losses as losses_lib
    model.train()
    if cfg.init == "frozen":
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm): module.eval()
    device = torch.device(device); use_amp = bool(cfg.amp and device.type == "cuda")
    total_loss = 0.0; n = 0; start = time.perf_counter()
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        targets = y
        if cfg.mix:
            x, targets = losses_lib.mix_batch(x, y, cfg.mix_alpha, cfg.mix)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
            loss = losses_lib.mixed_loss(criterion, logits, targets) if cfg.mix else criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer); scaler.update(); scheduler.step()
        if ema is not None: ema.update(model)
        total_loss += float(loss.detach()) * y.size(0); n += y.size(0)
    return {"train_loss": total_loss/max(1,n), "epoch_seconds": time.perf_counter()-start,
            "lr": max(group["lr"] for group in optimizer.param_groups)}


def evaluate(model, loader, criterion, device):
    """Chạy model trên một loader ở chế độ eval, KHÔNG tính gradient.

    Trả về (filenames: list[str], y_true: ndarray[N], logits: ndarray[N, 9], loss: float).
    Giữ đúng thứ tự của loader để ghép logit với tên file.

    TODO: model.eval(), torch.inference_mode(), gom kết quả. Softmax khi cần xác suất.
    """
    import torch
    model.eval(); device = torch.device(device)
    names, ys, zs = [], [], []; loss_sum = 0.0; n = 0
    with torch.inference_mode():
        for x, y, filenames in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x).float()
            loss_sum += float(criterion(logits, y))*y.size(0); n += y.size(0)
            names.extend(list(filenames)); ys.append(y.cpu().numpy()); zs.append(logits.cpu().numpy())
    return names, np.concatenate(ys), np.concatenate(zs), loss_sum/max(1,n)


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    """Vẽ đường cong training của một thí nghiệm -> curves/<exp_id>_<mota>.png (GUIDE.md mục 6.2).

    TODO: tối thiểu loss train/val và macro-F1 val theo epoch; có tiêu đề, nhãn trục, chú thích;
    khuyến khích thêm LR theo bước. Lưu bằng matplotlib với dpi đủ nét để đọc số.
    """
    import matplotlib.pyplot as plt
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [r["train_loss"] for r in history], label="train")
    axes[0].plot(epochs, [r["val_loss"] for r in history], label="val")
    axes[0].set(xlabel="Epoch", ylabel="Loss"); axes[0].legend(); axes[0].grid(alpha=.25)
    axes[1].plot(epochs, [r["macro_f1_val"] for r in history], label="macro-F1 val")
    axes[1].set(xlabel="Epoch", ylabel="Score", ylim=(0, 1)); axes[1].legend(); axes[1].grid(alpha=.25)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết. Trả về dict kết quả tóm tắt.

    TODO theo thứ tự:
      1. set_seed; tạo thư mục run_dir(cfg); ghi config.json (dataclasses.asdict(cfg))
      2. dataset.load_split + dataset.check_split (dừng nếu vi phạm S1-S6)
      3. dựng train/val loader (test loader chỉ tạo khi cfg.save_test_predictions)
      4. model.build_model, criterion (losses.build_criterion), optimizer, scheduler, scaler, EMA
      5. với mỗi epoch: train_one_epoch -> evaluate(val) -> ghi history (loss, macro-F1 val, lr...)
         và lưu checkpoint tốt nhất theo MACRO-F1 VAL (hòa thì lấy epoch sớm hơn)
      6. cuối: nạp checkpoint tốt nhất, lưu val logits và eval.save_predictions(pred_path(cfg, "val"), ...)
      7. NẾU cfg.save_test_predictions (chỉ ở Bước 4): đánh giá test đúng MỘT lần,
         lưu logits và eval.save_predictions(pred_path(cfg, "test"), ...)
      8. ghi history.csv, plot_curves(...), trả về dict tóm tắt
         (best_epoch, macro-F1 val, thời gian train mỗi epoch, số tham số, GMAC)
    Quy tắc: KHÔNG dùng test để chọn checkpoint hay bất kỳ quyết định nào (README.md, S4).
    """
    import torch
    import dataset as data_lib
    import model as model_lib
    import losses as loss_lib
    eval_lib = _load_eval_module()
    from sklearn.metrics import accuracy_score
    set_seed(cfg.seed)
    out = run_dir(cfg); out.mkdir(parents=True, exist_ok=True)
    (out/"config.json").write_text(json.dumps(__import__("dataclasses").asdict(cfg), indent=2), encoding="utf-8")
    train_df, val_df, test_df = data_lib.load_split(cfg.labels_dir, cfg.fold)
    split_stats = data_lib.check_split(train_df, val_df, test_df, cfg.images_dir)
    tr = data_lib.build_transforms(True, cfg.img_size, cfg.aug); ev = data_lib.build_transforms(False, cfg.img_size)
    train_loader = data_lib.make_loader(train_df, cfg.images_dir, tr, cfg.batch_size, True, cfg.sampler, cfg.num_workers, cfg.seed)
    val_loader = data_lib.make_loader(val_df, cfg.images_dir, ev, cfg.batch_size, False, num_workers=cfg.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model_lib.build_model(cfg.backbone, pretrained=cfg.init != "scratch", num_classes=9,
                                drop_rate=cfg.drop_rate, init=cfg.init).to(device)
    criterion_kw = {}
    if cfg.loss == "ls": criterion_kw["smoothing"] = cfg.label_smoothing or 0.1
    if cfg.loss == "focal": criterion_kw["gamma"] = cfg.focal_gamma
    if cfg.loss == "ce_weighted" or cfg.class_weight_beta is not None:
        counts = train_df.Label.value_counts().reindex(range(9), fill_value=0).to_numpy()
        criterion_kw["weight"] = loss_lib.class_weights(counts, cfg.class_weight_beta or 0.0).to(device)
    criterion = loss_lib.build_criterion(cfg.loss, **criterion_kw).to(device)
    optimizer = build_optimizer(net, cfg); scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    ema = EMA(net, cfg.ema_decay) if cfg.ema_decay else None
    best_f1, best_epoch, best_state, best_eval, durations = -1.0, None, None, None, []
    history = []
    for epoch in range(1, cfg.epochs+1):
        stats = train_one_epoch(net, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)
        eval_model = copy.deepcopy(net)
        if ema is not None: ema.copy_to(eval_model)
        names, y, logits, val_loss = evaluate(eval_model, val_loader, criterion, device)
        metrics = eval_lib.compute_metrics(y, logits.argmax(1), _softmax(logits))
        row = {"epoch": epoch, **stats, "val_loss": val_loss, "macro_f1_val": float(metrics["macro_f1"]),
               "top1_val": float(metrics["top1"])}
        history.append(row); durations.append(stats["epoch_seconds"])
        pd.DataFrame(history).to_csv(out/"history.csv", index=False)
        if metrics["macro_f1"] > best_f1:
            best_f1, best_epoch = float(metrics["macro_f1"]), epoch
            best_state = copy.deepcopy(eval_model.state_dict())
            best_eval = (names, y.copy(), logits.copy(), float(val_loss), metrics)
            torch.save({"model": best_state, "epoch": epoch, "config": __import__("dataclasses").asdict(cfg)}, out/"best.pt")
        del eval_model
    net.load_state_dict(best_state)
    names, y, logits, val_loss, metrics = best_eval
    np.savez_compressed(out/"val_logits.npz", filenames=np.asarray(names), y_true=y, logits=logits)
    eval_lib.save_predictions(pred_path(cfg, "val"), names, y, _softmax(logits))
    if cfg.save_test_predictions:
        # Only the explicitly requested final stage reaches this branch; no test metric affects selection.
        test_loader = data_lib.make_loader(test_df, cfg.images_dir, ev, cfg.batch_size, False, num_workers=cfg.num_workers)
        test_names, test_y, test_logits, _ = evaluate(net, test_loader, criterion, device)
        np.savez_compressed(out/"test_logits.npz", filenames=np.asarray(test_names), y_true=test_y, logits=test_logits)
        eval_lib.save_predictions(pred_path(cfg, "test"), test_names, test_y, _softmax(test_logits))
    plot_curves(history, Path("curves")/f"{cfg.exp_id}_seed{cfg.seed}.png", f"{cfg.exp_id} {cfg.backbone}")
    return {"exp_id": cfg.exp_id, "seed": cfg.seed, "best_epoch": best_epoch, "macro_f1_val": best_f1,
            "top1_val": float(metrics["top1"]), "mean_epoch_seconds": float(np.mean(durations)),
            "params_m": model_lib.count_params(net), "gmacs": model_lib.count_gmacs(net, cfg.img_size),
            "pretrained_tag": getattr(net, "pretrained_tag", "unknown"), "device": str(device), "split": split_stats}


def _softmax(logits):
    z = np.asarray(logits, dtype=np.float64); z -= z.max(axis=1, keepdims=True)
    p = np.exp(z); return p/p.sum(axis=1, keepdims=True)


def parse_overrides(pairs: list[str]) -> dict:
    """Biến ['seed=1', 'loss=focal', 'ema_decay=none'] thành dict, ép kiểu theo field của Config.

    TODO: tách key/value, báo lỗi rõ nếu key không có trong Config, ép int/float/bool/None theo kiểu field.
    """
    from dataclasses import fields
    defaults = {f.name: f.default for f in fields(Config)}
    optional_types = {"sampler": str, "mix": str, "class_weight_beta": float, "ema_decay": float}
    result = {}
    for pair in pairs:
        if "=" not in pair: raise ValueError(f"Cần KEY=VALUE: {pair}")
        key, raw = pair.split("=", 1)
        if key not in defaults: raise ValueError(f"Config không có trường {key!r}")
        default = defaults[key]
        if raw.lower() in {"none", "null"}: value = None
        elif key in optional_types:
            typ = optional_types[key]
            value = typ(raw)
        elif isinstance(default, bool):
            if raw.lower() not in {"true", "false", "1", "0"}: raise ValueError(f"Boolean không hợp lệ: {raw}")
            value = raw.lower() in {"true", "1"}
        elif isinstance(default, int): value = int(raw)
        elif isinstance(default, float): value = float(raw)
        else: value = raw
        result[key] = value
    return result


def main() -> None:
    """Điểm vào dòng lệnh: `python train.py --set exp_id=B01 backbone=resnet50 seed=0`.

    TODO: argparse nhận `--set KEY=VALUE ...`, dựng Config qua parse_overrides, gọi run(cfg), in kết quả.
    """
    parser = argparse.ArgumentParser(); parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    result = run(Config(**parse_overrides(args.set)))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
