"""Validation-time inference transforms, calibration, and Conv/BN fusion."""
from __future__ import annotations

import numpy as np


def predict_logits(model, loader, device, view=None):
    import torch
    model.eval(); names, labels, outputs = [], [], []
    with torch.inference_mode():
        for x, y, filenames in loader:
            x = x.to(device, non_blocking=True)
            if view is not None: x = view(x)
            outputs.append(model(x).float().cpu().numpy())
            labels.append(y.numpy()); names.extend(list(filenames))
    return names, np.concatenate(labels), np.concatenate(outputs)


def view_identity(x): return x
def view_hflip(x):
    import torch
    return torch.flip(x, dims=(-1,))


def views_multicrop(x, crop):
    _, _, h, w = x.shape
    if crop > h or crop > w: raise ValueError("crop lớn hơn ảnh đầu vào")
    coords = [(0, 0), (0, w-crop), (h-crop, 0), (h-crop, w-crop), ((h-crop)//2, (w-crop)//2)]
    return [x[:, :, y:y+crop, z:z+crop] for y, z in coords]


def views_multiscale(x, sizes):
    import torch.nn.functional as F
    return [F.interpolate(x, size=(int(s), int(s)), mode="bilinear", align_corners=False, antialias=True) for s in sizes]


def aggregate_views(logits_per_view, space="prob"):
    z = np.asarray(logits_per_view)
    if z.ndim != 3 or z.shape[0] < 1: raise ValueError("cần logits dạng (views, N, classes)")
    if space == "prob":
        z = z - z.max(axis=2, keepdims=True); p = np.exp(z); p /= p.sum(axis=2, keepdims=True)
        probs = p.mean(axis=0)
    elif space == "logit":
        z = z.mean(axis=0); z -= z.max(axis=1, keepdims=True); probs = np.exp(z); probs /= probs.sum(axis=1, keepdims=True)
    else: raise ValueError("space phải là prob hoặc logit")
    return probs / probs.sum(axis=1, keepdims=True)


def ensemble_probs(list_of_probs):
    p = np.asarray(list_of_probs, dtype=np.float64)
    if p.ndim != 3 or p.shape[0] < 1: raise ValueError("cần xác suất dạng (models, N, classes)")
    if not np.isfinite(p).all() or (p < 0).any(): raise ValueError("xác suất không hợp lệ")
    out = p.mean(axis=0)
    return out / out.sum(axis=1, keepdims=True)


def fit_temperature(val_logits, val_labels):
    import torch
    import torch.nn.functional as F
    logits = torch.as_tensor(val_logits, dtype=torch.float64)
    labels = torch.as_tensor(val_labels, dtype=torch.long)
    if logits.ndim != 2 or logits.shape[0] != labels.numel(): raise ValueError("val logits/labels không khớp")
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optim = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")
    def closure():
        optim.zero_grad(); loss = F.cross_entropy(logits / log_t.exp().clamp(1e-3, 1e3), labels); loss.backward(); return loss
    optim.step(closure)
    return float(log_t.detach().exp().clamp(1e-3, 1e3))


def apply_temperature(logits, T):
    if T <= 0: raise ValueError("T phải dương")
    z = np.asarray(logits, dtype=np.float64) / T
    z -= z.max(axis=1, keepdims=True); p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def fuse_conv_bn(model):
    import torch
    import torch.nn as nn
    import torch.nn.utils.parametrize as _parametrize
    del _parametrize
    model.eval()
    def recurse(parent):
        children = list(parent.named_children())
        for idx, (name, child) in enumerate(children):
            if isinstance(child, nn.BatchNorm2d) and idx > 0:
                prev_name, prev = children[idx-1]
                if isinstance(prev, nn.Conv2d):
                    fused = torch.nn.utils.fusion.fuse_conv_bn_eval(prev, child)
                    setattr(parent, prev_name, fused); setattr(parent, name, nn.Identity())
            else: recurse(child)
    recurse(model)
    return model
