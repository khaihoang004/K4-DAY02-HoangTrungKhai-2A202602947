"""Classification losses and batch-level Mixup/CutMix."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def build_criterion(kind="ce", **kw):
    if kind == "ce": return nn.CrossEntropyLoss(weight=kw.get("weight"))
    if kind == "ls": return LabelSmoothingCE(kw.get("smoothing", 0.1), weight=kw.get("weight"))
    if kind == "focal": return FocalLoss(kw.get("gamma", 2.0), alpha=kw.get("alpha"))
    if kind == "ce_weighted":
        if kw.get("weight") is None: raise ValueError("ce_weighted cần weight")
        return nn.CrossEntropyLoss(weight=kw["weight"])
    raise ValueError(f"loss không hỗ trợ: {kind}")


class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        if not 0 <= smoothing < 1: raise ValueError("smoothing phải thuộc [0,1)")
        self.smoothing = float(smoothing)
        self.register_buffer("weight", None if weight is None else torch.as_tensor(weight, dtype=torch.float32))

    def forward(self, logits, target):
        return F.cross_entropy(logits, target, weight=self.weight, label_smoothing=self.smoothing)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        if gamma < 0: raise ValueError("gamma phải không âm")
        self.gamma = float(gamma)
        self.register_buffer("alpha", None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        logpt = logp.gather(1, target.long().view(-1, 1)).squeeze(1)
        loss = -((1.0 - logpt.exp()).clamp_min(0).pow(self.gamma)) * logpt
        if self.alpha is not None: loss = loss * self.alpha.to(logits.device)[target.long()]
        return loss.mean()


def class_weights(counts, beta=0.0):
    c = torch.as_tensor(counts, dtype=torch.float64)
    if c.ndim != 1 or torch.any(c <= 0): raise ValueError("counts phải là vector đếm dương của mọi lớp")
    if beta == 0: w = c.reciprocal()
    elif 0 < beta < 1: w = (1.0 - beta) / (-torch.expm1(c * np.log(beta)))
    else: raise ValueError("beta phải bằng 0 hoặc thuộc (0,1)")
    return (w / w.mean()).to(torch.float32)


def mix_batch(x, y, alpha=1.0, mode="cutmix"):
    if alpha <= 0: raise ValueError("alpha phải dương")
    if mode not in ("mixup", "cutmix"): raise ValueError("mode phải là mixup hoặc cutmix")
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    ya, yb = y, y[perm]
    if mode == "mixup":
        mixed = lam * x + (1.0 - lam) * x[perm]
    else:
        _, _, h, w = x.shape
        ratio = np.sqrt(1.0 - lam)
        cut_h, cut_w = int(h * ratio), int(w * ratio)
        cy, cx = np.random.randint(h), np.random.randint(w)
        y1, y2 = max(cy-cut_h//2, 0), min(cy+(cut_h+1)//2, h)
        x1, x2 = max(cx-cut_w//2, 0), min(cx+(cut_w+1)//2, w)
        mixed = x.clone()
        mixed[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam = 1.0 - ((y2-y1)*(x2-x1) / float(h*w))
    return mixed, (ya, yb, lam)


def mixed_loss(criterion, logits, targets):
    ya, yb, lam = targets
    return lam * criterion(logits, ya) + (1.0 - lam) * criterion(logits, yb)
