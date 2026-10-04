"""Synchronized inference latency measurements."""
from __future__ import annotations
import time
import numpy as np


def bench(fn, warmup=10, iters=100, sync=None):
    if warmup < 10 or iters < 50: raise ValueError("cần warmup >= 10 và iters >= 50")
    for _ in range(warmup): fn()
    values = []
    for _ in range(iters):
        if sync: sync()
        t0 = time.perf_counter(); fn()
        if sync: sync()
        values.append((time.perf_counter()-t0)*1000)
    return {"p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)), "mean": float(np.mean(values)), "n": int(iters)}


def latency_report(model, batch_size, img_size, dtype="fp32", device="cuda", warmup=10, iters=100):
    import torch
    if dtype not in {"fp32", "amp", "fp16"}: raise ValueError("dtype phải là fp32, amp, fp16")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA không khả dụng")
    model = model.to(dev).eval()
    if dtype == "fp16": model = model.half()
    x = torch.randn(batch_size, 3, img_size, img_size, device=dev, dtype=torch.float16 if dtype == "fp16" else torch.float32)
    def forward():
        with torch.inference_mode():
            if dtype == "amp":
                with torch.autocast(device_type=dev.type, dtype=torch.float16 if dev.type == "cuda" else torch.bfloat16): model(x)
            else: model(x)
    report = bench(forward, warmup, iters, torch.cuda.synchronize if dev.type == "cuda" else None)
    report.update({"gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU", "dtype": dtype,
                   "batch": batch_size, "img_size": img_size, "images_per_s": batch_size/(report["p50"]/1000), "torch": torch.__version__})
    return report


def tta_latency(model, k_views, **kw):
    import torch
    if int(k_views) < 1: raise ValueError("k_views phải >= 1")
    class RepeatedViews(torch.nn.Module):
        def __init__(self, wrapped, count):
            super().__init__(); self.wrapped = wrapped; self.count = count
        def forward(self, x):
            result = None
            for _ in range(self.count): result = self.wrapped(x)
            return result
    result = latency_report(RepeatedViews(model, int(k_views)), **kw)
    result["views"] = int(k_views)
    return result
