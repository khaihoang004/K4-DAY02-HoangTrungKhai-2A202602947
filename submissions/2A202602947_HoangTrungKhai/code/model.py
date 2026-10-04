"""timm model construction and model size helpers."""
from __future__ import annotations

SUGGESTED_BACKBONES = {"resnet50": "resnet50", "resnext50": "resnext50_32x4d", "convnext_tiny": "convnext_tiny",
                       "deit_small": "deit_small_patch16_224", "swin_tiny": "swin_tiny_patch4_window7_224",
                       "efficientnet_b0": "efficientnet_b0", "mobilenetv3": "mobilenetv3_large_100"}


def build_model(name, pretrained=True, num_classes=9, drop_rate=0.0, init="finetune"):
    import timm
    if init not in {"scratch", "frozen", "finetune"}: raise ValueError(f"init không hợp lệ: {init}")
    use_pretrained = bool(pretrained and init != "scratch")
    model = timm.create_model(name, pretrained=use_pretrained, num_classes=num_classes, drop_rate=drop_rate)
    cfg = getattr(model, "pretrained_cfg", {}) or {}
    model.pretrained_tag = cfg.get("tag") or cfg.get("hf_hub_id") or ("imagenet" if use_pretrained else "random-init")
    if init == "frozen": freeze_backbone(model)
    return model


def _head_parameters(model):
    if not hasattr(model, "get_classifier"): return set()
    classifier = model.get_classifier()
    if classifier is None: return set()
    return {id(p) for p in classifier.parameters()}


def freeze_backbone(model):
    head = _head_parameters(model)
    for p in model.parameters(): p.requires_grad_(id(p) in head)
    model._freeze_backbone = True
    return None


def param_groups(model, lr_backbone, lr_head, weight_decay):
    head_ids = _head_parameters(model)
    groups = {}
    for p in model.parameters():
        if not p.requires_grad: continue
        is_head = id(p) in head_ids
        wd = float(weight_decay) if is_head or p.ndim > 1 else 0.0
        lr = float(lr_head if is_head else lr_backbone)
        groups.setdefault((lr, wd), []).append(p)
    return [{"params": ps, "lr": lr, "weight_decay": wd} for (lr, wd), ps in groups.items()]


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model, img_size=224):
    try:
        from fvcore.nn import FlopCountAnalysis
        import torch
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()
        with torch.inference_mode():
            value = FlopCountAnalysis(model, torch.zeros(1, 3, img_size, img_size, device=device)).total()
        model.train(was_training)
        return float(value) / 2e9  # fvcore counts multiply and add separately; report MACs.
    except ImportError:
        # Hook-based MAC estimate for Conv/Linear; attention matmuls may be omitted.
        import torch
        macs = [0]
        hooks = []
        def conv_hook(module, inputs, output):
            out = output
            kh, kw = module.kernel_size
            macs[0] += out.numel() * (module.in_channels // module.groups) * kh * kw
        def linear_hook(module, inputs, output): macs[0] += output.numel() * module.in_features
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d): hooks.append(m.register_forward_hook(conv_hook))
            elif isinstance(m, torch.nn.Linear): hooks.append(m.register_forward_hook(linear_hook))
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()
        try:
            with torch.inference_mode(): model(torch.zeros(1, 3, img_size, img_size, device=device))
        finally:
            for h in hooks: h.remove()
            model.train(was_training)
        return macs[0] / 1e9
