"""Validation-only TTA/calibration comparison and synchronized CUDA latency.

Usage from starter/: python inference_eval.py --checkpoint runs/F01/seed0/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import dataset
import model as model_lib
from inference import apply_temperature, fit_temperature
from train import _load_eval_module, _softmax


def timed(fn, device, warmup=10, iters=50):
    for _ in range(warmup): fn()
    values=[]
    for _ in range(iters):
        if device.type == "cuda": torch.cuda.synchronize(device)
        start=time.perf_counter(); fn()
        if device.type == "cuda": torch.cuda.synchronize(device)
        values.append((time.perf_counter()-start)*1000)
    return {f"p{q}_ms": float(np.percentile(values,q)) for q in (50,95,99)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",default="auto",help="auto selects highest validation macro-F1 run in training_summary.csv")
    ap.add_argument("--summary",default="runs/training_summary.csv")
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--config",default=None,help="checkpoint config.json; inferred from checkpoint if omitted")
    ap.add_argument("--images-dir",default="../data/images")
    ap.add_argument("--labels-dir",default="../data/labels")
    ap.add_argument("--out-dir",default="runs")
    ap.add_argument("--pred-dir",default="predictions")
    ap.add_argument("--batch-size",type=int,default=64)
    ap.add_argument("--workers",type=int,default=2)
    args=ap.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint=args.checkpoint
    if checkpoint=="auto":
        summary=pd.read_csv(args.summary)
        best=summary.sort_values("macro_f1_val",ascending=False).iloc[0]
        checkpoint=str(Path(args.out_dir)/str(best.exp_id)/f"seed{args.seed}"/"best.pt")
        print(f"Validation-selected checkpoint: {checkpoint}")
    blob=torch.load(checkpoint,map_location="cpu",weights_only=False)
    cfg=blob.get("config",{})
    backbone=cfg.get("backbone","convnext_tiny"); size=int(cfg.get("img_size",224))
    net=model_lib.build_model(backbone,pretrained=False,num_classes=9,drop_rate=0.0,init="scratch")
    net.load_state_dict(blob["model"]); net.to(device).eval()
    _,val_df,_=dataset.load_split(args.labels_dir,0)
    ds=dataset.make_loader(val_df,args.images_dir,dataset.build_transforms(False,size),args.batch_size,False,num_workers=args.workers)
    names=[]; labels=[]; logits=[]; flip_logits=[]; scales_logits=[]
    with torch.inference_mode():
        for x,y,fnames in ds:
            x=x.to(device,non_blocking=True)
            z=net(x).float().cpu().numpy(); zf=net(torch.flip(x,(-1,))).float().cpu().numpy()
            zs=[]
            for side in (192,256):
                xs=F.interpolate(x,size=(side,side),mode="bilinear",align_corners=False,antialias=True)
                zs.append(net(xs).float().cpu().numpy())
            names.extend(fnames); labels.append(y.numpy()); logits.append(z); flip_logits.append(zf)
            scales_logits.append(np.stack([z,*zs],axis=0))
    y=np.concatenate(labels); z=np.concatenate(logits); zf=np.concatenate(flip_logits)
    # mean-probability aggregation for views; full-resolution resize scales are included.
    p0=_softmax(z)
    pflip=(_softmax(z)+_softmax(zf))/2
    all_scale=np.concatenate(scales_logits,axis=1) # [view, N, class]
    pscale=np.mean([_softmax(v) for v in all_scale],axis=0)
    temperature=fit_temperature(z,y)
    ptemp=apply_temperature(z,temperature)
    methods=[("I00_1view",p0),("I01_hflip_prob",pflip),
             ("I02_multiscale_prob",pscale),("I03_temperature",ptemp)]
    ev=_load_eval_module(); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    pred=Path(args.pred_dir); pred.mkdir(parents=True,exist_ok=True)
    rows=[]
    for exp,probs in methods:
        m=ev.compute_metrics(y,probs.argmax(1),probs)
        ev.save_predictions(pred/f"{exp}_seed0_val.csv",names,y,probs)
        rows.append({"exp_id":exp,"method":exp,"macro_f1_val":m["macro_f1"],"top1_val":m["top1"],
                     "ece_val":m["ece"],"temperature":temperature if exp=="I03_temperature" else None})
    # Per-image batch=1 latency of the actual method forwards; includes CUDA synchronization.
    x=torch.randn(1,3,size,size,device=device)
    def one():
        with torch.inference_mode(): net(x)
    def flip():
        with torch.inference_mode():
            a=net(x); b=net(torch.flip(x,(-1,))); (torch.softmax(a,1)+torch.softmax(b,1)).mean()
    def multi():
        with torch.inference_mode():
            for side in (192,size,256): net(F.interpolate(x,size=(side,side),mode="bilinear",align_corners=False,antialias=True))
    lat1=timed(one,device)
    latencies={"I00_1view":lat1,"I01_hflip_prob":timed(flip,device),
               "I02_multiscale_prob":timed(multi,device),"I03_temperature":lat1}
    for row in rows:
        row.update(latencies.get(row["exp_id"],{})); row["device"]=torch.cuda.get_device_name(device) if device.type=="cuda" else "CPU"
    pd.DataFrame(rows).to_csv(out/"inference_summary.csv",index=False)
    (out/"inference_metadata.json").write_text(json.dumps({"checkpoint":checkpoint,"backbone":backbone,
        "temperature_fit_on_validation":temperature,"device":str(device)},indent=2))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=="__main__": main()
