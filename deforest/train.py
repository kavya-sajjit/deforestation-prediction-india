"""Train one model on one split and save predictions, curves and weights."""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import config as C
from .data.dataset import PatchDataset
from .metrics import best_f1_threshold, point_metrics
from .models.zoo import PRETRAINED, build_model, is_bcos


def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _prep(batch, resize: int, dev):
    x, nd = batch["x"].to(dev).float(), batch["ndvi"].to(dev).float()
    if resize and x.shape[-1] != resize:
        x = F.interpolate(x, size=(resize, resize), mode="bilinear", align_corners=False)
        if nd.shape[-1] > 1:
            nd = F.interpolate(nd, size=(resize, resize), mode="bilinear", align_corners=False)
    return x, nd


def _forward(model, x, nd, uses_ndvi):
    return model(x, nd) if uses_ndvi else model(x)


@torch.no_grad()
def predict(model, loader, resize, uses_ndvi) -> np.ndarray:
    dev = device(); model.eval(); out = []
    for b in loader:
        x, nd = _prep(b, resize, dev)
        out.append(torch.softmax(_forward(model, x, nd, uses_ndvi), 1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def run(variant: str, task: str, bands: List[str], history: bool, meta: pd.DataFrame,
        arrays: dict, train_idx, val_idx, test_idx, out_dir: str,
        cfg: C.TrainConfig = C.TrainConfig(), download: bool = True, tag: str = "") -> Dict:
    set_seed(cfg.seed)
    os.makedirs(out_dir, exist_ok=True)
    dev = device()
    uses_ndvi = variant == "bcos_pre_ndvi"
    resize = cfg.resize or (128 if variant in PRETRAINED else 0)
    kw = dict(task=task, bands=bands, history=history, with_ndvi=uses_ndvi)
    ds_tr = PatchDataset(meta, arrays, train_idx, augment=True, **kw)
    ds_va = PatchDataset(meta, arrays, val_idx, **kw)
    ds_te = PatchDataset(meta, arrays, test_idx, **kw)
    mk = lambda d, sh: DataLoader(d, batch_size=cfg.batch_size, shuffle=sh, num_workers=cfg.num_workers,
                                  pin_memory=dev.type == "cuda", drop_last=False)
    dl_tr, dl_va, dl_te = mk(ds_tr, True), mk(ds_va, False), mk(ds_te, False)

    model = build_model(variant, ds_tr.n_channels, download=download).to(dev)
    lr = cfg.lr_pretrained if variant in PRETRAINED else cfg.lr_scratch
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    y_tr = ds_tr.labels()
    pos_w = float(np.clip((y_tr == 0).sum() / max(1, (y_tr == 1).sum()), 1.0, 20.0))
    crit = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_w], device=dev))

    y_va = ds_va.labels()
    best, best_ap, bad, hist = None, -1.0, 0, []
    for ep in range(cfg.epochs):
        model.train(); t0 = time.time(); losses = []
        for b in dl_tr:
            x, nd = _prep(b, resize, dev)
            loss = crit(_forward(model, x, nd, uses_ndvi), b["y"].to(dev))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            losses.append(loss.item())
        sched.step()
        p_va = predict(model, dl_va, resize, uses_ndvi)
        ap = point_metrics(y_va, p_va, 0.5)["ap"] if len(np.unique(y_va)) > 1 else -float(np.mean(losses))
        hist.append(dict(epoch=ep, train_loss=float(np.mean(losses)), val_ap=ap, sec=time.time() - t0))
        print(f"  [{tag}] ep {ep:02d} loss {np.mean(losses):.4f} val AP {ap:.4f}")
        if ap > best_ap + 1e-4:
            best_ap, bad = ap, 0
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best)

    p_va = predict(model, dl_va, resize, uses_ndvi)
    thr = best_f1_threshold(y_va, p_va)
    p_te = predict(model, dl_te, resize, uses_ndvi)
    rows = []
    for split, idx, p in (("val", val_idx, p_va), ("test", test_idx, p_te)):
        m = meta.loc[idx]
        rows.append(pd.DataFrame(dict(idx=idx, region=m.region.values, block_r=m.block_r.values,
                                      block_c=m.block_c.values, y=m[ds_tr.label_col].values, p=p,
                                      split=split, threshold=thr)))
    preds = pd.concat(rows, ignore_index=True)
    preds.to_csv(os.path.join(out_dir, "preds.csv"), index=False)
    pd.DataFrame(hist).to_csv(os.path.join(out_dir, "history.csv"), index=False)
    te = preds[preds.split == "test"]
    res = point_metrics(te.y.to_numpy(), te.p.to_numpy(), thr)
    res.update(variant=variant, task=task, bands="+".join(bands), history=history,
               n_train=len(train_idx), n_val=len(val_idx), epochs_run=len(hist), pos_weight=pos_w)
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    torch.save(dict(state_dict=model.state_dict(), variant=variant, bands=bands, history=history,
                    task=task, n_channels=ds_tr.n_channels, resize=resize, threshold=thr),
               os.path.join(out_dir, "model.pt"))
    return dict(model=model, preds=preds, metrics=res, resize=resize, uses_ndvi=uses_ndvi,
                datasets=(ds_tr, ds_va, ds_te))
