"""Checks that the explanations are right, not just pretty.

1. Completeness   B-cos contributions must sum to the explained logit difference.
2. Localisation   On loss patches, what share of the positive evidence falls on
                  pixels Hansen marks as lost? Compared to chance (the share of
                  the patch that was lost) and to Grad-CAM on the black-box CNN.
3. Randomisation  Adebayo et al. (2018): explanations from a trained model must
                  differ from those of the same architecture with random weights.
4. Band reliance  Share of |contribution| per band, per region (B-cos only), and
                  test-time band ablation (replace one band by its training mean).
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Subset

from . import config as C
from .data.dataset import PatchDataset
from .metrics import point_metrics
from .models.zoo import build_model, gradcam, is_bcos
from .train import _prep, device, predict


def _sum_pool_to(m: torch.Tensor, size: int) -> torch.Tensor:
    """Sum-pool (B,H,W) maps down to size x size; sums are preserved exactly."""
    if m.shape[-1] == size:
        return m
    k = m.shape[-1] // size
    return F.avg_pool2d(m[:, None], k)[:, 0] * k * k


def maps_for(model, batch, resize, uses_ndvi, n_img: int | None = None) -> Dict[str, torch.Tensor]:
    """Contribution (B-cos) or Grad-CAM map per patch.

    `n_img` = number of imagery channels; history channels (past-loss mask and the
    constant density channel) are left out of the spatial map so it shows what the
    *imagery* contributes. Their share is still reported in band_share.
    """
    dev = device()
    x, nd = _prep(batch, resize, dev)
    size = batch["x"].shape[-1]
    if is_bcos(model):
        r = model.contributions(x, nd if uses_ndvi else None)
        img = r["per_band"][:, :n_img].sum(1) if n_img else r["pixels"]
        out = dict(map=_sum_pool_to(img, size).cpu(),
                   per_band=torch.stack([_sum_pool_to(r["per_band"][:, c], size)
                                         for c in range(r["per_band"].shape[1])], 1).cpu(),
                   completeness=r["completeness_error"].cpu(), logit=r["logit_diff"].cpu())
        if "ndvi" in r:
            out["ndvi_total"] = r["ndvi"].flatten(1).sum(1).cpu()
        return out
    cam = gradcam(model, x)
    return dict(map=F.interpolate(cam[:, None], size=(size, size), mode="bilinear",
                                  align_corners=False)[:, 0].cpu())


def evaluate_explanations(result: Dict, variant: str, out_dir: str, max_patches: int = 256,
                          download: bool = False) -> Dict:
    model, resize, uses_ndvi = result["model"], result["resize"], result["uses_ndvi"]
    ds_te: PatchDataset = result["datasets"][2]
    y = ds_te.labels()
    pos = np.where(y == 1)[0][:max_patches]
    if len(pos) == 0:
        return {}
    dl = DataLoader(Subset(ds_te, pos), batch_size=32, shuffle=False)

    rand = build_model(variant, ds_te.n_channels, download=False).to(device())
    loc, chance, rho, comp, band_share, ndvi_tot = [], [], [], [], [], []
    for b in dl:
        m = maps_for(model, b, resize, uses_ndvi, _n_img(ds_te))
        mr = maps_for(rand, b, resize, uses_ndvi, _n_img(ds_te))
        mask = b["mask"]
        posmap = m["map"].clamp(min=0)
        tot = posmap.flatten(1).sum(1)
        ok = tot > 0
        loc += ((posmap * mask).flatten(1).sum(1)[ok] / tot[ok]).tolist()
        chance += mask.flatten(1).mean(1)[ok].tolist()
        for a, r_ in zip(m["map"].abs().flatten(1).numpy(), mr["map"].abs().flatten(1).numpy()):
            if a.std() > 0 and r_.std() > 0:          # undefined for a constant (all-zero) map
                rho.append(spearmanr(a, r_).correlation)
        if "completeness" in m:
            comp += m["completeness"].tolist()
            ab = m["per_band"].abs().flatten(2).sum(2)
            band_share += (ab / ab.sum(1, keepdim=True)).tolist()
        if "ndvi_total" in m:
            ndvi_tot += (m["ndvi_total"] / m["logit"].abs().clamp(min=1e-6)).tolist()

    mean = lambda v: float(np.nanmean(v)) if len(v) else float("nan")
    res = dict(
        variant=variant, n_patches=int(len(loc)),
        localisation=mean(loc), chance=mean(chance),
        lift_over_chance=mean(loc) / max(1e-9, mean(chance)) if len(loc) else float("nan"),
        spearman_vs_random_model=mean(rho),
    )
    if comp:
        res["max_completeness_error"] = float(np.max(comp))
        res["band_share"] = dict(zip(_channel_names(ds_te), np.mean(band_share, 0).round(4).tolist()))
    if ndvi_tot:
        res["ndvi_share_of_logit"] = float(np.mean(ndvi_tot))
    with open(os.path.join(out_dir, "explanations.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    return res


def _n_img(ds: PatchDataset) -> int:
    return ds.n_channels - (2 if ds.history else 0)


def _channel_names(ds: PatchDataset) -> List[str]:
    b = [C.BAND_NAMES[i] for i in ds.band_ix]
    names = [f"{n}_t0" for n in b] if ds.task == "forecast" else [f"{n}_t0" for n in b] + [f"{n}_t1" for n in b]
    if ds.history:
        names += ["past_loss_mask", "past_loss_density"]
    return names


def band_ablation(result: Dict, out_dir: str) -> pd.DataFrame:
    """Test-time ablation: replace one channel by its training mean; report AP drop per region."""
    model, resize, uses_ndvi = result["model"], result["resize"], result["uses_ndvi"]
    ds_tr, _, ds_te = result["datasets"]
    names = _channel_names(ds_te)
    sample = np.random.default_rng(0).choice(len(ds_tr), min(512, len(ds_tr)), replace=False)
    means = torch.stack([ds_tr[i]["x"] for i in sample]).mean(dim=(0, 2, 3)).numpy()
    regions = ds_te.meta.loc[ds_te.idx, "region"].to_numpy()
    y = ds_te.labels()

    def ap_by_region(ds):
        p = predict(model, DataLoader(ds, batch_size=128), resize, uses_ndvi)
        out = {"all": point_metrics(y, p, 0.5)["ap"]}
        for r in np.unique(regions):
            k = regions == r
            out[r] = point_metrics(y[k], p[k], 0.5)["ap"] if len(np.unique(y[k])) > 1 else float("nan")
        return out

    base = ap_by_region(ds_te)
    rows = []
    for ch, name in enumerate(names):
        abl = PatchDataset(ds_te.meta, ds_te.arrays, ds_te.idx, task=ds_te.task,
                           bands=[C.BAND_NAMES[i] for i in ds_te.band_ix], history=ds_te.history,
                           with_ndvi=ds_te.with_ndvi, band_fill={ch: float(means[ch])})
        ap = ap_by_region(abl)
        for r in base:
            rows.append(dict(channel=name, region=r, ap_full=base[r], ap_ablated=ap[r],
                             ap_drop=base[r] - ap[r]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "band_ablation.csv"), index=False)
    return df


def gallery(result: Dict, out_png: str, n: int = 6):
    """RGB | Hansen loss | contribution map for the n most confident loss patches."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    model, resize, uses_ndvi = result["model"], result["resize"], result["uses_ndvi"]
    ds_te = result["datasets"][2]
    preds = result["preds"]
    te = preds[preds.split == "test"].reset_index(drop=True)
    order = te[te.y == 1].sort_values("p", ascending=False).index[:n]
    if len(order) == 0:
        return
    b = next(iter(DataLoader(Subset(ds_te, list(order)), batch_size=n)))
    m = maps_for(model, b, resize, uses_ndvi, _n_img(ds_te))["map"]
    fig, ax = plt.subplots(len(order), 3, figsize=(7.5, 2.5 * len(order)))
    ax = np.atleast_2d(ax)
    t0 = ds_te.arrays["t0"]
    for i, k in enumerate(order):
        j = int(te.loc[k, "idx"])
        rgb = np.clip(t0[j][[0, 1, 2]].astype(np.float32).transpose(1, 2, 0) * 2.5, 0, 1)
        lim = float(m[i].abs().max()) or 1.0
        ax[i, 0].imshow(rgb); ax[i, 0].set_title(f"{te.loc[k, 'region']}  p={te.loc[k, 'p']:.2f}", fontsize=8)
        ax[i, 1].imshow(b["mask"][i], cmap="Reds", vmin=0, vmax=1); ax[i, 1].set_title("Hansen loss", fontsize=8)
        ax[i, 2].imshow(m[i], cmap="bwr", vmin=-lim, vmax=lim); ax[i, 2].set_title("imagery evidence (red = for loss)", fontsize=8)
        for a in ax[i]:
            a.axis("off")
    plt.tight_layout(); plt.savefig(out_png, dpi=120); plt.close(fig)
