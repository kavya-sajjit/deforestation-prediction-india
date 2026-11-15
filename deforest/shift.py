"""Do a model's reasons survive a move to a region it has never seen?

Accuracy under distribution shift is measured all the time. This module measures
what happens to the *explanations*. Every model is already trained twice:

  * "block" split: tested on held-out 5 km blocks of regions it trained on
                   (in-distribution, ID)
  * "loro"  split: tested on a whole region it never saw (out-of-distribution, OOD)

Every patch of a region is therefore scored twice, once by a model that knew the
region and once by a model that did not. For the same patches we compare

  1. skill          average precision (AP) minus its chance level
  2. localisation   share of positive evidence on pixels that were later lost,
                    divided by chance (the share of the patch that was lost)
  3. deletion gap   remove the pixels the explanation ranks highest and see how
                    fast P(loss) falls, compared with removing random pixels.
                    This is a causal check: it asks whether the model really
                    uses the pixels its explanation points at.
  4. band reliance  share of |contribution| per input channel (B-cos only). The
                    Jensen-Shannon distance between the ID and OOD reliance
                    vectors needs NO labels, so it could be computed on a new
                    region before any ground truth exists.

The question for the paper: do explanations degrade faster than accuracy, and
does the label-free reliance shift (4) track the accuracy drop (1)?

    python -m deforest.shift results                          # writes results/SHIFT.md, shift.csv, shift_<task>.png
    python -m deforest.shift results --data data/v2 --backfill   # first add per-patch records to runs trained
                                                                 # before this module existed (uses model.pt)

`per_patch()` is called by deforest.experiments for every neural run; `report()`
needs both the block and the loro runs of a model to exist.
"""
from __future__ import annotations

import glob
import os
from typing import Callable, Dict, List, Sequence

import numpy as np
import pandas as pd

FRACS = (0.05, 0.10, 0.20, 0.30, 0.50)     # share of pixels removed along the deletion curve
# A "share kept" is a ratio, and a ratio of two near-chance numbers is noise. It is reported only
# when the in-distribution value is clearly above chance and rests on enough loss patches.
MIN_SKILL = 0.05                           # AP must exceed chance by this much (and by half of chance)
MIN_LIFT = 1.5                             # localisation must be at least 1.5x chance
MIN_DEL = 0.05                             # deletion gap must be at least this large
MIN_POS = 30                               # loss patches needed before any ratio is shown


# --------------------------------------------------------------------------- deletion (torch)
def deletion_gap(p_loss: Callable, x, maps, fill, n_img: int, fracs: Sequence[float] = FRACS,
                 seed: int = 0, ndvi=None, ndvi_fill: float = 0.5) -> Dict[str, np.ndarray]:
    """Deletion curves for a batch.

    p_loss(x, ndvi) -> P(loss) per patch, shape (B,)
    x      (B, C, H, W) inputs; the first n_img channels are imagery
    maps   (B, H, W) attribution maps (higher = more evidence for loss)
    fill   (C,) value that replaces a removed pixel in each channel (training mean)

    Pixels are removed in order of attribution ("attr") and in random order
    ("rand"). Returns P(loss) after each step for both, and per patch

        gap = mean over steps of [ P_rand - P_attr ]

    gap > 0 means removing the pixels the explanation points at lowers P(loss)
    faster than removing the same number of random pixels. An explanation
    unrelated to the model's behaviour has an expected gap of 0.
    """
    import torch
    B, _, H, W = x.shape
    g = torch.Generator().manual_seed(seed)
    order = dict(attr=torch.argsort(maps.reshape(B, -1), dim=1, descending=True),
                 rand=torch.stack([torch.randperm(H * W, generator=g) for _ in range(B)]))
    fill_t = torch.as_tensor(fill, dtype=x.dtype)[:n_img].view(1, n_img, 1, 1)
    with_nd = ndvi is not None and ndvi.shape[-1] == W
    p0 = p_loss(x, ndvi).detach().cpu().numpy()
    curves = {k: [] for k in order}
    for k, o in order.items():
        for f in fracs:
            n = max(1, int(round(f * H * W)))
            gone = torch.zeros(B, H * W, dtype=torch.bool)
            gone.scatter_(1, o[:, :n], True)
            gone = gone.view(B, 1, H, W)
            xm = x.clone()
            xm[:, :n_img] = torch.where(gone, fill_t.expand(B, -1, H, W), x[:, :n_img])
            nm = torch.where(gone, torch.full_like(ndvi, ndvi_fill), ndvi) if with_nd else ndvi
            curves[k].append(p_loss(xm, nm).detach().cpu().numpy())
    attr, rand = np.stack(curves["attr"], 1), np.stack(curves["rand"], 1)
    return dict(p0=p0, attr=attr, rand=rand, gap=(rand - attr).mean(1))


def _train_fill(ds_tr, n: int = 512):
    """Per-channel mean of a sample of training patches (and mean NDVI)."""
    import torch
    pick = np.random.default_rng(0).choice(len(ds_tr), min(n, len(ds_tr)), replace=False)
    items = [ds_tr[int(i)] for i in pick]
    fill = torch.stack([it["x"] for it in items]).float().mean(dim=(0, 2, 3))
    nd = float(torch.stack([it["ndvi"] for it in items]).float().mean()) if ds_tr.with_ndvi else 0.5
    return fill, nd


def per_patch(result: Dict, variant: str, out_dir: str, max_patches: int = 256) -> pd.DataFrame:
    """One row per sampled test patch: localisation, deletion gap, band reliance.

    Loss patches (up to max_patches) get all three measures. The same number of
    no-loss patches is added for band reliance only, because that measure is meant
    to work without labels.
    """
    import torch
    from torch.utils.data import DataLoader, Subset

    from .explain import _channel_names, _n_img, maps_for
    from .train import _forward, _prep, device

    model, resize, uses_ndvi = result["model"], result["resize"], result["uses_ndvi"]
    ds_tr, _, ds_te = result["datasets"]
    y = ds_te.labels()
    pos = np.where(y == 1)[0][:max_patches]
    neg = np.where(y == 0)[0]
    neg = np.random.default_rng(0).choice(neg, min(len(neg), max(len(pos), 32)), replace=False)
    take = np.sort(np.concatenate([pos, neg])).astype(int)
    if len(take) == 0:
        return pd.DataFrame()
    fill, nd_fill = _train_fill(ds_tr)
    n_img, names, dev = _n_img(ds_te), _channel_names(ds_te), device()
    p_of = dict(zip(result["preds"].query("split == 'test'").idx, result["preds"].query("split == 'test'").p))

    @torch.no_grad()
    def p_loss(x, nd):
        xb, nb = _prep(dict(x=x, ndvi=nd), resize, dev)
        return torch.softmax(_forward(model, xb, nb, uses_ndvi), 1)[:, 1]

    model.eval()
    rows: List[dict] = []
    for b in DataLoader(Subset(ds_te, take.tolist()), batch_size=32, shuffle=False):
        m = maps_for(model, b, resize, uses_ndvi, n_img)
        mask, posmap = b["mask"], m["map"].clamp(min=0)
        tot = posmap.flatten(1).sum(1)
        loc = ((posmap * mask).flatten(1).sum(1) / tot.clamp(min=1e-12)).numpy()
        chance = mask.flatten(1).mean(1).numpy()
        d = deletion_gap(p_loss, b["x"].float(), m["map"], fill, n_img, ndvi=b["ndvi"].float(), ndvi_fill=nd_fill)
        share = None
        if "per_band" in m:
            ab = m["per_band"].abs().flatten(2).sum(2)
            share = (ab / ab.sum(1, keepdim=True).clamp(min=1e-12)).numpy()
        for i, j in enumerate(b["idx"].tolist()):
            is_pos = int(b["y"][i]) == 1
            ok = is_pos and float(tot[i]) > 0
            r = dict(idx=j, region=ds_te.meta.at[j, "region"], block_r=int(ds_te.meta.at[j, "block_r"]),
                     block_c=int(ds_te.meta.at[j, "block_c"]), y=int(b["y"][i]), p=float(p_of.get(j, np.nan)),
                     loc=float(loc[i]) if ok else np.nan, chance=float(chance[i]) if ok else np.nan,
                     deletion_gap=float(d["gap"][i]) if is_pos else np.nan)
            if share is not None:
                r.update({f"share_{n}": float(s) for n, s in zip(names, share[i])})
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "explanations_per_patch.csv"), index=False)
    return df


def backfill(data_dir: str, root: str = "results") -> int:
    """Compute per-patch records for finished runs that do not have them yet.

    Runs are skipped by deforest.experiments once metrics.json exists, so models
    trained before this module was added never get explanations_per_patch.csv.
    This rebuilds each run's split, reloads its saved weights and fills the gap
    without retraining. Returns the number of runs filled in.
    """
    import torch

    from .data.dataset import PatchDataset
    from .data.io import load_all
    from .data.splits import assign_block_folds, splits
    from .models.zoo import build_model
    from .train import device

    meta, arrays = load_all(data_dir)
    meta["fold"] = assign_block_folds(meta)
    done = 0
    for task, col in (("forecast", "y_forecast"), ("detect", "y_detect")):
        for scheme in ("block", "loro"):
            todo = [m for m in sorted(glob.glob(os.path.join(root, task, scheme, "*", "*", "model.pt")))
                    if not os.path.exists(os.path.join(os.path.dirname(m), "explanations_per_patch.csv"))]
            if not todo:
                continue
            sp = {name: (tr, te) for name, tr, _, te in splits(meta, scheme, col)}
            for mp in todo:
                out = os.path.dirname(mp)
                ck = torch.load(mp, map_location="cpu")
                tr, te = sp[os.path.basename(out)]
                preds = pd.read_csv(os.path.join(out, "preds.csv"))
                if set(preds[preds.split == "test"].idx) != set(te.tolist()):
                    print(f"skip {out}: its test patches do not match --data (was the dataset rebuilt?)")
                    continue
                kw = dict(task=task, bands=ck["bands"], history=ck["history"],
                          with_ndvi=ck["variant"] == "bcos_pre_ndvi")
                model = build_model(ck["variant"], ck["n_channels"], download=False,
                            logit_temperature=ck.get("logit_temperature"))
                model.load_state_dict(ck["state_dict"])
                res = dict(model=model.to(device()), resize=ck["resize"], uses_ndvi=kw["with_ndvi"], preds=preds,
                           datasets=(PatchDataset(meta, arrays, tr, **kw), None, PatchDataset(meta, arrays, te, **kw)))
                per_patch(res, ck["variant"], out)
                done += 1
                print(f"backfilled {out}")
    return done


# --------------------------------------------------------------------------- report (no torch)
def _ap(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def _lift(d: pd.DataFrame) -> float:
    d = d.dropna(subset=["loc", "chance"])
    return float(d["loc"].mean() / d["chance"].mean()) if len(d) and d["chance"].mean() > 0 else float("nan")


def _shares(d: pd.DataFrame) -> np.ndarray | None:
    cols = [c for c in d.columns if c.startswith("share_")]
    return d[cols].mean().to_numpy() if cols and len(d) else None


def _js(a, b) -> float:
    """Jensen-Shannon distance (base 2): 0 = same reliance, 1 = disjoint."""
    from scipy.spatial.distance import jensenshannon
    if a is None or b is None or len(a) != len(b):
        return float("nan")
    return float(jensenshannon(a, b, base=2))


def _retention(v_in: float, v_out: float, chance: float, min_skill: float = MIN_SKILL, n_pos: int = MIN_POS) -> float:
    """Share of the above-chance part that survives the move. NaN if there was too little to lose."""
    s_in = v_in - chance
    ok = np.isfinite(s_in) and np.isfinite(v_out) and s_in >= min_skill and n_pos >= MIN_POS
    return float((v_out - chance) / s_in) if ok else float("nan")


def _load(root: str, task: str, scheme: str, exp: str):
    preds, expl, runs = [], [], 0
    for d in sorted(glob.glob(os.path.join(root, task, scheme, exp, "*"))):
        pf, ef = os.path.join(d, "preds.csv"), os.path.join(d, "explanations_per_patch.csv")
        if not os.path.exists(pf):
            continue
        runs += 1
        p = pd.read_csv(pf)
        preds.append(p[p.split == "test"])
        if os.path.exists(ef) and os.path.getsize(ef) > 5:
            expl.append(pd.read_csv(ef))
    cat = lambda v: pd.concat(v, ignore_index=True) if v else pd.DataFrame()
    return cat(preds), cat(expl), runs, len(expl)


def _pair_stats(pi: pd.DataFrame, po: pd.DataFrame, ei: pd.DataFrame, eo: pd.DataFrame) -> Dict[str, float]:
    """All ID/OOD numbers for one set of patches (already restricted to one region)."""
    out = dict(ap_in=_ap(pi.y.to_numpy(), pi.p.to_numpy()), ap_out=_ap(po.y.to_numpy(), po.p.to_numpy()),
               chance_ap=float(pi.y.mean()) if len(pi) else float("nan"))
    if len(ei) and len(eo):
        out.update(lift_in=_lift(ei), lift_out=_lift(eo),
                   del_in=float(ei.deletion_gap.mean()), del_out=float(eo.deletion_gap.mean()),
                   reliance_js=_js(_shares(ei), _shares(eo)))
    return out


def compare(root: str, task: str, exp: str, n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    """ID vs OOD, per held-out region, on the patches both runs scored."""
    pi, ei, runs_in, expl_runs_in = _load(root, task, "block", exp)
    po, eo, _, _ = _load(root, task, "loro", exp)
    if pi.empty or po.empty:
        return pd.DataFrame()
    rng, rows = np.random.default_rng(seed), []
    for region in sorted(set(po.region) & set(pi.region)):
        a, b = pi[pi.region == region], po[po.region == region]
        common = np.intersect1d(a.idx, b.idx)
        a, b = a[a.idx.isin(common)].sort_values("idx"), b[b.idx.isin(common)].sort_values("idx")
        ea = ei[(ei.region == region)] if len(ei) else ei
        eb = eo[(eo.region == region)] if len(eo) else eo
        if len(ea) and len(eb):
            ce = np.intersect1d(ea.idx, eb.idx)
            ea, eb = ea[ea.idx.isin(ce)], eb[eb.idx.isin(ce)]
        s = _pair_stats(a, b, ea, eb)
        n_pos = int(a.y.sum())
        n_epos = int(ea.y.sum()) if len(ea) else 0          # loss patches with explanation records in both runs
        row = dict(experiment=exp, task=task, region=region, patches=len(common), positives=n_pos,
                   id_folds=runs_in, id_expl_folds=expl_runs_in, expl_positives=n_epos, **s)
        ap_min = max(MIN_SKILL, 0.5 * s["chance_ap"]) if np.isfinite(s["chance_ap"]) else MIN_SKILL
        row["ap_retention"] = _retention(s["ap_in"], s["ap_out"], s["chance_ap"], ap_min, n_pos)
        if "lift_in" in s:
            row["lift_retention"] = _retention(s["lift_in"], s["lift_out"], 1.0, MIN_LIFT - 1.0, n_epos)
            row["del_retention"] = _retention(s["del_in"], s["del_out"], 0.0, MIN_DEL, n_epos)

        # paired block bootstrap: resample 5 km blocks, recompute both sides on the same blocks
        key = lambda d: d.block_r.astype(str) + "|" + d.block_c.astype(str)
        groups = {k: i for k, i in a.groupby(key(a)).groups.items()}
        ga = {k: a.loc[i] for k, i in groups.items()}
        gb = {k: g for k, g in b.groupby(key(b))}
        gea = {k: g for k, g in ea.groupby(key(ea))} if len(ea) else {}
        geb = {k: g for k, g in eb.groupby(key(eb))} if len(eb) else {}
        names, d_ap, d_ret = list(ga), [], []
        for _ in range(n_boot if len(names) > 1 else 0):
            pick = [names[i] for i in rng.choice(len(names), len(names), replace=True)]
            cat = lambda g: pd.concat([g[k] for k in pick if k in g], ignore_index=True) if any(k in g for k in pick) else pd.DataFrame()
            t = _pair_stats(cat(ga), cat(gb), cat(gea), cat(geb))
            d_ap.append(t["ap_in"] - t["ap_out"])
            if "lift_in" in t and np.isfinite(row.get("lift_retention", np.nan)) and np.isfinite(row["ap_retention"]):
                d_ret.append(_retention(t["lift_in"], t["lift_out"], 1.0, 1e-6, n_epos)
                             - _retention(t["ap_in"], t["ap_out"], t["chance_ap"], 1e-6, n_pos))
        ci = lambda v: tuple(np.nanpercentile(v, [2.5, 97.5])) if np.isfinite(v).sum() > 20 else (np.nan, np.nan)
        row["ap_drop_lo"], row["ap_drop_hi"] = ci(np.asarray(d_ap, float))
        # < 0 means explanations kept a smaller share of their above-chance quality than accuracy did
        row["expl_minus_ap_retention_lo"], row["expl_minus_ap_retention_hi"] = ci(np.asarray(d_ret, float))
        rows.append(row)
    return pd.DataFrame(rows)


def report(root: str = "results") -> str:
    """Write SHIFT.md, shift.csv and one figure per task. Returns the markdown."""
    frames = []
    for task_dir in sorted(glob.glob(os.path.join(root, "*", "loro"))):
        task = os.path.basename(os.path.dirname(task_dir))
        for exp_dir in sorted(glob.glob(os.path.join(task_dir, "*"))):
            if os.path.isdir(os.path.join(root, task, "block", os.path.basename(exp_dir))):
                frames.append(compare(root, task, os.path.basename(exp_dir)))
    frames = [f for f in frames if len(f)]
    if not frames:
        return ""
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(os.path.join(root, "shift.csv"), index=False)

    f3 = lambda v: "–" if not np.isfinite(v) else f"{v:.3f}"
    f2 = lambda v: "–" if not np.isfinite(v) else f"{v:.2f}"
    pct = lambda v: "–" if not np.isfinite(v) else f"{100 * v:.0f}%"
    L = ["# Explanations under regional shift (auto-generated by `python -m deforest.shift`)\n",
         "ID = the model trained on other blocks of the same region (block split). "
         "OOD = the model never saw the region (leave-one-region-out). Both columns are computed on the same patches. "
         "Retention = share of the above-chance part that survives the move (100% = nothing lost; "
         f"shown only when there are at least {MIN_POS} loss patches and the ID value is clearly above chance: "
         f"AP by {MIN_SKILL} and by half of chance, localisation at least {MIN_LIFT}x chance, deletion gap at least {MIN_DEL}).\n"]
    for task, t in df.groupby("task"):
        L.append(f"\n## {task}\n")
        if (t.id_folds < t.id_folds.max()).any() or t.id_folds.max() < 5:
            lo, hi = int(t.id_folds.min()), int(t.id_folds.max())
            L.append(f"*ID numbers use {lo if lo == hi else f'{lo}–{hi}'} of 5 spatial folds, "
                     "so only patches in those folds are compared.*\n")
        if "id_expl_folds" in t.columns and (t.id_expl_folds < t.id_folds).any():
            k = t[t.id_expl_folds > 0]
            L.append(f"*Explanation columns: only {int(k.id_expl_folds.min()) if len(k) else 0}"
                     f"–{int(t.id_expl_folds.max())} of {int(t.id_folds.max())} in-distribution folds have per-patch records "
                     "(the others have no saved model), so localisation, deletion gap and reliance are compared on the "
                     "patches of those folds only. The count is the second number in the Patches column.*\n")
        L.append("| Model | Held-out region | Patches (loss) | AP ID → OOD | AP drop 95% CI | AP kept | "
                 "Localisation ÷ chance ID → OOD | Kept | Deletion gap ID → OOD | Kept | Reliance shift (JS) |")
        L.append("|" + "---|" * 11)
        g = lambda r, c: float(r[c]) if c in t.columns and pd.notna(r[c]) else float("nan")
        for _, r in t.iterrows():
            L.append(f"| {r.experiment} | {r.region} | {r.patches} ({r.positives}; {int(r.expl_positives)} explained) | "
                     f"{f3(r.ap_in)} → {f3(r.ap_out)} | [{f3(r.ap_drop_lo)}, {f3(r.ap_drop_hi)}] | {pct(r.ap_retention)} | "
                     f"{f2(g(r, 'lift_in'))} → {f2(g(r, 'lift_out'))} | {pct(g(r, 'lift_retention'))} | "
                     f"{f3(g(r, 'del_in'))} → {f3(g(r, 'del_out'))} | {pct(g(r, 'del_retention'))} | {f3(g(r, 'reliance_js'))} |")
        L.append("\nA dash means the value is undefined or not trustworthy: no loss patches with positive image "
                 "evidence, too few loss patches, an in-distribution value too close to chance to take a ratio of, "
                 "or (reliance shift) a model that is not B-cos.")
        if "lift_retention" in t.columns:
            k = t.dropna(subset=["ap_retention", "lift_retention"])
            if len(k):
                worse = int((k.lift_retention < k.ap_retention).sum())
                sure = int((k.expl_minus_ap_retention_hi < 0).sum())
                L.append(f"\nLocalisation kept a smaller share than AP in **{worse} of {len(k)}** model-region pairs "
                         f"({sure} with a 95% block-bootstrap interval entirely below zero).")
            k = t.dropna(subset=["reliance_js", "ap_in", "ap_out"])
            if len(k) >= 4:
                from scipy.stats import spearmanr
                rho = spearmanr(k.reliance_js, k.ap_in - k.ap_out).correlation
                L.append(f"\nLabel-free check: Spearman correlation between reliance shift and AP drop across "
                         f"{len(k)} model-region pairs = **{rho:.2f}**. With this few pairs it is a lead to follow, not a result.")
            _figure(t, os.path.join(root, f"shift_{task}.png"), task)
            L.append(f"\n![Retention of accuracy vs explanations](shift_{task}.png)\n")
    text = "\n".join(L)
    with open(os.path.join(root, "SHIFT.md"), "w") as fh:
        fh.write(text)
    print(text)
    return text


def _figure(t: pd.DataFrame, path: str, task: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = t.dropna(subset=["ap_retention", "lift_retention"])
    if k.empty:
        return
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    marks = dict(zip(sorted(k.region.unique()), "o^sDvP*X"))
    for (exp, region), g in k.groupby(["experiment", "region"]):
        ax.scatter(g.ap_retention.clip(-0.5, 1.5), g.lift_retention.clip(-0.5, 1.5), marker=marks[region],
                   s=46, label=f"{exp} · {region}")
    ax.plot([-0.5, 1.5], [-0.5, 1.5], ls="--", c="grey", lw=1)
    ax.axhline(0, c="lightgrey", lw=0.8); ax.axvline(0, c="lightgrey", lw=0.8)
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("Share of above-chance AP kept on an unseen region")
    ax.set_ylabel("Share of above-chance localisation kept")
    ax.set_title(f"{task}: below the dashed line, explanations\ndegrade faster than accuracy", fontsize=9)
    ax.legend(fontsize=5.5, loc="upper left", ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("results", nargs="?", default="results")
    ap.add_argument("--data", default="data/v2")
    ap.add_argument("--backfill", action="store_true",
                    help="add per-patch records to finished runs that lack them (needs their model.pt)")
    a = ap.parse_args()
    if a.backfill:
        backfill(a.data, a.results)
    if not report(a.results):
        print("Nothing to compare yet: a model needs both a --scheme block and a --scheme loro run.")
