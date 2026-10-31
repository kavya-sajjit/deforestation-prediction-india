"""Turn model scores into risk levels people can act on, and measure how right they are.

    python -m deforest.risk_levels --data data/v2 --results results --model bcos_rgbn_hist

Three steps:

1. Calibrate. A raw network score of 0.7 does not mean "70% chance". We fit an
   isotonic regression from score to outcome, CROSS-FITTED by fold: the mapping
   used for fold k is learned only from the other folds' held-out predictions, so
   no square is calibrated with its own label.
2. Tier. Calibrated risk is cut into tiers (config.RISK_TIERS). For every tier we
   report how many held-out squares fell in it and how many really lost forest in
   2020-23, with a 95% block-bootstrap interval. That observed hit rate is the
   honest meaning of the tier, and it is what the website shows.
3. Hotspots. Touching high/elevated squares are joined into hotspots and ranked by
   expected forest at risk (sum of calibrated risk x forest hectares).

Outputs results/risk_report.json and results/reliability.png.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.ndimage import label as cc_label
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from . import config as C

PATCH_HA = (C.PATCH * 10) ** 2 / 1e4  # 40.96 ha


# --------------------------------------------------------------------------- load
def load_oof(results: str, model: str, scheme: str = "block") -> pd.DataFrame:
    """Held-out (test) predictions of every fold, with the fold name attached."""
    rows = []
    for f in sorted(glob.glob(os.path.join(results, "forecast", scheme, model, "*", "preds.csv"))):
        d = pd.read_csv(f)
        d = d[d.split == "test"].copy()
        d["fold"] = os.path.basename(os.path.dirname(f))
        rows.append(d)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows).drop_duplicates("idx").set_index("idx")


# --------------------------------------------------------------------------- calibrate
def calibrate(oof: pd.DataFrame) -> pd.Series:
    """Cross-fitted isotonic calibration. Returns calibrated risk q per idx."""
    q = pd.Series(np.nan, index=oof.index)
    folds = oof.fold.unique()
    if len(folds) < 2:  # nothing to cross-fit against: fall back, and say so
        print("warning: one fold only, calibration is fitted in-sample")
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof.p, oof.y)
        return pd.Series(iso.predict(oof.p), index=oof.index)
    for k in folds:
        tr, te = oof.fold != k, oof.fold == k
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof.p[tr], oof.y[tr])
        q[te] = iso.predict(oof.p[te])
    return q


def tier_of(q: float) -> str:
    for name, lo in C.RISK_TIERS:
        if q >= lo:
            return name
    return C.RISK_TIERS[-1][0]


# --------------------------------------------------------------------------- measure
def reliability(y: np.ndarray, q: np.ndarray, bins: int = 10) -> List[Dict]:
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (q >= a) & ((q < b) if b < 1 else (q <= b))
        if m.sum():
            out.append(dict(lo=round(a, 2), hi=round(b, 2), n=int(m.sum()),
                            predicted=float(q[m].mean()), observed=float(y[m].mean())))
    return out


def ece(rel: List[Dict], n: int) -> float:
    """Expected calibration error: average gap between predicted and observed, weighted by bin size."""
    return float(sum(r["n"] * abs(r["predicted"] - r["observed"]) for r in rel) / max(n, 1))


def tier_table(df: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> List[Dict]:
    """Per tier: squares, real losses, observed hit rate and a block-bootstrap 95% CI."""
    rng = np.random.default_rng(seed)
    keys = df[["region", "block_r", "block_c"]].astype(str).agg("|".join, axis=1).to_numpy()
    blocks = np.unique(keys)
    members = {b: np.flatnonzero(keys == b) for b in blocks}
    out = []
    for name, lo in C.RISK_TIERS:
        m = (df.tier == name).to_numpy()
        y = df.y.to_numpy()
        n, k = int(m.sum()), int(y[m].sum())
        rates = []
        for _ in range(n_boot):
            idx = np.concatenate([members[b] for b in rng.choice(blocks, len(blocks))])
            mm = m[idx]
            if mm.sum():
                rates.append(y[idx][mm].mean())
        ci = (float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))) if rates else (None, None)
        out.append(dict(tier=name, min_risk=lo, squares=n, lost=k,
                        hit_rate=(k / n) if n else None, ci=ci,
                        share_of_all_loss=(k / max(int(y.sum()), 1))))
    return out


def report(oof: pd.DataFrame, model: str) -> Dict:
    y, p, q = oof.y.to_numpy(), oof.p.to_numpy(), oof.q.to_numpy()
    rel_raw, rel_cal = reliability(y, np.clip(p, 0, 1)), reliability(y, q)
    return dict(
        model=model, squares=int(len(oof)), base_rate=float(y.mean()),
        brier_raw=float(brier_score_loss(y, np.clip(p, 0, 1))), brier_calibrated=float(brier_score_loss(y, q)),
        brier_baseline=float(brier_score_loss(y, np.full_like(q, y.mean()))),
        ece_raw=ece(rel_raw, len(y)), ece_calibrated=ece(rel_cal, len(y)),
        reliability_raw=rel_raw, reliability=rel_cal,
        tiers=tier_table(oof),
        note=("Calibrated with isotonic regression, cross-fitted across spatial folds. "
              "Hit rate = share of held-out squares in that tier that really lost >=2% forest in 2020-23."),
    )


# --------------------------------------------------------------------------- hotspots
def hotspots(cells: List[Dict], lon: np.ndarray, lat: np.ndarray, rows: int, cols: int) -> List[Dict]:
    """Join touching high/elevated squares (8-neighbour) and rank by expected forest at risk.

    cells: dicts with r, c, f (forest share), q (calibrated risk), t (tier).
    Each cell in a hotspot gets cell['h'] = hotspot rank (1 = most forest at risk).
    """
    grid = np.zeros((rows, cols), bool)
    at = {}
    for i, p in enumerate(cells):
        if p.get("t") in C.HOTSPOT_TIERS:
            grid[p["r"], p["c"]] = True
            at[(p["r"], p["c"])] = i
    lab, n = cc_label(grid, structure=np.ones((3, 3)))
    spots = []
    for k in range(1, n + 1):
        rr, cc = np.nonzero(lab == k)
        if len(rr) < C.HOTSPOT_MIN_SQUARES:
            continue
        members = [cells[at[(r, c)]] for r, c in zip(rr, cc)]
        forest_ha = sum(m["f"] for m in members) * PATCH_HA
        at_risk_ha = sum(m["f"] * m["q"] for m in members) * PATCH_HA
        qs = np.array([m["q"] for m in members])
        # centre and bounds from the patch-corner grid
        lo = [float(lon[r, c]) for r, c in zip(rr, cc)] + [float(lon[r + 1, c + 1]) for r, c in zip(rr, cc)]
        la = [float(lat[r, c]) for r, c in zip(rr, cc)] + [float(lat[r + 1, c + 1]) for r, c in zip(rr, cc)]
        spots.append(dict(
            squares=len(rr), forest_ha=round(forest_ha), at_risk_ha=round(at_risk_ha),
            mean_risk=round(float(qs.mean()), 3), max_risk=round(float(qs.max()), 3),
            tier="high" if (qs >= dict(C.RISK_TIERS)["high"]).any() else "elevated",
            lon=round(float(np.mean(lo)), 5), lat=round(float(np.mean(la)), 5),
            bbox=[round(min(lo), 5), round(min(la), 5), round(max(lo), 5), round(max(la), 5)],
            cells=[[int(r), int(c)] for r, c in zip(rr, cc)],
        ))
    spots.sort(key=lambda s: -s["at_risk_ha"])
    spots = spots[: C.HOTSPOT_TOP]
    for rank, s in enumerate(spots, 1):
        s["id"] = rank
        for r, c in s["cells"]:
            cells[at[(r, c)]]["h"] = rank
    return spots


def hotspots_geojson(region: str, spots: List[Dict], lon: np.ndarray, lat: np.ndarray) -> Dict:
    """One feature per hotspot square, for upload to Global Forest Watch or a GIS."""
    feats = []
    for s in spots:
        for r, c in s["cells"]:
            ring = [[float(lon[r, c]), float(lat[r, c])], [float(lon[r, c + 1]), float(lat[r, c + 1])],
                    [float(lon[r + 1, c + 1]), float(lat[r + 1, c + 1])], [float(lon[r + 1, c]), float(lat[r + 1, c])]]
            ring.append(ring[0])
            feats.append(dict(type="Feature", geometry=dict(type="Polygon", coordinates=[ring]),
                              properties=dict(region=region, hotspot=s["id"], tier=s["tier"],
                                              mean_risk=s["mean_risk"])))
    return dict(type="FeatureCollection", features=feats)


# --------------------------------------------------------------------------- plot
def plot_reliability(rep: Dict, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="perfect")
    for key, lab, col in (("reliability_raw", f"raw (ECE {rep['ece_raw']:.3f})", "#b9673a"),
                          ("reliability", f"calibrated (ECE {rep['ece_calibrated']:.3f})", "#3f6340")):
        r = rep[key]
        ax.plot([b["predicted"] for b in r], [b["observed"] for b in r], "o-", c=col, label=lab)
    ax.set_xlabel("predicted risk")
    ax.set_ylabel("share that really lost forest")
    ax.set_title(f"Reliability · {rep['model']}")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(results: str, model: str) -> tuple:
    oof = load_oof(results, model)
    if oof.empty:
        return oof, None
    oof["q"] = calibrate(oof)
    oof["tier"] = oof.q.map(tier_of)
    rep = report(oof, model)
    return oof, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--model", default="bcos_rgbn_hist")
    a = ap.parse_args()
    oof, rep = run(a.results, a.model)
    if rep is None:
        raise SystemExit(f"no held-out predictions for {a.model} under {a.results}/forecast/block")
    with open(os.path.join(a.results, "risk_report.json"), "w") as fh:
        json.dump(rep, fh, indent=1)
    plot_reliability(rep, os.path.join(a.results, "reliability.png"))
    print(f"base rate {rep['base_rate']:.3f} | Brier raw {rep['brier_raw']:.4f} -> calibrated "
          f"{rep['brier_calibrated']:.4f} (always-guess-base-rate {rep['brier_baseline']:.4f}) | "
          f"ECE {rep['ece_raw']:.3f} -> {rep['ece_calibrated']:.3f}")
    for t in rep["tiers"]:
        hr = "–" if t["hit_rate"] is None else f"{t['hit_rate']:.0%} [{t['ci'][0]:.0%}, {t['ci'][1]:.0%}]"
        print(f"  {t['tier']:9s} risk>={t['min_risk']:.2f}: {t['squares']:5d} squares, {t['lost']:4d} lost, "
              f"hit rate {hr}, {t['share_of_all_loss']:.0%} of all loss")


if __name__ == "__main__":
    main()
