"""Long-horizon projection of forest loss per 640 m square, with a backtest.

The trained models forecast one window (2020-23) from 2019 imagery. Nothing in this project can
see 2036, so this module does something narrower and says so: it projects each square's recent
rate of clearing forward, and measures how well that worked in the past.

Method (per region, per square):

    remaining forest  R(y) = forest in 2000 - loss up to and including y      (share of the square)
    clearing rate     r    = loss in the last K years / forest-years exposed in those years,
                             shrunk towards the region's rate with s pseudo-years of exposure
                             (squares with little history borrow from the region)
    projection        R(t) = R(T) * (1 - r) ** (t - T)                         for t > T

T is the last observed year. K and s are chosen on an early validation split (history up to 2008,
scored on 2009-2012) and then tested on a later one it never saw (history up to 2012, scored on every
year from 2013 to T). The test numbers are what the website shows.

The low/high range reruns the projection with the other window lengths. It is a spread of
plausible trend choices, not a statistical confidence interval.

What this is not: it has no roads, prices, policy or climate, and it cannot anticipate a new mine
or a protected-area decision. It answers "if the recent pace continues, where does the forest go?".

    python -m deforest.projection web/public/data            # adds projections to every region file
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np

PATCH_HA = 40.96            # 640 m x 640 m
END = 2036
WINDOWS = (3, 5, 8)         # years of history used for the rate
SHRINK = (0.0, 2.0, 5.0, 10.0)
EVENT = 0.02                # "lost at least 2% of its remaining forest", as for the risk model


# --------------------------------------------------------------------------- arrays
def _arrays(doc: dict):
    """f (n,), loss (n, Y) with loss[:, j] = share of square lost in year y0 + j, and y0."""
    ps = doc["patches"]
    years = sorted({int(y) for p in ps for y in (p.get("l") or {})})
    y0, y1 = 2001, max(years + [int(doc.get("last_observed", 0) or 0)])
    f = np.array([p["f"] for p in ps], float)
    loss = np.zeros((len(ps), y1 - y0 + 1))
    for i, p in enumerate(ps):
        for y, v in (p.get("l") or {}).items():
            if int(y) >= y0:
                loss[i, int(y) - y0] = v
    return f, loss, y0, y1


def _remaining(f, loss, y0, year):
    """Share of each square still forest at the end of `year`."""
    k = year - y0 + 1
    return np.clip(f - loss[:, :max(k, 0)].sum(axis=1), 0.0, None)


def rates(f, loss, y0, T, K, s):
    """Shrunk annual clearing rate per square from the K years ending at T."""
    L = np.zeros(len(f))
    E = np.zeros(len(f))
    for y in range(T - K + 1, T + 1):
        E += _remaining(f, loss, y0, y - 1)
        L += loss[:, y - y0]
    tot_e = E.sum()
    rbar = L.sum() / tot_e if tot_e > 0 else 0.0
    R = _remaining(f, loss, y0, T)
    r = (L + s * rbar * R) / np.maximum(E + s * R, 1e-12)
    return np.clip(np.where(R > 0, r, 0.0), 0.0, 1.0), rbar


def project(f, loss, y0, T, K, s, until):
    """Additional share of each square lost after T, cumulative, for years T+1..until."""
    r, _ = rates(f, loss, y0, T, K, s)
    R = _remaining(f, loss, y0, T)
    h = np.arange(1, until - T + 1)
    return R[:, None] * (1 - (1 - r[:, None]) ** h[None, :])


# --------------------------------------------------------------------------- scoring
def _ap(y, score):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, score)) if 0 < y.sum() < len(y) else float("nan")


def score(f, loss, y0, T, K, s, until):
    """How a projection made at T compares with what happened in T+1..until."""
    pred = project(f, loss, y0, T, K, s, until)
    actual = np.cumsum(loss[:, T + 1 - y0: until - y0 + 1], axis=1)
    R = _remaining(f, loss, y0, T)
    keep = R > 0.05                                   # squares that still had forest to lose
    p, a, R = pred[keep], actual[keep], R[keep]
    event = (a[:, -1] / R) >= EVENT
    rank = p[:, -1] / R
    last_year_only = loss[keep, T - y0] / R           # naive: rank by last year's loss alone
    from scipy.stats import spearmanr
    return dict(
        origin=T, until=until, squares=int(keep.sum()), events=int(event.sum()),
        chance_ap=round(float(event.mean()), 4),
        ap=round(_ap(event, rank), 4),
        ap_last_year_only=round(_ap(event, last_year_only), 4),
        spearman=round(float(spearmanr(p[:, -1], a[:, -1]).correlation), 4),
        total_ratio=round(float(p[:, -1].sum() / max(a[:, -1].sum(), 1e-12)), 3),
        mse=float(((p - a) ** 2).mean()),
        by_year=[dict(year=T + 1 + j, projected_ha=round(float(p[:, j].sum() * PATCH_HA), 1),
                      actual_ha=round(float(a[:, j].sum() * PATCH_HA), 1)) for j in range(p.shape[1])],
    )


def choose(f, loss, y0, val_origin=2008, val_until=2012):
    """Pick (K, s) with the lowest per-square error on the validation split."""
    best = None
    for K in WINDOWS:
        if val_origin - K + 1 < y0:
            continue
        for s in SHRINK:
            m = score(f, loss, y0, val_origin, K, s, val_until)["mse"]
            if best is None or m < best[0]:
                best = (m, K, s)
    return best[1], best[2]


# --------------------------------------------------------------------------- per region
def run_region(doc: dict, end: int = END, test_origin: int = 2012) -> dict:
    f, loss, y0, T = _arrays(doc)
    K, s = choose(f, loss, y0)
    test = score(f, loss, y0, test_origin, K, s, T)
    test.pop("mse")
    pj = project(f, loss, y0, T, K, s, end)
    totals = {k: [] for k in ("central", "low", "high")}
    alt = [project(f, loss, y0, T, k, s, end).sum(axis=0) * PATCH_HA for k in WINDOWS]
    for j in range(end - T):
        vals = [a[j] for a in alt]
        totals["central"].append(round(float(pj[:, j].sum() * PATCH_HA), 1))
        totals["low"].append(round(float(min(vals)), 1))
        totals["high"].append(round(float(max(vals)), 1))
    for p, row in zip(doc["patches"], pj):
        p["pj"] = [round(float(v), 4) for v in row]
    _, rbar = rates(f, loss, y0, T, K, s)
    doc["last_observed"] = T
    doc["projection"] = dict(
        origin=T, end=end, window=K, shrink=s, region_rate=round(float(rbar), 5),
        years=list(range(T + 1, end + 1)), total_ha=totals,
        remaining_ha=round(float(_remaining(f, loss, y0, T).sum() * PATCH_HA), 1),
        backtest=test,
        method="recent clearing rate per square, shrunk to the region rate, projected forward; "
               "backtested from 2012 on every later observed year",
    )
    return doc["projection"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data", nargs="?", default="web/public/data")
    ap.add_argument("--end", type=int, default=END)
    a = ap.parse_args()
    report = {}
    for path in sorted(glob.glob(os.path.join(a.data, "*.json"))):
        if path.endswith(("summary.json", "projection.json")):
            continue
        with open(path) as fh:
            doc = json.load(fh)
        if "patches" not in doc:
            continue
        pr = run_region(doc, a.end)
        with open(path, "w") as fh:
            json.dump(doc, fh, separators=(",", ":"))
        report[doc["region"]] = {k: v for k, v in pr.items() if k != "years"}
        b = pr["backtest"]
        print(f"[{doc['region']}] observed to {pr['origin']}, window {pr['window']} y, shrink {pr['shrink']} | "
              f"backtest {b['origin']}->{b['until']}: AP {b['ap']} (chance {b['chance_ap']}, last-year-only "
              f"{b['ap_last_year_only']}), total projected/actual {b['total_ratio']} | "
              f"{pr['end']}: {pr['total_ha']['central'][-1]:.0f} ha more lost "
              f"({pr['total_ha']['low'][-1]:.0f}-{pr['total_ha']['high'][-1]:.0f}) of {pr['remaining_ha']:.0f} ha left")
        if doc.get("synthetic"):
            print("   (this file is DEMO data, so these numbers mean nothing)")
    with open(os.path.join(a.data, "projection.json"), "w") as fh:
        json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
