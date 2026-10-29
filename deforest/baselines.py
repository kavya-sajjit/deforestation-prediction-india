"""Non-deep baselines. A CNN result means little unless it beats these.

history_lr   logistic regression on "where was forest already being cleared?"
             (past-loss share in the patch, past-loss density within 2.5 km,
             forest share). Deforestation spreads from existing frontiers, so
             this is the trivial forecast every model must beat.
gbm          gradient-boosted trees on hand-made spectral features (band means
             and spreads, NDVI) plus the history features; for detection also
             the T1 features and the T0->T1 change (dNDVI).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data.io import ndvi01
from .metrics import best_f1_threshold, point_metrics


def features(meta: pd.DataFrame, arrays: dict, task: str) -> pd.DataFrame:
    def stats(a, tag):
        a = a.astype(np.float32)
        f = {}
        for b, n in enumerate(["red", "green", "blue", "nir"]):
            f[f"{n}_mean_{tag}"] = a[:, b].mean((1, 2))
            f[f"{n}_std_{tag}"] = a[:, b].std((1, 2))
        nd = ndvi01(a[:, 0], a[:, 3])
        f[f"ndvi_mean_{tag}"], f[f"ndvi_std_{tag}"] = nd.mean((1, 2)), nd.std((1, 2))
        f[f"ndvi_p10_{tag}"] = np.percentile(nd.reshape(len(nd), -1), 10, axis=1)
        return f
    f = stats(arrays["t0"], "t0")
    if task == "detect":
        f.update(stats(arrays["t1"], "t1"))
        f["dndvi_mean"] = f["ndvi_mean_t1"] - f["ndvi_mean_t0"]
        f["dndvi_p10"] = f["ndvi_p10_t1"] - f["ndvi_p10_t0"]
    df = pd.DataFrame(f, index=meta.index)
    for c in ("forest_frac", "hist_loss_frac", "hist_density"):
        df[c] = meta[c]
    return df


HISTORY_COLS = ["hist_loss_frac", "hist_density", "forest_frac"]


def run(name: str, task: str, meta, X: pd.DataFrame, train_idx, val_idx, test_idx, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    col = "y_forecast" if task == "forecast" else "y_detect"
    if name == "history_lr":
        cols = HISTORY_COLS
        model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
    elif name == "gbm":
        cols = list(X.columns)
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced",
                                               random_state=0)
    else:
        raise ValueError(name)
    model.fit(X.loc[train_idx, cols], meta.loc[train_idx, col])
    p_va = model.predict_proba(X.loc[val_idx, cols])[:, 1]
    thr = best_f1_threshold(meta.loc[val_idx, col].to_numpy(), p_va)
    p_te = model.predict_proba(X.loc[test_idx, cols])[:, 1]
    rows = []
    for split, idx, p in (("val", val_idx, p_va), ("test", test_idx, p_te)):
        m = meta.loc[idx]
        rows.append(pd.DataFrame(dict(idx=idx, region=m.region.values, block_r=m.block_r.values,
                                      block_c=m.block_c.values, y=m[col].values, p=p, split=split,
                                      threshold=thr)))
    preds = pd.concat(rows, ignore_index=True)
    preds.to_csv(os.path.join(out_dir, "preds.csv"), index=False)
    te = preds[preds.split == "test"]
    res = point_metrics(te.y.to_numpy(), te.p.to_numpy(), thr)
    res.update(variant=name, task=task)
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    return res
