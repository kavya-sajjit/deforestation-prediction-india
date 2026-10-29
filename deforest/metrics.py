"""Metrics for an imbalanced binary task, with spatially honest uncertainty.

Accuracy is reported but never headlined: with few loss patches, a model that
always says "no loss" can score >90% accuracy. The headline metric is average
precision (area under the precision-recall curve), compared against its chance
level, which equals the positive rate.

Confidence intervals use a *block* bootstrap: whole 5 km blocks are resampled,
because neighbouring patches are not independent.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score, accuracy_score)


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold chosen on the VALIDATION set only, then frozen for test."""
    if y.sum() == 0:
        return 0.5
    ts = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 99)))
    f1s = [f1_score(y, p >= t, zero_division=0) for t in ts]
    return float(ts[int(np.argmax(f1s))])


def capture_at_top(y: np.ndarray, p: np.ndarray, area_frac: float = 0.10,
                   weight: np.ndarray | None = None) -> float:
    """Share of future loss found in the `area_frac` highest-risk patches.

    This is how a risk map is used in practice: an NGO can only check a small
    share of the forest, so what matters is how much loss sits in the area it
    checks. Chance level equals area_frac.
    """
    w = y.astype(float) if weight is None else weight * y
    if w.sum() == 0:
        return float("nan")
    k = max(1, int(round(area_frac * len(p))))
    top = np.argsort(-p)[:k]
    return float(w[top].sum() / w.sum())


def point_metrics(y: np.ndarray, p: np.ndarray, thr, weight=None) -> Dict[str, float]:
    yhat = (p >= thr).astype(int)
    one_class = len(np.unique(y)) < 2
    return dict(
        n=int(len(y)), positives=int(y.sum()), pos_rate=float(y.mean()),
        ap=float("nan") if one_class else float(average_precision_score(y, p)),
        roc_auc=float("nan") if one_class else float(roc_auc_score(y, p)),
        f1=float(f1_score(y, yhat, zero_division=0)),
        precision=float(precision_score(y, yhat, zero_division=0)),
        recall=float(recall_score(y, yhat, zero_division=0)),
        balanced_acc=float(balanced_accuracy_score(y, yhat)) if not one_class else float("nan"),
        mcc=float(matthews_corrcoef(y, yhat)) if not one_class else float("nan"),
        accuracy=float(accuracy_score(y, yhat)),
        brier=float(brier_score_loss(y, p)),
        capture_top10=capture_at_top(y, p, 0.10, weight),
        threshold=float(np.mean(thr)),
    )


def block_bootstrap(df: pd.DataFrame, metric: str = "ap", n_boot: int = 1000, seed: int = 0) -> tuple:
    """95% CI for `metric` by resampling (region, block) clusters with replacement.

    df needs columns y, p, region, block_r, block_c (and threshold for point metrics).
    """
    rng = np.random.default_rng(seed)
    keys = df[["region", "block_r", "block_c"]].astype(str).agg("|".join, axis=1)
    groups = {k: g.index.to_numpy() for k, g in df.groupby(keys)}
    names = list(groups)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(names), len(names), replace=True)
        idx = np.concatenate([groups[names[i]] for i in pick])
        y, p = df.loc[idx, "y"].to_numpy(), df.loc[idx, "p"].to_numpy()
        if len(np.unique(y)) < 2:
            continue
        thr = df.loc[idx, "threshold"].to_numpy() if "threshold" in df else 0.5
        vals.append(point_metrics(y, p, thr)[metric])
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
