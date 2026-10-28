"""Run every experiment on every spatial split, then summarise.

    python -m deforest.experiments --data data/v2 --out results --profile core
    python -m deforest.experiments --data data/v2 --out results --profile core --scheme loro

Profiles: "quick" (smoke test), "core" (what the README reports), "full" (adds
extra seeds-worth of variants). Re-running skips finished runs, so a Colab
disconnect only loses the run in progress.
"""
from __future__ import annotations

import argparse
import json
import os

from . import baselines
from . import config as C
from .data.io import load_all
from .data.splits import assign_block_folds, check_no_leak, splits

RGB, RGBN = ["red", "green", "blue"], ["red", "green", "blue", "nir"]

# name -> (kind, variant, bands, history)
EXPERIMENTS = {
    "forecast": {
        "history_lr":        ("tab", "history_lr", None, True),
        "gbm":               ("tab", "gbm", None, True),
        "cnn_rgbn":          ("nn", "cnn", RGBN, False),
        "bcos_rgb":          ("nn", "bcos", RGB, False),
        "bcos_rgbn":         ("nn", "bcos", RGBN, False),
        "bcos_pre_rgb":      ("nn", "bcos_pre_rgb", RGB, False),
        "bcos_pre_inflate":  ("nn", "bcos_pre_inflate", RGBN, False),
        "bcos_pre_project":  ("nn", "bcos_pre_project", RGBN, False),
        "bcos_pre_ndvi":     ("nn", "bcos_pre_ndvi", RGB, False),
        "cnn_rgbn_hist":     ("nn", "cnn", RGBN, True),
        "bcos_rgbn_hist":    ("nn", "bcos", RGBN, True),
    },
    "detect": {
        "gbm":               ("tab", "gbm", None, False),
        "cnn_bitemporal":    ("nn", "cnn", RGBN, False),
        "bcos_bitemporal_rgb": ("nn", "bcos", RGB, False),
        "bcos_bitemporal":   ("nn", "bcos", RGBN, False),
    },
}
ABLATE = {"bcos_rgbn", "bcos_rgbn_hist", "bcos_bitemporal"}
PROFILES = {
    "quick": dict(epochs=2, patience=2),
    "core": dict(epochs=30, patience=8),
    "full": dict(epochs=50, patience=12),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/v2")
    ap.add_argument("--out", default="results")
    ap.add_argument("--tasks", nargs="*", default=["forecast", "detect"])
    ap.add_argument("--scheme", default="block", choices=["block", "loro"])
    ap.add_argument("--profile", default="core", choices=list(PROFILES))
    ap.add_argument("--only", nargs="*", default=None, help="run only these experiment names")
    ap.add_argument("--folds", nargs="*", default=None, help="e.g. fold0 fold1")
    ap.add_argument("--no-download", action="store_true", help="random init instead of ImageNet B-cos weights")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    from . import explain, train   # torch imported only when needed
    meta, arrays = load_all(a.data)
    meta["fold"] = assign_block_folds(meta)
    cfg = C.TrainConfig(num_workers=a.workers, **PROFILES[a.profile])

    for task in a.tasks:
        col = "y_forecast" if task == "forecast" else "y_detect"
        X = baselines.features(meta, arrays, task)
        for k_split, (split_name, tr, va, te) in enumerate(splits(meta, a.scheme, col)):
            if a.folds and split_name not in a.folds:
                continue
            check_no_leak(meta, tr, te)
            for name, (kind, variant, bands, hist) in EXPERIMENTS[task].items():
                if a.only and name not in a.only:
                    continue
                out = os.path.join(a.out, task, a.scheme, name, split_name)
                if os.path.exists(os.path.join(out, "metrics.json")):
                    continue
                print(f"== {task} | {a.scheme} | {split_name} | {name}  "
                      f"(train {len(tr)}, val {len(va)}, test {len(te)}, test positives {int(meta.loc[te, col].sum())})")
                if kind == "tab":
                    baselines.run(variant, task, meta, X, tr, va, te, out)
                    continue
                res = train.run(variant, task, bands, hist, meta, arrays, tr, va, te, out, cfg,
                                download=not a.no_download, tag=name)
                expl = explain.evaluate_explanations(res, variant, out)
                print("   explanations:", json.dumps({k: v for k, v in expl.items() if k != "band_share"}))
                if name in ABLATE:
                    explain.band_ablation(res, out)
                if k_split == 0:
                    explain.gallery(res, os.path.join(out, "gallery.png"))

    from .summarize import summarize
    summarize(a.out)


if __name__ == "__main__":
    main()
