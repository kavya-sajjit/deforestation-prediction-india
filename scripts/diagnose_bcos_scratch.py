"""Why does the from-scratch B-cos detection model sit at cross-entropy ln 2?

    python scripts/diagnose_bcos_scratch.py --data data/v2 --out diagnostics --temperatures none 0.5 0.2 0.1
    python scripts/diagnose_bcos_scratch.py --data data/v2 --out diagnostics --lrs 1e-3 3e-4 1e-4

First run on the real data (15 Nov 2026, fold 0, 10 epochs): the logit temperature is NOT the cause.
With none, 0.5, 0.2 and 0.1 the loss was 1.0 to 4.7 in epoch 0 and exactly 0.693 from epoch 1 on
(only 0.1 began to move, at epoch 8). A loss that starts above ln 2 and then pins to it means the
network collapsed to near-zero logits during the first epoch, which points at the step size
(AdamW at 1e-3 with no warm-up), so --lrs tests that next.

Trains `bcos_bitemporal` (the model that failed) on spatial fold 0 for a few epochs per temperature
and writes one row per epoch to <out>/bcos_scratch_temperature.csv. Read it like this:
  * train_loss stuck near 0.693 and test AP near chance  -> still on the plateau
  * first-epoch loss far above 1                         -> temperature too small, logits blow up
  * loss falling steadily and test AP well above chance  -> candidate for config.SCRATCH_LOGIT_TEMPERATURE
This is a diagnostic. Its numbers never go into the README.
"""
import argparse
import os
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from deforest import config as C, train                                   # noqa: E402
from deforest.data.io import load_all                                     # noqa: E402
from deforest.data.splits import assign_block_folds, splits               # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/v2")
ap.add_argument("--out", default="results/diagnostics")
ap.add_argument("--temperatures", nargs="+", default=["none"])
ap.add_argument("--lrs", nargs="+", type=float, default=[C.TrainConfig().lr_scratch])
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--workers", type=int, default=2)
a = ap.parse_args()

meta, arrays = load_all(a.data)
meta["fold"] = assign_block_folds(meta)
name, tr, va, te = next(iter(splits(meta, "block", "y_detect")))
os.makedirs(a.out, exist_ok=True)
rows = []
for t in a.temperatures:
  for lr in a.lrs:
    T = None if t.lower() == "none" else float(t)
    C.SCRATCH_LOGIT_TEMPERATURE = T
    tag = f"T={T},lr={lr:g}"
    print(f"=== logit temperature {T}, lr {lr:g}  ({name}: train {len(tr)}, val {len(va)}, test {len(te)})", flush=True)
    work = tempfile.mkdtemp()
    r = train.run("bcos", "detect", ["red", "green", "blue", "nir"], False, meta, arrays, tr, va, te, work,
                  C.TrainConfig(epochs=a.epochs, patience=a.epochs, num_workers=a.workers, lr_scratch=lr),
                  download=False, tag=tag)
    m = r["metrics"]
    for h in pd.read_csv(os.path.join(work, "history.csv")).to_dict("records"):
        rows.append(dict(setting=tag, temperature=t, lr=lr, **h, test_ap=m["ap"], chance=m["pos_rate"]))
    print(f"    test AP {m['ap']:.3f}  (chance {m['pos_rate']:.3f})", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, f"bcos_scratch_{'lr' if len(a.lrs) > 1 else 'temperature'}.csv"), index=False)
print(pd.DataFrame(rows).groupby("setting").agg(first_loss=("train_loss", "first"), last_loss=("train_loss", "last"),
                                                    best_val_ap=("val_ap", "max"), test_ap=("test_ap", "first"),
                                                    chance=("chance", "first")).to_string())
