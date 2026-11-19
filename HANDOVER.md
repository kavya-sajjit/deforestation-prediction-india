# Where this project stands and how to continue (written 16 November 2026)

Read this first if you are picking the project up without help. Everything below can be done from a
browser with a free Google Colab account.

## State

| Piece | State |
|---|---|
| Code: data, models, training, explanation checks, known-vs-unseen comparison (`deforest/shift.py`) | done, tests pass (`pytest -q tests/ -k "not end_to_end"`) |
| Results for the pretrained B-cos models, both CNNs and the two baselines, known and unseen regions | done, on Drive; snapshot in [`results/snapshots/2026-11-15`](results/snapshots/2026-11-15/README.md) |
| Results for the four from-scratch B-cos models | **stale**: trained with too high a learning rate. Retraining started 16 Nov; check below whether it finished |
| Same learning-rate check for the standard CNN | not done |
| More regions (two of four have too few loss patches), three seeds | not done |
| README results block, figures, risk maps | not regenerated since the retraining |
| Paper | plan and first findings in [`paper/PLAN.md`](paper/PLAN.md); nothing written |

## The retraining started on 16 November: check before trusting it

A retraining of the four from-scratch B-cos models (block folds, then unseen regions) was started on
16 November around 15:55 IST from a **second Google account**, not the one that holds the earlier
results. Contact with that Colab session was lost about 16 minutes in, so whether it finished is unknown.

To find out, mount that account's Drive in Colab and run:

```
!tail -3 $WORK/retrain_bcos_block.log; tail -3 $WORK/retrain_bcos_loro.log
!grep -l '"lr": 0.0001' $WORK/results/*/*/bcos_*/*/metrics.json | wc -l    # 36 means all runs are done
```

Then pick one Drive and keep everything there. Simplest: copy `results/` from the second account's
`deforestation-v2` folder into the first account's (keep both `forecast` and `detect`, let newer files
replace older ones), and run step 3 below from the first account. Runs that are already finished at
the new learning rate are skipped. Before merging, check that both Drives hold the same dataset:
`!md5sum $WORK/data/v2/*.csv` must print the same values on both.

## Where the files are

- **GitHub** (this repo): code, plan, snapshot of results.
- **Google Drive, folder `deforestation-v2`** of the account that ran Colab: `data/v2` (the dataset),
  `results/` (every run: `metrics.json`, `preds.csv`, `model.pt`, plus `SUMMARY.md`, `SHIFT.md`,
  `shift.csv`), `diagnostics/`, and `retrain_*.log`.
  **Use the same Google account every time.** Colab mounts the Drive of whichever account is signed
  in; a different account has a different (or empty) `deforestation-v2` folder.

## Continue in Colab

1. Open `notebooks/03_v2_colab.ipynb` from GitHub in Colab. Runtime → Change runtime type → T4 GPU.
2. Run the first two code cells (mount Drive, clone the repo and install).
3. Run this in a new cell. It is safe to run again after a disconnect: finished runs are skipped,
   stale or weightless runs are redone.

   ```
   !python -m deforest.experiments --data $WORK/data/v2 --out $WORK/results --profile core --scheme block --only priority
   !python -m deforest.experiments --data $WORK/data/v2 --out $WORK/results --profile core --scheme loro --only priority
   !python -m deforest.shift $WORK/results --data $WORK/data/v2 --backfill
   !python -m deforest.shift $WORK/results
   ```

   How to tell it is finished: each `deforest.experiments` call prints only the summary tables and no
   `== forecast | ...` lines when there is nothing left to train.
4. Put the results in the repo: run section 5 of the notebook (`summarize`, copy `results/`, then
   `scripts/update_readme.py`), download `results/` without the `model.pt` files, and commit them
   with the README.

## Map to 2036 (added 19 November 2026)

- **Observed loss to 2025.** `deforest.web_export` now reads the newest Hansen release it can reach
  (GFC-2025-v1.13, then v1.12, then v1.11) for the map only. Training labels still use v1.11.
- **Projection 2026-2036.** `deforest/projection.py` carries each square's recent clearing pace
  forward, shrunk towards its region's pace, and backtests itself (history to 2012, scored on every
  later observed year). The map shows projected years with a dashed outline and a "projected" label,
  and the panel shows the backtest score and a low/high range. It is a trend scenario, not a model
  forecast; roads, mines and policy are not inputs.
- To regenerate: run the first two code cells of the Colab notebook, then section 6. To add the
  projection to map files you already have, run `python -m deforest.projection web/public/data`.

## Next steps, in order

1. Finish retraining the four from-scratch B-cos models (step 3 above) and compare with the
   snapshot. Expect `bcos_bitemporal` to rise a lot; the "0.92 against 0.31" detection gap quoted in
   earlier letters is no longer valid.
2. Give the standard CNN the same check:
   edit `lr_scratch` in `deforest/config.py` to 3e-4 and 1e-4 in turn, train `cnn_bitemporal` on fold 0,
   and keep the value with the best validation AP. Without this, comparing CNN and B-cos is unfair.
3. Add regions with at least 100 loss patches each (`scripts/scan_hansen_aoi.py` finds candidates,
   `deforest.data.build` downloads them), then rerun step 3 above.
4. Three seeds for the headline models (`seed` in `TrainConfig`, separate `--out` folder per seed).
5. Only then write: `paper/PLAN.md` sections 2, 11 and 12 say which hypotheses held.

## What can be said today, and what cannot

Can: on regions it never saw, a baseline that only knows past clearing keeps its score while the image
models lose part of theirs; the pretrained interpretable models' channel reliance stays almost
unchanged while their accuracy falls. One seed, two regions with enough data.

Cannot: any claim about how much accuracy interpretability costs (the from-scratch B-cos numbers are
stale and the CNN is untuned); anything about central India or Odisha; anything described as
significant.
