# Results snapshot, 15 November 2026

**What this is.** A hand-copied snapshot of the tables produced on Colab (T4) by
`deforest.summarize`, `deforest.shift` and `scripts/diagnose_bcos_scratch.py`, rounded to three
decimals. The full files (`SUMMARY.md`, `SHIFT.md`, `shift.csv`, figures, every `preds.csv` and
`model.pt`) are in the Google Drive folder `deforestation-v2/results`. This snapshot exists so the
state of the project is visible in the repository; the README is still filled only by
`scripts/update_readme.py`.

**What is stale.** The four from-scratch B-cos models (`bcos_rgb`, `bcos_rgbn`, `bcos_rgbn_hist`,
`bcos_bitemporal`) were trained with a learning rate that was too high (see section 3). Their rows
below understate them and will change after retraining. Everything else is current.

Setup: real Sentinel-2 imagery, labels from Hansen Global Forest Change, 4 regions, one seed.
"Known region" = spatial block cross-validation (5 folds). "Unseen region" = leave-one-region-out.
AP = average precision; its chance level is the share of patches that lost forest.

## 1. Scores, pooled over the four regions

| Forecast model | AP, known regions (chance 0.132) | AP, unseen regions (chance 0.117) |
|---|---|---|
| cnn_rgbn_hist | 0.559 ± 0.056 | 0.191 ± 0.193 |
| gbm | 0.504 ± 0.033 | 0.224 ± 0.178 |
| bcos_pre_inflate | 0.482 ± 0.027 | 0.154 ± 0.167 |
| bcos_pre_project | 0.475 ± 0.043 | 0.162 ± 0.172 |
| bcos_rgbn_hist *(stale)* | 0.466 ± 0.078 | 0.201 ± 0.183 |
| bcos_pre_rgb | 0.449 ± 0.052 | not run |
| bcos_pre_ndvi | 0.446 ± 0.049 | 0.181 ± 0.175 |
| history_lr | 0.431 ± 0.070 | 0.212 ± 0.174 |
| cnn_rgbn | 0.412 ± 0.061 | not run |
| bcos_rgbn *(stale)* | 0.229 ± 0.025 | 0.093 ± 0.106 |
| bcos_rgb *(stale)* | 0.212 ± 0.032 | 0.118 ± 0.145 |

| Detection model | AP, known regions (chance 0.210) | AP, unseen regions (chance 0.166) |
|---|---|---|
| cnn_bitemporal | 0.918 ± 0.018 | 0.453 ± 0.278 |
| gbm | 0.859 ± 0.025 | 0.535 ± 0.176 |
| bcos_bitemporal *(stale)* | 0.308 ± 0.073 | 0.136 ± 0.146 |

Mean ± SD over splits. The unseen-region mean averages four regions with very different chance
levels, so read section 2 before quoting it.

## 2. Scores inside each region (known regions)

Pooled scores hide the regions. Two of the four have too few loss patches to rank anything.

| Forecast | central_india | northeast_hills | odisha_mining | western_ghats |
|---|---|---|---|---|
| *chance* | *0.012* | *0.348* | *0.034* | *0.073* |
| *loss patches* | *16* | *565* | *25* | *150* |
| cnn_rgbn_hist | 0.070 | 0.591 | 0.103 | 0.268 |
| gbm | 0.063 | 0.563 | 0.053 | 0.260 |
| bcos_pre_inflate | 0.012 | 0.552 | 0.086 | 0.187 |
| bcos_pre_project | 0.029 | 0.546 | 0.036 | 0.185 |
| bcos_rgbn_hist *(stale)* | 0.062 | 0.505 | 0.055 | 0.127 |
| bcos_pre_rgb | 0.012 | 0.543 | 0.033 | 0.200 |
| bcos_pre_ndvi | 0.080 | 0.533 | 0.034 | 0.168 |
| history_lr | 0.085 | 0.501 | 0.085 | 0.166 |
| cnn_rgbn | 0.013 | 0.482 | 0.031 | 0.210 |
| bcos_rgbn *(stale)* | 0.013 | 0.306 | 0.039 | 0.081 |
| bcos_rgb *(stale)* | 0.018 | 0.325 | 0.031 | 0.098 |

| Detection | central_india | northeast_hills | odisha_mining | western_ghats |
|---|---|---|---|---|
| *chance* | *0.016* | *0.503* | *0.050* | *0.094* |
| *loss patches* | *22* | *1056* | *37* | *193* |
| cnn_bitemporal | 0.209 | 0.967 | 0.619 | 0.614 |
| gbm | 0.383 | 0.909 | 0.606 | 0.524 |
| bcos_bitemporal *(stale)* | 0.020 | 0.469 | 0.037 | 0.083 |

## 3. Why from-scratch B-cos was stuck (diagnostic, detection fold 0, 10 epochs, one seed)

| Setting | Loss, epoch 0 | Loss, epoch 9 | Best validation AP | Test AP (chance 0.198) |
|---|---|---|---|---|
| lr 1e-3, no logit temperature (the old default) | 0.98 | 0.693 | 0.41 | 0.41 |
| lr 1e-3, temperature 0.5 | 1.34 | 0.692 | 0.40 | 0.44 |
| lr 1e-3, temperature 0.2 | 2.50 | 0.693 | 0.44 | 0.48 |
| lr 1e-3, temperature 0.1 | 4.73 | 0.657 | 0.41 | 0.42 |
| lr 3e-4 | 0.81 | 0.548 | 0.67 | 0.66 |
| **lr 1e-4 (the new default)** | 0.70 | 0.282 | 0.89 | 0.89 |

At 1e-3 the network collapses to cross-entropy ln 2 after one epoch. A logit temperature does not
help; a ten times smaller step does. This is enough to change the default and not enough to quote:
it is one fold and one seed, and the standard CNN's learning rate has not had the same check.

## 4. Known region against unseen region, same patches

Full table: [`shift_rounded.csv`](shift_rounded.csv). `ap_kept` is the share of above-chance AP that
survives the move; it is blank when there are fewer than 30 loss patches or the known-region AP is
not clearly above chance.

What the table shows, for the two regions with enough loss patches (northeast hills, Western Ghats):

| Forecast model | Northeast hills, known → unseen (kept) | Western Ghats, known → unseen (kept) |
|---|---|---|
| history_lr | 0.501 → 0.508 (–) | 0.166 → 0.166 (100%) |
| gbm | 0.563 → 0.509 (75%) | 0.260 → 0.238 (89%) |
| cnn_rgbn_hist | 0.591 → 0.514 (69%) | 0.268 → 0.154 (42%) |
| bcos_pre_inflate | 0.552 → 0.432 (41%) | 0.187 → 0.139 (58%) |
| bcos_pre_project | 0.546 → 0.444 (49%) | 0.185 → 0.155 (74%) |
| bcos_pre_ndvi | 0.533 → 0.473 (67%) | 0.168 → 0.155 (87%) |

1. **The image models' edge over the past-loss baseline is local.** On unseen regions the baseline
   keeps its score; the CNN and the pretrained B-cos models keep 41 to 87% of their above-chance AP
   and no longer beat it.
2. **Stable reasons, falling accuracy.** Pretrained B-cos models rely on nearly the same input
   channels in known and unseen regions (Jensen-Shannon distance at most 0.03), yet their AP falls.
   Across 24 model-region pairs the correlation between reliance shift and AP drop is -0.39, so
   reliance shift is not an early warning here.
3. **Forecast explanations do not localise future loss even on known regions** (0.8 to 1.5 times
   chance), so "do explanations degrade faster than accuracy" cannot be tested for forecasting.
4. **Detection CNN (Grad-CAM):** localisation 2.7 to 3.8 times chance on known regions, keeping 59 to
   83% on unseen ones, about in step with AP (65 to 85%).

## 5. Limits

One seed. Four regions, two of them with 16 and 25 forecast loss patches. Most intervals are wide
and many include zero (see `shift.csv` on Drive for the block-bootstrap intervals). The standard
CNN's learning rate is untuned. Nothing here is a claim yet; `paper/PLAN.md` lists what a claim needs.
