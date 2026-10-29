# Forecasting and Explaining Forest Loss in India with B-cos Networks

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c) ![License: MIT](https://img.shields.io/badge/license-MIT-green)

This project predicts **where forest is likely to be cleared next** and **detects where it already was**, using Sentinel-2 satellite imagery over four Indian forest biomes. Every prediction comes with an explanation that is exact by construction. It uses B-cos networks (Böhle et al., CVPR 2022), whose output is a dynamic linear function of the input pixels.

| Task | Input | Question | Ground truth |
|---|---|---|---|
| **Forecast** | Sentinel-2, Jan–Apr 2019 (R, G, B, NIR) + where forest was lost before 2019 | Will at least 2% of this 640 m forest patch be lost in **2020–2023**? | Hansen et al. Global Forest Change v1.11 |
| **Detect** | Sentinel-2, early 2019 **and** early 2024 | Was at least 2% of this forest patch lost between the two dates? | Hansen GFC v1.11 |

The research question behind it: **which spectral bands does the model need, and in which regions does dropping one hurt most?** This is a data-minimisation question: does an average score hide regions that pay for a smaller input?

---

## Contents
1. [Data and labels](#data-and-labels)
2. [Models](#models)
3. [How the evaluation avoids fooling itself](#how-the-evaluation-avoids-fooling-itself)
4. [Results](#results)
5. [Checking the explanations](#checking-the-explanations)
6. [Reproduce](#reproduce)
7. [Limitations](#limitations)
8. [What changed from v1](#what-changed-from-v1)
9. [References](#references)

---

## Data and labels

**Imagery.** Sentinel-2 L2A from Microsoft Planetary Computer. Each time window becomes a per-pixel **median composite** of the least-cloudy scenes. Clouds, shadows, cirrus and no-data are removed with the Scene Classification Layer. Scenes after 25 Jan 2022 (processing baseline 04.00) have their +1000 offset removed, so 2019 and 2024 reflectances are comparable. Bands are B04, B03, B02 and B08 at 10 m, scaled to [0, 1] with fixed (not data-dependent) constants.

**Labels come from an independent source, not from the imagery.** Forest is ≥30% canopy cover in 2000 (Hansen `treecover2000`), minus anything lost before 2019. Loss and its year come from Hansen `lossyear`. Patches are 64 × 64 px (640 m), **do not overlap**, and must be ≥30% forest at the start of 2019.

| Rule | Value | Why |
|---|---|---|
| Positive | ≥2% of the patch lost in the target years | about 0.8 ha, well above single-pixel noise |
| Negative | **no** loss in the target years | clean contrast |
| Ambiguous | anything in between → **excluded** | not forced into a class |
| Forecast: 2019 loss | patches with >0.5% loss in 2019 excluded | the early-2019 composite might already show it |
| History (forecast input) | loss 2001–2018 only | known before the forecast date, so no leakage from the future |

Regions (30 km × 30 km each; boxes in [`deforest/config.py`](deforest/config.py)):

| Region | Biome |
|---|---|
| Western Ghats (Kodagu–Wayanad) | Tropical evergreen / moist deciduous |
| Odisha (Keonjhar) | Sal forest with open-cast iron-ore mining |
| Northeast (Mizoram) | Subtropical hill forest with shifting cultivation |
| Central India (Balaghat) | Mixed deciduous and teak |

Patch and class counts per region are written to `data/v2/dataset_summary.json` and repeated in the results section.

## Models

All B-cos models use the **official B-cos v2 code** ([B-cos/B-cos-v2](https://github.com/B-cos/B-cos-v2)). For a B-cos network the decision logit is exactly

$$z_{\text{loss}}(x) - z_{\text{no loss}}(x) \;=\; \sum_{c,h,w} \mathbf{W}(x)_{c,h,w}\; e_{c,h,w}, \qquad e = [x,\; 1-x]$$

with **no bias term**: the B-cos layers have no biases, and the logit layer's constant offset is set to zero here (it would cancel in the difference anyway). The contribution map $\mathbf{W}(x) \odot e$ therefore accounts for the entire decision. In the maps we show, the past-loss input channels are left out so the map shows what the *imagery* contributes; their share is reported separately. This is a property of the architecture, not an experimental finding. The tests check it numerically for every variant.

| Name | What it is |
|---|---|
| `history_lr` | Logistic regression on past-loss share, past-loss density within 2.5 km and forest share. Deforestation spreads from existing frontiers, so **every model must beat this.** |
| `gbm` | Gradient-boosted trees on hand-made spectral features (band means and spreads, NDVI) + history (+ change in NDVI for detection) |
| `cnn_*` | Standard ResNet-18, a black box, trained from scratch |
| `bcos_rgb`, `bcos_rgbn` | B-cos ResNet-18 from scratch; `rgbn` takes the 8-channel input `[R,G,B,NIR,1−R,1−G,1−B,1−NIR]` |
| `bcos_rgbn_hist` | as above + the past-loss channels: the main forecasting model |
| `bcos_pre_rgb` | ImageNet-pretrained B-cos ResNet-18, RGB only: the reference for the three NIR adaptations |
| `bcos_pre_inflate` | pretrained; first layer inflated to 8 channels, NIR filters initialised from the mean RGB filter |
| `bcos_pre_project` | pretrained; learned 4→3 band projection whose rows sum to 1, so that `P(1−x) = 1 − Px` and the model stays exactly linear |
| `bcos_pre_ndvi` | pretrained RGB trunk; NDVI enters after layer 1. NDVI is a ratio, **not** linear in the pixels, so it is a separate input and its contribution is reported separately |
| `*_bitemporal` | detection: both dates stacked (16 encoded channels) |

**Why pretrained models need adapting at all:** ImageNet B-cos weights expect 6 channels (RGB + inverse), so a near-infrared band has nowhere to go. The from-scratch models do not have this problem. They are the fourth option, and the comparison shows whether ImageNet pretraining is worth the adaptation.

## How the evaluation avoids fooling itself

* **Spatial block cross-validation.** Patches are grouped into 5.12 km blocks and blocks are assigned to 5 folds. Training patches touching a test block are dropped (a one-patch buffer), so test pixels never appear in training and neighbouring patches cannot leak. `check_no_leak()` asserts this before every run.
* **Leave-one-region-out.** Train on three biomes and test on the fourth. This is the harder test of whether the model transfers.
* **Imbalance-aware metrics.** The headline is **average precision (PR-AUC)** next to its chance level (the positive rate), plus F1, precision, recall, balanced accuracy and MCC. The decision threshold is chosen on the validation fold and frozen before testing. Accuracy is reported but never headlined.
* **Risk-map metric.** *Loss in top-10% risk* is the share of all actual loss found in the 10% of patches the model ranks riskiest. That is the area an NGO could realistically check. Chance level is 10%.
* **Uncertainty.** Mean ± SD across folds, plus a 95% CI from a **block bootstrap** (resampling whole 5 km blocks, because neighbouring patches are not independent).
* **Baselines.** Each neural model is compared with `history_lr` and `gbm` on identical splits.

## Results

> Every number below is produced by `python -m deforest.experiments` and pasted by `python scripts/update_readme.py`. Nothing is typed by hand.

<!-- RESULTS:START -->
*Not run yet. Run the commands in [Reproduce](#reproduce), then `python scripts/update_readme.py`.*
<!-- RESULTS:END -->

## Checking the explanations

Pretty heatmaps are not evidence. For every model, on the test patches that really lost forest:

1. **Completeness.** B-cos contributions must sum to the logit difference. The maximum error is reported.
2. **Localisation.** What share of the positive evidence lies on pixels Hansen marks as lost, divided by the share of the patch that was lost (chance)? The same score is computed for **Grad-CAM on the black-box CNN** for comparison. For the forecast task, loss has not happened yet in the input image, so read this as "evidence lies where loss later occurs".
3. **Randomisation sanity check** (Adebayo et al., 2018). The Spearman correlation between maps from the trained model and from the same architecture with random weights. It should be low; if it isn't, the map reflects the architecture, not what was learned.
4. **Band reliance.** The share of |contribution| per input band, and the drop in AP per region when one band is replaced by its training mean.

## Reproduce

```bash
git clone https://github.com/kavya-sajjit/deforestation-prediction-india.git
cd deforestation-prediction-india
pip install -r requirements.txt

pytest -q tests/                              # offline checks on synthetic data (~1-2 min)
python -m deforest.data.build --out data/v2   # download Sentinel-2 + Hansen, build patches
python -m deforest.experiments --data data/v2 --out results --profile core            # 5-fold spatial CV
python -m deforest.experiments --data data/v2 --out results --profile core --scheme loro
python -m deforest.risk_map --data data/v2 --region odisha_mining \
    --model results/forecast/loro/bcos_rgbn_hist/test_odisha_mining/model.pt --out maps
python scripts/update_readme.py
```

On a free Colab T4 GPU, `notebooks/03_v2_colab.ipynb` runs everything. Runs that have finished are skipped on restart. To move a region box onto an area with more loss, use `python scripts/scan_hansen_aoi.py --bbox ...`, and record the new box here.

## Limitations

* **Hansen labels are not field truth.** They are 30 m (our pixels are 10 m). They count rotational shifting cultivation and plantation harvest as "loss", and they miss gradual degradation. The model learns Hansen's definition of loss.
* **Four 30 km regions** are a small sample of India. Leave-one-region-out results show how far that goes.
* **One forecast date.** The forecast uses a single early-2019 composite and history. Roads, settlements, protected-area status and mining leases are known drivers that are not yet inputs.
* **Patch-level labels** (640 m) cannot say which hectare inside the patch will go.
* **Explanation localisation** for forecasting measures association with *future* loss, not a cause.

## What changed from v1

v1 (Oct 2026) reported 99.2% accuracy for a standard CNN and 96.1% for B-cos. Those numbers did not measure what they appeared to:

| v1 problem | Effect | v2 fix |
|---|---|---|
| Labels came from thresholds on RGB indices (Green Leaf Index, red/green ratio) | The CNN only had to relearn a formula computed from its own input: circular | Labels from Hansen GFC, an independent product |
| 128 px patches at stride 64 (50% overlap), split at random | Test pixels appeared in training | Non-overlapping patches, spatial block CV with buffer, leave-one-region-out |
| One date; "deforested" meant "not forest now" | Farmland and towns counted as deforestation | Forecast from history + T0; detection from two dates; only patches that were forest at T0 |
| 127 intact vs 759 "deforested" patches, accuracy headlined | Accuracy rewards the majority class | AP vs chance, F1, MCC, balanced accuracy; class-weighted loss |
| NIR (B08) downloaded but dropped | No vegetation-sensitive band | RGB vs RGB+NIR compared directly; three adaptations for pretrained models |
| README formula had a bias term; "linear fidelity error" shown as a result | Misstated B-cos | Formula corrected; completeness is a unit test, not a result |

The v1 code and notebooks remain in `src/`, `train_cnn.py` and `notebooks/` for reference. The state-level Random Forest on GFW tables (`notebooks/01_explore_and_clean.ipynb`) is unchanged.

## References

* Böhle, M., Fritz, M., Schiele, B. (2022). *B-cos Networks: Alignment is All We Need for Interpretability.* CVPR. arXiv:2205.10268
* Böhle, M., Singh, N., Fritz, M., Schiele, B. (2024). *B-cos Alignment for Inherently Interpretable CNNs and Vision Transformers.* IEEE TPAMI.
* Arya, S., Rao, S., Böhle, M., Schiele, B. (2024). *B-cosification: Transforming Deep Neural Networks to be Inherently Interpretable.* NeurIPS. arXiv:2411.00715
* Hansen, M. C. et al. (2013). *High-Resolution Global Maps of 21st-Century Forest Cover Change.* Science 342, 850–853.
* Roberts, D. R. et al. (2017). *Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure.* Ecography 40, 913–929.
* Adebayo, J. et al. (2018). *Sanity Checks for Saliency Maps.* NeurIPS.
* Irvin, J. et al. (2020). *ForestNet: Classifying Drivers of Deforestation in Indonesia using Deep Learning on Satellite Imagery.* NeurIPS Workshop on Tackling Climate Change with ML.
* Biega, A. J. et al. (2020). *Operationalizing the Legal Principle of Data Minimization for Personalization.* SIGIR.
* ESA Sentinel-2 MSI L2A, via Microsoft Planetary Computer STAC.
