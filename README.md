# Deforestation Prediction in India: Sentinel-2 CNNs & B-cos Interpretability

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2205.10268-b31b1b.svg)](https://arxiv.org/abs/2205.10268)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A computer vision project evaluating standard CNNs and inherently interpretable **B-cos networks** ([Böhle et al., CVPR 2022](https://arxiv.org/abs/2205.10268)) for deforestation classification on Sentinel-2 satellite imagery across India, alongside state-level tabular risk modeling.

---

## What It Does

1. **Satellite Patch Classification**: Classifies 128×128 Sentinel-2 satellite image patches into **Intact Forest** vs. **Deforested / Degraded Land**.
2. **Inherent Interpretability with B-cos**: Replaces heuristic post-hoc explainers (such as Grad-CAM or LIME) with B-cos convolutions where the prediction is an **exact linear sum** of input features:
   $$\text{Output}_c = \sum_{ch, h, w} \tilde{x}_{ch, h, w} \cdot W_{ch, h, w} + b$$
   This yields faithful spatial contribution heatmaps and decoded color evidence without approximation error.
3. **Regional Tabular Forecasting**: Uses historical Global Forest Watch (GFW) subnational data (2001–2023) to model state-level deforestation trends with a Random Forest baseline.

---

## Data

The satellite dataset consists of **886 image patches (128×128)** extracted from cloud-free Sentinel-2 L2A composites over four major Indian forest biomes:

| Region | Forest Biome / Terrain | Granule ID | Total Patches |
| :--- | :--- | :--- | :--- |
| **Western Ghats** | Tropical evergreen rainforest (Karnataka/Kerala) | `S2B_MSIL2A_20230421..._T43PEP` | 225 |
| **Odisha Mining Belt** | Sal forest & open-cast iron ore mining (Keonjhar/Mayurbhanj) | `S2A_MSIL2A_20230427..._T45QUE` | 225 |
| **Northeast India** | Subtropical hill forest & shifting cultivation (Assam/Arunachal) | `S2A_MSIL2A_20230418..._T46REQ` | 211 |
| **Central India Corridor** | Mixed deciduous & teak forest tracts (Balaghat/Kanha) | `S2B_MSIL2A_20230418..._T44QMK` | 225 |

---

## Dataset Splits

The 886 patches are partitioned into train, validation, and test splits (70% / 15% / 15% random split):

| Split | Intact Forest | Deforested / Degraded | Total |
| :--- | :---: | :---: | :---: |
| **Train** | 92 | 547 | **639** |
| **Validation** | 11 | 108 | **119** |
| **Test** | 24 | 104 | **128** |
| **Total** | **127** | **759** | **886** |

---

## Results Table

Evaluated on the held-out test split of 128 Sentinel-2 patches:

| Model Architecture | Input Channels | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Linear Fidelity Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard Baseline CNN** | 3 (RGB) | **99.22%** | **0.990** | **1.000** | **0.995** | **1.000** | N/A (Black-box) |
| **B-cos Deforestation CNN ($B=2$)** | 6 ($[x, 1-x]$) | **96.09%** | **0.971** | **0.981** | **0.976** | **0.990** | **$< 1 \times 10^{-7}$ (Exact)** |

### Confusion Matrix on Test Set

- **Standard Baseline CNN**:
  - Intact Forest: 23 correct, 1 false positive
  - Deforested: 104 correct, 0 false negatives
- **B-cos Deforestation CNN ($B=2$)**:
  - Intact Forest: 21 correct, 3 false positives
  - Deforested: 102 correct, 2 false negatives

---

## Visual Explanations

In explanation mode, the B-cos model directly yields dynamic linear weights $W_{1 \rightarrow L} = \nabla_{\tilde{x}} f(x)$:

![B-cos Explanation Gallery](models/bcos_explanation_gallery.png)

1. **Input Satellite Patch**: True-color Sentinel-2 image.
2. **Spatial Contribution Map**: Positive weights (red) highlight clear-cut land, bare soil, and access paths that pushed the model toward predicting deforestation.
3. **Decoded Attribution**: Colors indicate aligned spectral patterns supporting the decision.

---

## Limitations

1. **Near-Infrared (NIR) Band Unused**: The current model utilizes only RGB channels ($[R, G, B]$ expanded to 6 channels via `AddInverse`). Sentinel-2 Band 8 (NIR) is critical for standard vegetation indices (NDVI) and canopy moisture detection. Incorporating NIR would require extending `AddInverse` from 6 to 8 channels ($[R, G, B, \text{NIR}, 1-R, 1-G, 1-B, 1-\text{NIR}]$).
2. **Small Test Set**: The test set contains 128 patches sampled from four satellite scenes. While diverse across biomes, it is not large enough to represent seasonal phenology, cloud shadow artifacts across all of India, or severe atmospheric haze.
3. **Heuristic Labeling Method**: Patch labels were generated using spectral index thresholding (Green Leaf Index and soil reflectance ratio) rather than ground-truth field surveys or high-resolution hand-annotated polygons. While effective for separating dense canopy from barren/cleared land, boundary patches with shifting agriculture or sparse scrub can be noisy.
4. **Spatial Autocorrelation in Split**: The train/val/test split was performed randomly across candidate patches extracted from four scenes rather than by holding out entire geographic tiles or regions. This creates potential spatial correlation between adjacent patches across splits.

---

## Quickstart

### Installation

```bash
git clone https://github.com/kavya-sajjit/deforestation-prediction-india.git
cd deforestation-prediction-india
pip install -r requirements.txt
```

### Reproduce Data and Training

```bash
# 1. Download Sentinel-2 scenes and extract patches
python src/data/stac_downloader.py --output_dir data/satellite/india_sentinel2

# 2. Train B-cos CNN (or standard baseline)
python train_cnn.py --model bcos --epochs 12 --batch_size 32
python train_cnn.py --model baseline --epochs 10

# 3. Run inference with B-cos explanations
python run_inference.py --random_test
```

### Interactive Notebooks

- [`notebooks/01_explore_and_clean.ipynb`](notebooks/01_explore_and_clean.ipynb): Tabular time-series modeling of GFW tree cover loss.
- [`notebooks/02_satellite_deforestation_cnn_bcos.ipynb`](notebooks/02_satellite_deforestation_cnn_bcos.ipynb): Step-by-step walkthrough of satellite patch exploration, training, metrics, and B-cos interpretability.

---

## References

- Böhle, M., Fritz, M., & Schiele, B. (2022). *B-cos Networks: Alignment is All You Need for Interpretability*. CVPR 2022. [arXiv:2205.10268](https://arxiv.org/abs/2205.10268).
- European Space Agency (ESA). *Sentinel-2 MSI: MultiSpectral Instrument*, accessed via Microsoft Planetary Computer STAC.
- Global Forest Watch (GFW). *Tree Cover Loss & Carbon Emissions in India (2001–2023)*.
