# Paper plan: do a forest-loss model's reasons survive a move to an unseen region?

Working title: **"Right for the Same Reasons? Explanations of an Inherently Interpretable Forest-Loss Model under Regional Shift"**

This file is the plan of record for turning the repository into a paper. Nothing here is a result.
Results go into `results/` and reach the README only through `scripts/update_readme.py`.

## 1. The claim we are testing

Distribution-shift work measures how much *accuracy* a model loses on a new region. Interpretability
work checks explanations on data that looks like the training data. We test the combination: when a
forest-loss model is applied to a biome it never saw, **do its explanations degrade faster than its
accuracy, and can a label-free change in what the model relies on warn us before labels arrive?**

B-cos networks make this testable, because their explanation is the exact computation
(contributions sum to the logit), so a change in the explanation is a change in the model's reasoning,
not an artefact of a post-hoc method. Grad-CAM on a black-box CNN is the comparison.

## 2. Hypotheses, stated before the runs

| | Hypothesis | Measured by (`deforest/shift.py`) | What would count against it |
|---|---|---|---|
| H1 | Explanations keep a smaller share of their above-chance quality than AP does on an unseen region | localisation retention minus AP retention, paired block bootstrap, per model and region | interval includes 0 or is positive in most regions |
| H2 | The effect is causal, not only spatial | deletion gap ID vs OOD | deletion gap unchanged while localisation falls (then localisation is measuring label geometry, not reasoning) |
| H3 | Reliance shift (JS distance between channel-share vectors, no labels) tracks the AP drop | Spearman across model-region pairs | no monotone relation once there are enough pairs |
| H4 | ImageNet-pretrained B-cos transfers its reasons better than B-cos from scratch | H1 measures, `bcos_pre_*` vs `bcos_rgbn` | no difference |

Negative answers are publishable here. "Exact explanations stay faithful under shift even when accuracy
falls" would be just as useful to someone deciding whether to trust a map of a new region.

## 3. What exists, what is missing

| Piece | State |
|---|---|
| Independent labels, non-overlapping patches, block CV with buffer, LORO splits | done (`deforest/data`) |
| 11 forecast and 4 detection models, baselines every model must beat | done (`deforest/experiments.py`) |
| Completeness, localisation, randomisation check, band ablation | done (`deforest/explain.py`) |
| Per-patch explanation records, deletion gap, ID-vs-OOD report with paired bootstrap | done in this change (`deforest/shift.py`), tested on synthetic data only |
| Block runs for all 5 folds | check `results/` on Drive: the fast profile runs 3 of 5 folds, and the comparison only covers patches in folds that were run |
| LORO runs for the neural models | **not run yet: this is the experiment the paper depends on** |
| More than four regions | missing; see section 5 |
| Multiple seeds | missing; one seed per run today |

## 4. Run order (Colab T4, results on Drive)

1. `python -m deforest.shift $WORK/results --data $WORK/data/v2 --backfill`
   adds per-patch records to the block runs that are already trained (no retraining).
2. `python -m deforest.experiments ... --scheme block --folds fold3 fold4 --only priority`
   so every patch has an ID prediction.
3. `python -m deforest.experiments ... --scheme loro --only priority`
   the OOD runs. Four regions x 9 neural models.
4. `python -m deforest.shift $WORK/results` then `python scripts/update_readme.py`.
5. Read `results/SHIFT.md` against section 2 and write down what held and what did not, before
   changing anything.

Section 4b of `notebooks/03_v2_colab.ipynb` runs steps 1 to 4.

## 5. What a reviewer will ask, and the work that answers it

| Likely objection | Planned answer | Cost |
|---|---|---|
| "Four regions is four data points." | Add 4 to 8 more 30 km boxes with `scripts/scan_hansen_aoi.py` (e.g. Eastern Ghats, Arunachal, Jharkhand, Bastar, Nilgiris), so H3 has 8 to 12 pairs per model | about 15 min of download per region, then reruns |
| "One seed." | 3 seeds for the four headline models; report seed SD next to fold SD | 3x compute on 4 models |
| "Localisation needs Hansen masks, which are 30 m and noisy." | Deletion gap needs no mask; report both and where they disagree | done |
| "Mean-fill deletion creates off-distribution inputs." | Add a blur-fill variant and report whether the ranking of models changes | small code change |
| "Is the reliance shift just a change in class balance?" | Compute channel shares on a label-balanced sample and on all patches; report both | small |
| "Why not a geospatial foundation model?" | Stretch: one pretrained EO backbone with a linear probe as an accuracy reference. Not B-cos, so accuracy only | medium |
| "Does this matter for use?" | Tie to the risk tiers: do High-tier hit rates hold in the unseen region (`risk_levels.py` on LORO predictions) | small |

## 6. Figures and tables

1. Map of regions and one example per region: RGB, Hansen loss, contribution map from the ID model, contribution map from the OOD model, same patch.
2. Main table: AP, localisation and deletion gap, ID and OOD, per model (mean over regions, block-bootstrap CI).
3. Retention scatter (`shift_forecast.png`): accuracy kept vs explanation kept, one point per model and region.
4. Channel reliance bars, ID vs OOD, per region, for `bcos_rgbn_hist`.
5. Reliance shift vs AP drop (H3).
6. Appendix: v1-to-v2 audit (the 99% accuracy that was label leakage), completeness errors, randomisation check.

## 7. Outline (4 pages + references for a workshop; 8 for a full version)

1. Introduction: a risk map of an unseen forest comes with reasons; can they be trusted?
2. Related work: B-cos and B-cosification; sanity checks for saliency; spatial CV in ecology; ForestNet and deforestation forecasting; explanation robustness under shift.
3. Setup: data, labels, splits, models (short; the README has the detail).
4. Measuring explanations under shift: the four measures and the paired design.
5. Results: H1 to H4 in order.
6. Limits: Hansen labels, number of regions, one forecast date, patch-level labels.
7. What this means for anyone shipping a forest-risk map.

## 8. Where to send it

Deadlines move every year; confirm on the workshop page before planning around one.

| Venue | Fit | Timing seen in recent editions |
|---|---|---|
| Tackling Climate Change with Machine Learning (Climate Change AI workshop at ICLR and NeurIPS) | best first target: short papers, a Proposals track for early work, and a mentorship programme for first-time authors | ICLR edition closed early February in 2025; NeurIPS edition closes around late summer |
| EarthVision (CVPR workshop) | remote-sensing audience, archival proceedings, 8 pages | early March in 2025; CVPR 2027 workshops are 20 to 21 June 2027 |
| An explainable-AI workshop at CVPR, ICCV or NeurIPS | if H1/H2 turn out to be the strongest part | varies |
| arXiv preprint | post once the main table is final, whatever the venue | any time |

Realistic sequence: results table by mid-December 2026, more regions and seeds in January, 4-page
submission to the first open call in early 2027, full version after reviews.

## 9. Authorship and credit

Kavya Sajjit is first author. Ask Dr. Nilesh Rathod (DJSCE) whether he will advise and co-author.
Coding assistance from an AI model must be disclosed the way the venue requires; check its policy.
Every number must be reproducible from a command in the README by the author herself, and she should
be able to explain each measure in section 2 without notes before submitting.

## 10. Definition of done for a submission

- [ ] 5 block folds and 4+ LORO regions for every headline model
- [ ] 3 seeds for headline models
- [ ] at least 8 regions
- [ ] `results/SHIFT.md` regenerated from scratch on a clean clone
- [ ] every hypothesis in section 2 marked supported, not supported or inconclusive, with the interval
- [ ] limitations section written before the abstract
- [ ] `pytest -q tests/` passes

## 11. First full run (14 November 2026, Colab T4, one seed, four regions)

Numbers are in `results/SHIFT.md` on Drive. Read against section 2:

| | Status | What the run showed |
|---|---|---|
| H1 | **cannot be tested for forecasting yet** | In-distribution localisation is already at chance (0.8 to 1.2x) for the forecast models, so there is no explanation quality to lose. For the detection CNN (Grad-CAM, 2 to 13x chance) localisation fell roughly in step with AP, not faster. |
| H2 | inconclusive | Deletion gaps of the B-cos forecast models are near zero in both settings: image pixels barely move the prediction. |
| H3 | **not supported** | Spearman between reliance shift and AP drop = -0.32 over 24 model-region pairs. |
| H4 | partly | Pretrained B-cos keeps almost the same channel reliance on unseen regions (JS up to 0.03, against 0.04 to 0.17 from scratch), yet its AP still falls. Stable reasons did not mean stable accuracy. |

Findings that were not hypotheses:

- On unseen regions the past-loss baseline keeps all of its skill, while the CNN and pretrained B-cos models keep roughly 45 to 90% of theirs and no longer beat it.
- Pooled AP hides the regions. In central India (16 loss patches) and Odisha (25) every forecast model is near that region's chance level; the pooled figure is carried by the northeast hills and the Western Ghats. `summarize.py` now prints AP inside each region.
- `bcos_bitemporal` does not train (loss stays at ln 2). Fix or drop before any write-up.
- Only folds 3 and 4 of the block runs have saved weights, so in-distribution explanation records cover two of five folds. Retrain folds 0 to 2.

Consequence for the paper: the title question has a different answer from the one planned for. The
candidate framing is now "interpretable does not mean transferable: exact explanations stay stable
while forecasts of forest loss stop beating a past-loss baseline on unseen regions". This needs more
regions with enough loss patches (at least 100 each) and three seeds before it is a claim.
