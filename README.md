
# OASIS-1 Dementia Benchmark: Leakage-Aware MRI + Clinical Baselines

This repository implements a **reproducible benchmark for dementia classification on OASIS-1**, using **subject-level (leakage-aware) splits**, **clinical/morphometric baselines**, and a **processed MRI baseline**.

**Goal:** evaluate whether image models actually add value over structured features under a *correct evaluation setup*.

> **Main claim:**
> Clinical + morphometric features provide a strong baseline signal; MRI models are only meaningful when evaluated directly against this baseline under leakage-aware subject splits.

---

# 🧠 Problem

Can dementia-related signal (`CDR > 0` vs `CDR = 0`) be predicted from structural MRI, and how does MRI modelling compare to tabular clinical/morphometric baselines when **data leakage is explicitly controlled**?

---

# 📊 Dataset

Current benchmark uses a **3-disc subset of OASIS-1** (`disc1–disc3`), with:

* **MR1 only** (one scan per subject)
* **subject-level splits**
* **rows with missing CDR removed**

Example snapshot (disc1 only, for illustration):

| Item              | Value |
| ----------------- | ----: |
| Sessions indexed  |    39 |
| Subjects (MR1)    |    39 |
| Labelled sessions |    25 |
| Dementia cases    |    12 |
| Non-dementia      |    13 |

<p align="center">
  <img src="docs/img/y_counts.png" alt="Label counts" width="340" />
</p>

> Results below are from the **3-disc subset**, not the full dataset.

---

# 📈 Results (cross-validated classical baseline)

Subject-level cross-validation (3 discs):

| Model               | ROC-AUC (mean ± std) | AUC pooled 95% CrI | Bal Acc pooled 95% CrI | Brier |
| ------------------- | -------------------- | ------------------ | ---------------------- | ----- |
| Logistic Regression | **0.80 ± 0.14**      | 0.79 [0.68, 0.89]  | **0.71 [0.59, 0.80]**  | 0.186 |
| Random Forest       | 0.81 ± 0.12          | 0.79 [0.68, 0.88]  | 0.68 [0.56, 0.77]      | 0.183 |
| Gradient Boosting   | 0.78 ± 0.09          | 0.78 [0.67, 0.87]  | 0.67 [0.55, 0.77]      | 0.278 |

Bayesian intervals above are computed from pooled out-of-fold predictions:

* `logistic regression`: sensitivity `0.68 [0.50, 0.82]`, specificity `0.74 [0.59, 0.86]`
* `random forest`: sensitivity `0.71 [0.53, 0.84]`, specificity `0.64 [0.49, 0.78]`
* `gradient boosting`: sensitivity `0.61 [0.44, 0.77]`, specificity `0.72 [0.57, 0.84]`

Single-split snapshot (n=14 test subjects, for illustration only):

| Model               | Input      |       ROC-AUC | Balanced acc | Brier |   ECE |
| ------------------- | ---------- | ------------: | -----------: | ----: | ----: |
| Logistic regression | tabular    |         0.979 |        0.875 | 0.117 | 0.176 |
| Random forest       | tabular    |         1.000 |        0.750 | 0.125 | 0.291 |
| Gradient boosting   | tabular    |         0.833 |        0.812 | 0.199 | 0.208 |
| **2D CNN (Improved)**| **MRI**    |     **0.750** |    **0.500***| 0.247 |     — |
| Fusion              | multimodal |           WIP |          WIP |     — |     — |

> \* *Note: The CNN's balanced accuracy reflects its current "conservative" bias, where it correctly ranks subjects but remains hesitant to cross the 0.5 decision threshold.*

Representative single-split plots from the latest tabular run:

<p align="center">
  <img src="docs/img/roc.png" alt="Tabular ROC" width="250" />
  <img src="docs/img/tab_compare_auc.png" alt="Tabular model comparison" width="250" />
  <img src="docs/img/tab_reliability.png" alt="Tabular reliability" width="250" />
</p>

---

# 🔑 Key takeaway

> **Structured features already carry strong dementia signal.**

* Logistic regression is the most **reliable baseline**:

  * strong ROC-AUC
  * stable balanced accuracy
  * better calibration than tree models
  * strongest pooled Bayesian interval profile
* Random forest shows **perfect ranking on some splits**, but poorer calibration and threshold stability
* Errors concentrate on **older nondemented controls**, indicating:

  * the model learns age/atrophy signal
  * but struggles to separate ageing from pathology
* Dominant features are consistent across models:

  * `nWBV` is the strongest feature
  * `Educ` is the next strongest linear signal
  * `Age` is the clearest secondary tree-based driver

👉 MRI models must beat this baseline **under the same split** to be meaningful.

---

# ⚠️ Important note (small-data behaviour)

* Test sets are small → metrics are **high variance**
* Cross-validation reduces optimism (≈0.98 → ≈0.80 AUC)
* This benchmark is designed to expose **small-data pitfalls**, not hide them

---

# 🧪 The MRI Journey: From Mode Collapse to Biological Signal

Building a deep learning model for dementia on small cohorts (like OASIS-1 discs) is a lesson in humility. Our image baseline underwent a significant evolution to reach its current state.

### Phase 1: The "Lazy" Model (Failure)
Initial attempts with naive 2D CNNs suffered from **mode collapse**. The models would converge to a "global bias," outputting near-constant probabilities (~0.504) for every subject. They learned the prevalence of the disease but failed to see the individual. 

### Phase 2: Finding Focus (The Breakthrough)
To break the bias, we introduced three critical architectural shifts:
1.  **Attention Pooling:** Instead of treating all brain slices as equally important, we added an attention mechanism. This allowed the model to "weigh" different parts of the brain, focusing on regions where atrophy is most diagnostic.
2.  **Regularization & Stability:** Batch Normalization and Dropout were added to force the model to learn robust, generalizable features rather than memorizing noise.
3.  **Spatial Augmentation:** We introduced random rotations and intensity shifts, teaching the model that dementia-related atrophy is structural, not just a matter of pixel intensity.

### Phase 3: The Conservative Predictor (Current Status)
The result is a model that finally **finds the signal**. With an **ROC-AUC of 0.75**, it now successfully ranks dementia patients higher than controls. 

However, it remains a **"Conservative Predictor."** The probabilities are tightly clustered in the 0.48-0.49 range. It "sees" the pathology (hence the strong ranking) but is not yet confident enough to "diagnose" them at a standard 0.5 threshold. This highlights the core challenge of the benchmark: image models are catching up, but they have a steep hill to climb before they can provide the same confident, clear-cut signal as a simple clinical measure of brain volume.

---

# 🔍 Explainability (xAI): What is the model "thinking"?

The benchmark includes a dedicated explainability layer to ensure models are making decisions for the right reasons.

### 1. Tabular: Clinical Logic
Our Logistic Regression baseline is not just a "black box"; its decisions align closely with established neurology:
*   **Atrophy matters:** `nWBV` (Normalized Whole Brain Volume) is the strongest predictor. Its negative coefficient (**-0.80**) confirms that as brain volume decreases, the probability of a dementia diagnosis increases.
*   **Cognitive Reserve:** `Educ` (Education) is the second most influential feature (**-0.71**). This supports the "reserve" hypothesis, where higher education acts as a protective factor against clinical symptoms.
*   **Demographic Bias:** The model identifies `SES` and `Age` as secondary drivers, correctly capturing how social and biological factors interact with pathology.

### 2. MRI: The Reality of "Small Data"
The xAI analysis of the 2D CNN reveals how architectural refinements and augmentation can extract signal even from small cohorts:
*   **Ranking Signal:** Improved runs (using Attention Pooling and Spatial Augmentation) show a significant jump in **ROC-AUC to 0.75**. This indicates the model is successfully ranking dementia cases higher than controls.
*   **Confidence vs. Discrimination:** Despite the strong ranking, the model remains "hesitant," with probabilities compressed in the **0.48–0.49** range. This leads to a **Balanced Accuracy of 0.50** at a fixed 0.5 threshold, highlighting the need for calibrated thresholding or more aggressive training.
*   **Grad-CAM Sanity Checks:** 
    *   The model achieves a **~60-80% brain-mask fraction**.
    *   Low correlation with random models (**rand_cam_corr ~0.15**) suggests it is learning specific structural features rather than global noise, though it still has room to improve compared to the tabular baseline.

### 3. The "Value Add" Test
By comparing these two, we ask: **Does the MRI provide any signal that `nWBV` and `Age` don't already capture?** Currently, the xAI results suggest the tabular features are doing the heavy lifting, setting a high bar for any image-based model to clear.

---

# 🧠 Main contribution
* Bayesian-bootstrap `ROC-AUC` intervals
* credible intervals for `sensitivity`, `specificity`, and `balanced accuracy`

Current fixed small-model smoke run (`coronal3_tiny_mean`, 1 epoch):

* ROC-AUC `0.54 [0.21, 0.83]`
* balanced accuracy `0.50 [0.35, 0.61]`
* `p_std = 0.00030`, indicating near-constant predictions even in the recommended small-model setting

That smoke result is included to show uncertainty-aware reporting, not as a final MRI benchmark.

Recommended current MRI baseline entrypoint, run in a separate shell:

```bash
bash scripts/bench_cnn_best.sh
```

This runs the current fixed small-model candidate:

* coronal slices
* 2.5D (`ch=3`)
* `tiny` architecture
* mean slice pooling
* PyTorch Lightning trainer with checkpointing, early stopping, gradient clipping, and GPU auto-detection

---

# 🧠 Main contribution

* Leakage-aware pipeline:

  ```
  index → manifest → subject splits → baselines → error analysis
  ```
* Dataset-style workflow via **manifest CSV**
* Explicit comparison:

  * tabular vs MRI vs fusion
* Calibration + uncertainty analysis
* Explainability layer:

  * tabular permutation importance
  * standardised logistic coefficients
  * per-subject local linear contributions
  * CNN Grad-CAM overlays with random-model sanity checks
* Focus on **reproducibility over raw performance**

---

# 🧭 Why this matters

Deep learning often appears strong on small medical datasets due to:

* leakage
* duplicated scans
* shortcut learning

This benchmark enforces:

* subject-level evaluation
* transparent cohort definition
* explicit baseline comparison

👉 making results **trustworthy and interpretable**

---

# 🗂️ Data layout

* `data/raw/oasis1/` → immutable downloads
* `data/oasis1/` → extracted discs + manifests
* `data/processed/oasis1/` → derived outputs
* `reports/` → metrics, plots, analysis

---

# ⚙️ Quick start

```bash
uv sync
uv pip install -e .
```

---

# 🔁 Pipeline

## 1. Prepare dataset

```bash
bash scripts/prep_oasis1.sh
```

## 2. Tabular benchmark

```bash
bash scripts/bench_tab.sh
```

## 3. MRI baseline

```bash
bash scripts/bench_cnn_best.sh
```

## 4. Explainability

```bash
bash scripts/xai.sh
```

Outputs:

* `reports/xai/tab/summary.md`
* `reports/xai/tab/logreg_coef.png`
* `reports/xai/tab/logreg_perm.png`
* `reports/xai/cnn/summary.md`
* `reports/xai/cnn/*.png`
