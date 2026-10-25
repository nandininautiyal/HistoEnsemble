# Patient-Wise Robustness Benchmark on BreaKHis

Benign vs. malignant classification on **BreaKHis** with ImageNet-pretrained CNNs, evaluated strictly at the **patient level**, and used to study four questions that single-split accuracy tables cannot answer: how stable model rankings are, whether augmentation-induced stain robustness transfers across shift families, whether validation-tuned sensitivity targets hold on new patients, and whether stain-perturbation disagreement is a useful abstention signal.

> **Status: preliminary.** All numbers below come from one full run (seed 42, 5 patient-grouped folds, 2 architectures, 4 augmentations = 40 trained models). They are within-dataset results on a single-lab dataset. See [Limitations](#limitations) before citing anything.

---

## Why this project

Many published BreaKHis results split at the *image* level, so images from the same patient land in both train and test, which inflates accuracy. Here every patient appears in exactly one test fold. Under that protocol the headline numbers are far lower than commonly reported ones (accuracy 0.82–0.88, AUC 0.90–0.94), and differences between configurations are small relative to fold-to-fold variation. The project therefore focuses on *measuring that uncertainty properly* and on robustness and reliability questions:

1. **Do rankings hold up across patient folds?**
2. **Does robustness from one augmentation family transfer to another, including to a shift family never used in training?**
3. **If you tune a threshold for "≥ 95 % sensitivity", does it deliver that on new patients?**
4. **Can disagreement across stain-perturbed views flag the model's errors better than max-softmax?**

HED jitter, Reinhard colour transfer, conformal risk control, test-time augmentation and FGSM/PGD are all existing techniques. The contribution is the controlled, patient-level, cross-validated comparison with patient-clustered statistics, not a new method.

## Dataset

[BreaKHis](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/) (Spanhol et al., 2016): 7,909 H&E images from 82 patients (2,480 benign, 5,429 malignant) at 40×, 100×, 200× and 400×. All magnifications are pooled. On Kaggle, attach `ambarish/breakhis`.

Patient keys are built as `<id>|<subtype>`. Parsing ids alone yields 81 patients because one id (`14-13412`) is shared by two subtypes; adding the subtype recovers the documented 82 and can only split a colliding id, never leak a patient. The notebook prints the patient count and any collisions.

## Repository contents

| File | Description |
|------|-------------|
| `breakhis-v3-robustness-crc-abstention.ipynb` | Main notebook (training, shift benchmark, bootstrap, threshold analysis, abstention, plots) |
| `breakhis-v3-run.ipynb` | Executed copy with the outputs reported below |
| `breast-cancer-detection-breakhis-run-1.ipynb`, `…-run-2.ipynb` | Earlier iterations, superseded; kept for reference |
| `figures/` | Copy the PNGs produced by the notebook here (see [Outputs](#outputs)) |

## Method overview

- **Splits:** patient-grouped, class-stratified 5-fold (`StratifiedGroupKFold`). Inside each training portion, ~20 % of patients form a validation set used only for early stopping, temperature scaling and threshold tuning. Test patients are never seen during training or model selection.
- **Models (default):** DenseNet121, EfficientNet-B0. Others available in the config: ResNet50, ResNeXt50, MobileNetV2, VGG16, AlexNet, ConvNeXt-Tiny, ShuffleNetV2. ImageNet weights, new 2-class head, full fine-tuning.
- **Training:** AdamW (backbone 5e-5, head 5e-4, weight decay 1e-2), cosine schedule, label smoothing 0.1, class-weighted loss, mixed precision, ≤ 12 epochs with early stopping (patience 3) on validation AUC. Images are squashed to 224×224 and cached as a uint8 tensor; all augmentation and shift operations run on the GPU.
- **Augmentation tiers:**
  - `geom`: random flips + rotation ±30° (control)
  - `jitter`: `geom` + ColorJitter (brightness/contrast/saturation 0.3, hue 0.1)
  - `hed`: `geom` + HED stain jitter (σ = 0.1)
  - `both`: `geom` + ColorJitter + HED
  - (`basic`, resize only, is available but not in the default grid)
- **Test-time shift benchmark (same test images for every model):**
  - **HED**: per-image scale/offset of haematoxylin/eosin/DAB concentrations, severities 0–0.4
  - **ColorJitter**: brightness/contrast/saturation shift, severities 0–0.4
  - **Reinhard** (*independent family, never used in training*): lαβ statistics interpolated toward those of a random *other* training-set image; 0 = unchanged, 1 = full transfer
- **Statistics:** predictions pooled across folds; patients resampled with replacement (500 resamples) for 95 % CIs on paired AUC / balanced-accuracy differences; Kendall τ between folds' rankings.
- **Threshold analysis:** for target miss rate α, three ways of choosing the threshold, all scored by the *mean per-patient miss rate on held-out malignant patients*: validation-tuned; naive pooled out-of-fold quantile; and **patient-level conformal risk control** (Angelopoulos et al., 2022), leave-one-patient-out on pooled out-of-fold predictions.
- **Abstention:** uncertainty from max-softmax (MSP) vs. spread/entropy over K = 6 HED-perturbed views vs. a dihedral-TTA control, on clean and shifted inputs. The base prediction is always the single clean view, so only the ranking quality differs.
- **Adversarial (sanity check only):** FGSM and PGD (7 steps), ε ∈ {1, 2}/255, on 150 test images of fold 0.
- **Ensemble:** soft vote over the best augmentation per architecture, chosen by validation AUC (never test).

## Running on Kaggle

1. Create a notebook and upload `breakhis-v3-robustness-crc-abstention.ipynb`.
2. **Add Input** → attach BreaKHis (`ambarish/breakhis`).
3. Session options: GPU, **Internet On** (needed once to download pretrained weights).
4. Set `CFG["QUICK_TEST"] = True` for a short smoke test, then set it back to `False`.
5. Run all (or **Save Version → Save & Run All**) for the full experiment.

Key settings are in the config cell (`CFG`): `MODELS`, `AUGS`, `SEEDS`, `FOLDS_TO_RUN`, `EPOCHS`, `SHIFT`, `SHIFT_REF`, `ALPHAS`, `N_BOOT`, `EXTERNAL_DIR`.

**Runtime:** per-run times printed in the log were roughly 3.6–8 min (EfficientNet-B0 at the low end, DenseNet121 at the high end) including the post-hoc shift and abstention evaluation, so the default 40-run grid is on the order of 4 hours. This is an estimate from those logs, not a separate benchmark.

**Resuming / splitting across sessions:** every finished run is pickled to `/kaggle/working/runs/` and skipped on re-run (`RESUME=True`). Run subsets of folds (e.g. `FOLDS_TO_RUN=[0,1]` then `[2,3,4]`), carrying the `runs/` folder between sessions as a Kaggle dataset. All analysis sections (9 onward) need only the pickles, not a GPU.

**Optional external set:** set `CFG["EXTERNAL_DIR"]` to a folder with `benign/` and `malignant/` subfolders to evaluate every model on it. Without it, no cross-site claim can be made.

## Outputs

Written to `/kaggle/working`:

- `results_per_run.csv`, `shift_curves.csv`, `bootstrap_paired_differences.csv`, `bootstrap_absolute.csv`
- `threshold_transfer.csv`, `abstention_per_run.csv`, `adversarial_sanity.csv` (and `external_eval.csv` if an external set is configured)
- Figures: `shift_curves.png`, `transfer_matrix.png`, `ranking_stability.png`, `threshold_transfer.png`, `abstention_curves.png`
- `runs/*.pkl`: per-run predictions, shifted predictions and abstention signals

## Results (preliminary: seed 42, 5 folds)

### Main performance

Mean ± sd over the 5 test folds (about 16 patients each). The sd is across folds, not a confidence interval.

| Model | Aug | Accuracy | AUC | Recall | Specificity | F1 |
|-------|-----|----------|-----|--------|-------------|----|
| DenseNet121 | geom | 0.848 ± 0.059 | 0.919 ± 0.065 | 0.852 | 0.830 ± 0.171 | 0.884 |
| DenseNet121 | jitter | **0.877 ± 0.062** | **0.938 ± 0.047** | 0.885 | 0.861 ± 0.112 | **0.908** |
| DenseNet121 | hed | 0.853 ± 0.077 | 0.920 ± 0.061 | 0.854 | 0.848 ± 0.119 | 0.887 |
| DenseNet121 | both | 0.851 ± 0.049 | 0.922 ± 0.051 | 0.856 | 0.845 ± 0.115 | 0.887 |
| EfficientNet-B0 | geom | 0.834 ± 0.065 | 0.895 ± 0.076 | 0.851 | 0.796 ± 0.133 | 0.874 |
| EfficientNet-B0 | jitter | 0.872 ± 0.067 | 0.920 ± 0.066 | 0.894 | 0.825 ± 0.151 | 0.906 |
| EfficientNet-B0 | hed | 0.838 ± 0.055 | 0.900 ± 0.065 | 0.854 | 0.799 ± 0.153 | 0.879 |
| EfficientNet-B0 | both | 0.824 ± 0.042 | 0.911 ± 0.068 | 0.807 | 0.854 ± 0.147 | 0.862 |
| Soft-vote ensemble | | 0.869 ± 0.049 | 0.934 ± 0.054 | 0.881 | 0.836 ± 0.158 | 0.902 |

The ensemble does **not** beat the best single model (DenseNet121 + `jitter`).

### 1. Rankings are unstable across patient folds

Mean Kendall τ between folds' rankings of the 8 configurations is **0.43** (range 0.14–0.64). Per-configuration fold AUCs span wide ranges (e.g. EfficientNet-B0 + `geom`: 0.78–0.98; DenseNet121 + `jitter`: 0.88–0.98), comparable to or larger than most gaps between configurations. A single split can easily crown the wrong model. See `figures/ranking_stability.png`.

### 2. Stain robustness is augmentation-family-specific

AUC drop (clean minus shifted; **lower is better**) at the reference severity, mean over folds:

| Model | Training aug | HED 0.3 | ColorJitter 0.3 | Reinhard 1.0 (independent) |
|-------|--------------|---------|-----------------|----------------------------|
| DenseNet121 | geom | 0.156 | 0.036 | 0.254 |
| DenseNet121 | jitter | 0.153 | **0.011** | 0.244 |
| DenseNet121 | hed | 0.080 | 0.048 | 0.161 |
| DenseNet121 | both | **0.074** | 0.015 | **0.121** |
| EfficientNet-B0 | geom | 0.153 | 0.050 | 0.273 |
| EfficientNet-B0 | jitter | 0.163 | **0.011** | 0.256 |
| EfficientNet-B0 | hed | 0.089 | 0.054 | 0.196 |
| EfficientNet-B0 | both | **0.084** | 0.012 | **0.148** |

Paired AUC differences, 95 % patient-clustered bootstrap CI (unadjusted for multiple comparisons; DenseNet121 / EfficientNet-B0):

| Scenario | Comparison | DenseNet121 | EfficientNet-B0 |
|----------|------------|-------------|-----------------|
| Clean | jitter − geom | +0.030 [0.005, 0.065] | +0.027 [0.001, 0.061] |
| ColorJitter 0.3 | jitter − geom | +0.056 [0.030, 0.091] | +0.069 [0.041, 0.101] |
| ColorJitter 0.3 | hed − geom | −0.004 [−0.024, 0.025] | −0.001 [−0.024, 0.027] |
| HED 0.3 | hed − geom | +0.077 [0.050, 0.108] | +0.070 [0.043, 0.099] |
| HED 0.3 | jitter − geom | +0.026 [−0.010, 0.062] | +0.015 [−0.010, 0.047] |
| Reinhard 1.0 | hed − geom | +0.095 [0.069, 0.124] | +0.085 [0.050, 0.119] |
| Reinhard 1.0 | jitter − geom | +0.026 [−0.012, 0.062] | +0.041 [0.011, 0.078] |
| Reinhard 1.0 | both − geom | +0.129 [0.082, 0.173] | +0.137 [0.091, 0.180] |

In short: ColorJitter training protects against ColorJitter shift but not stain (HED/Reinhard) shift; HED training roughly halves the HED and Reinhard drops but gives no ColorJitter benefit; `both` protects against all three families, at some cost in clean AUC relative to `jitter` for DenseNet121 (−0.027 [−0.052, −0.006]). Even `both` loses 0.12–0.15 AUC under full Reinhard transfer, so absolute robustness remains poor. See `figures/transfer_matrix.png` and `figures/shift_curves.png`.

### 3. Validation-tuned thresholds under-deliver; conformal control is conservative

Realised mean per-patient miss rate (FNR) on held-out malignant patients vs. the target α, range over the 8 model/augmentation configurations (58 malignant patients in the calibration pool):

| Target α | Validation-tuned | Naive pooled | Patient-level CRC | CRC specificity |
|----------|------------------|--------------|-------------------|-----------------|
| 0.02 | 0.026–0.048 | 0.026–0.030 | 0.003–0.005 | 0.09–0.21 |
| 0.05 | 0.059–0.080 | 0.058–0.065 | 0.035–0.039 | 0.46–0.61 |
| 0.10 | 0.095–0.125 | 0.107–0.115 | 0.087–0.089 | 0.67–0.80 |
| 0.20 | 0.169–0.221 | 0.202–0.211 | 0.188–0.189 | 0.82–0.91 |

- Validation-tuned thresholds overshoot the target at low α (≈1.2–2.4× the target at α = 0.02–0.05).
- Patient-level CRC meets every target but is **conservative** (realised FNR below α) and costs specificity; at α = 0.02 it is unusable (the guarantee requires ≥ 1/α malignant patients in the calibration pool).
- The **naive pooled** threshold is close to nominal with better specificity. Much of the validation-tuning failure therefore comes from the small validation set (~12 patients), not from the absence of a conformal procedure. CRC's added value is the guarantee and its measured cost.
- Temperature scaling did not help: it increased ECE in 72 % of the 40 runs (mean +0.010), most clearly for EfficientNet-B0. Raw ECE is already small (0.045–0.053), partly because of label smoothing.

See `figures/threshold_transfer.png`.

### 4. Stain-perturbation disagreement as an abstention signal

Area under the risk–coverage curve (AURC, **lower is better**), mean over all 40 single-model runs (4 augmentations × 2 models × 5 folds; the ensemble is excluded):

| Condition (base accuracy) | MSP | TTA-dihedral std | TTA-HED std |
|---------------------------|-----|------------------|-------------|
| Clean (0.850) | 0.077 | **0.068** | 0.084 |
| Reinhard 0.5 (0.785) | 0.128 | **0.113** | 0.124 |
| Reinhard 1.0 (0.656) | 0.304 | 0.275 | **0.250** |
| HED 0.3 (0.736) | 0.199 | 0.177 | **0.157** |

- HED-view disagreement is the best signal only under **strong** shift (HED 0.3, Reinhard 1.0), where it beats max-softmax in 97–100 % of runs.
- On clean data and at Reinhard 0.5, the plain dihedral-TTA control is as good or better, so much of the gain is "any test-time augmentation helps", not stain-specific.
- Practical effect is small: accuracy at 90 % coverage is nearly identical across scores (e.g. 0.671 for MSP vs 0.673 for TTA-HED std under Reinhard 1.0), and error-detection AUROC under shift is only 0.58–0.70.

See `figures/abstention_curves.png`.

### 5. Adversarial sanity check (exploratory)

Accuracy on 150 fold-0 test images (clean: 0.86–0.97):

| | FGSM 1/255 | FGSM 2/255 | PGD 1/255 | PGD 2/255 |
|---|---|---|---|---|
| Range over models/augs | 0.17–0.44 | 0.14–0.42 | 0.00–0.03 | 0.00 |

Undefended CNNs collapse under PGD, as already documented for computational pathology (Ghaffari Laleh et al., 2022). The `both` models had lower FGSM accuracy (0.14 at 2/255) than the others (0.29–0.42), but this is one fold with n = 150 and is not a finding.

## Limitations

- **One seed, two architectures.** Fold-to-fold sd is large (AUC sd 0.05–0.08). Rankings and effect sizes need more seeds (`SEEDS=[42,43,44]`) and more architectures before general claims.
- **No external site.** BreaKHis comes from one laboratory. All shifts are synthetic or within-dataset; no cross-site claim is supported. An external set (e.g. BACH, BreCaHAD) is the most valuable missing piece.
- **The Reinhard shift is confounded and extreme.** Matching global lαβ statistics to a random other image also changes tissue composition (e.g. amount of white background), not only stain. Level 1.0 is harsher than real stain variation; level 0.5 is more plausible.
- **Training HED strength is probably low.** Across-patient variation in mean stain concentrations is equivalent to uniform levels of about 0.6–1.8 (a proxy that also includes tissue content), while training used σ = 0.1. A sweep over σ may change the `hed` and `both` conclusions.
- **Multiple comparisons.** 48 paired CIs are reported without correction. Treat a few pre-specified comparisons as primary and the rest as exploratory.
- **CRC approximations.** Calibration uses out-of-fold predictions from different fold models (a cross-conformal approximation to exchangeability), the guarantee is marginal over patients, and only one seed's pooled predictions are used.
- **Abstention gains are modest** at practical coverage levels (see above) and were not tested with a paired bootstrap.
- **Small test folds.** ~16 test patients per fold; one hard fold (fold 1: e.g. DenseNet121 + `geom` AUC 0.87, specificity 0.61) strongly affects means.
- **Pooled magnifications, 224×224 squashing, binary task only.** Magnification-wise and 8-class results are not produced by this code.
- This is research code, not a clinical tool.

## References

- F. A. Spanhol, L. S. Oliveira, C. Petitjean, L. Heutte, "A dataset for breast cancer histopathological image classification," IEEE Trans. Biomed. Eng., 2016.
- D. Tellez et al., "Quantifying the effects of data augmentation and stain color normalization in convolutional neural networks for computational pathology," Med. Image Anal., 2019.
- N. Ghaffari Laleh et al., "Adversarial attacks and adversarial robustness in computational pathology," Nat. Commun., 2022.
- A. N. Angelopoulos, S. Bates, A. Fisch, L. Lei, T. Schuster, "Conformal risk control," 2022.
- E. Reinhard, M. Adhikhmin, B. Gooch, P. Shirley, "Color transfer between images," IEEE Comput. Graph. Appl., 2001.
- C. Guo et al., "On calibration of modern neural networks," ICML, 2017.
- A. Madry et al., "Towards deep learning models resistant to adversarial attacks," ICLR, 2018.

## Authors and acknowledgements

Anika Jalan, Aindrila Kundu, Department of Computer Science, Netaji Subhas University of Technology, Delhi. Thanks to the creators of BreaKHis (P&D Laboratory, Paraná, Brazil).

