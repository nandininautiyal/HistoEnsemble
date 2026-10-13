# Breast Cancer Histopathology Classification on BreaKHis

Patient-wise benchmark of ImageNet-pretrained CNNs for benign vs. malignant classification on the **BreaKHis** dataset, extended with stain-shift robustness, adversarial robustness, and calibrated selective prediction.

> **Status:** work in progress. Results are not final; the tables below are filled in after the full Kaggle run.

---

## Why this project

Many published BreaKHis results split the data at the *image* level, so images from the same patient end up in both train and test. This inflates accuracy. Here every patient appears in exactly one split, and we ask:

1. **Does colour augmentation help or hurt?** Compare resize-only, geometry-only (control), ColorJitter, and stain-aware HED jitter.
2. **How robust are the models to stain shift?** Evaluate each trained model on the same test images under controlled synthetic stain/colour shifts.
3. **How robust are they to adversarial perturbations?** White-box FGSM and PGD.
4. **Can calibration fix over-calling malignant?** Temperature scaling, validation-chosen thresholds, and risk-coverage (abstention) curves.

HED jitter, FGSM/PGD and temperature scaling are existing techniques. The contribution is the controlled, patient-wise comparison on BreaKHis, not a new method.

## Dataset

[BreaKHis](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/) (Spanhol et al., 2016): 7,909 H&E microscopy images from 82 patients (2,480 benign, 5,429 malignant) at 40x, 100x, 200x and 400x. On Kaggle, attach the `ambarish/breakhis` dataset to the notebook.

Patient ids are parsed from the file names; the notebook prints the number of patients it finds.

## Repository contents

| File | Description |
|------|-------------|
| `Breast_Cancer_Detection_Kaggle_v2.ipynb` | Main notebook (training, evaluation, all plots) |
| `Breast_Cancer_Detection_Kaggle.ipynb` | Earlier simpler version (kept for reference) |

## Method overview

- **Split:** patient-grouped, class-stratified 5-fold (`StratifiedGroupKFold`). Inside each training portion, about 20% of patients form a validation set used only for early stopping, temperature scaling and threshold selection.
- **Models:** ResNet50, DenseNet121, EfficientNet-B0 by default (also available: ResNeXt50, MobileNetV2, VGG16, AlexNet, ConvNeXt-Tiny, ShuffleNetV2). ImageNet weights, new 2-class head, full fine-tuning.
- **Training:** AdamW (backbone 5e-5, head 5e-4), label smoothing, class-weighted loss, mixed precision, up to 15 epochs with early stopping on validation AUC.
- **Augmentation tiers:**
  - `basic`: resize only
  - `geom`: flips + rotation (control)
  - `jitter`: `geom` + ColorJitter(0.3, 0.3, 0.3, 0.1)
  - `hed`: `geom` + HED stain jitter
- **Stain-shift benchmark:** random per-image HED-concentration scaling/offset, and brightness/contrast/saturation shifts, at five severities (0 to 0.4).
- **Adversarial evaluation:** L-infinity FGSM and PGD with a margin loss in fp32, on a fixed random subset of test images.
- **Calibration:** temperature scaling fitted on validation patients; operating points (Youden, 99% sensitivity) chosen on validation and applied to test; risk-coverage curves.
- **Ensemble:** soft vote over the best augmentation per architecture, chosen by validation AUC (never test).

## Running on Kaggle

1. Create a new notebook and upload `Breast_Cancer_Detection_Kaggle_v2.ipynb`.
2. **Add Input** and attach the BreaKHis dataset.
3. Session options: GPU (T4 or P100) and **Internet On** (needed once to download pretrained weights).
4. Set `CFG["QUICK_TEST"] = True` for a roughly 2-minute smoke test, then set it back to `False`.
5. Use **Save Version, Save & Run All (Commit)** for the full run.

Main settings are in the config cell (`CFG`): `MODELS`, `AUGS`, `EPOCHS`, `PATIENCE`, `FOLDS_TO_RUN` (use `[0,1,2,3,4]` for full cross-validation), `RUN_ROBUSTNESS`.

Estimated runtime (not yet measured on Kaggle): about 2 hours on a T4 for one fold with the default 3 models x 4 augmentations. Full 5-fold CV is about 5 times that.

## Outputs

Written to `/kaggle/working`:

- `results_per_fold.csv`, `results_summary.csv`
- `stain_shift_results.csv`, `adversarial_results.csv`
- Figures: `eda_distribution.png`, `eda_samples.png`, `training_curves.png`, `metrics_comparison.png`, `heatmap_model_aug.png`, `confusion_matrices.png`, `roc_pr_curves.png`, `stain_shift_robustness.png`, `adversarial_robustness.png`, `calibration_selective.png`, `misclassified.png`

## Results

_To be filled in after the full run._

| Model | Augmentation | Accuracy | F1 | AUC | Specificity | Spec @ 99% sens | ECE (cal.) |
|-------|--------------|----------|----|-----|-------------|-----------------|------------|
| | | | | | | | |

## Limitations

- BreaKHis has only about 82 patients, so a single split has roughly 16 test patients and noisy estimates. Report mean and standard deviation over folds.
- Stain shifts are synthetic, not real scanner or site shift. An external dataset (for example BACH) would be needed for a stronger generalisation claim.
- Adversarial results are white-box, on a subset of test images, with a limited set of budgets.
- This is research code, not a clinical tool.

## References

- F. A. Spanhol, L. S. Oliveira, C. Petitjean, L. Heutte, "A dataset for breast cancer histopathological image classification," IEEE Trans. Biomed. Eng., 2016.
- D. Tellez et al., "Quantifying the effects of data augmentation and stain color normalization in convolutional neural networks for computational pathology," Med. Image Anal., 2019.
- C. Guo et al., "On calibration of modern neural networks," ICML, 2017.
- A. Madry et al., "Towards deep learning models resistant to adversarial attacks," ICLR, 2018.

## Acknowledgements

Netaji Subhas University of Technology, Delhi, and the creators of BreaKHis (P&D Laboratory, Parana, Brazil).