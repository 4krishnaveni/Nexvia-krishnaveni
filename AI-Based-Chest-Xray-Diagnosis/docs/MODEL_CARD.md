# Model card: CXR opacity classifier and segmenter

> **FOR EDUCATION ONLY. NOT FOR CLINICAL USE.** These models were built for a training project, have not been
> clinically validated, and must never inform care of a real patient.

## Intended use
Teaching demo of an AI-assisted radiology workflow (DICOM in, AI suggestion + explanation, human approve/reject, audit trail).

## Models
* **Classifier**: DenseNet121 (ImageNet init), 1 logit "pneumonia-like lung opacity", 256x256, temperature-scaled.
* **Segmenter**: MONAI 2-D U-Net trained on *box-derived* masks (weak supervision). The brief names a 3-D U-Net; 2-D chest X-rays have no depth axis.
* **Explanation**: Grad-CAM on the last DenseNet feature map.

## Data
RSNA Pneumonia Detection Challenge (Kaggle), about 26.7k frontal CXRs, patient-level 70/15/15 split (seed 42). Labels derive from radiologist annotation of opacities.

## Results (copy from `reports/metrics.json` after `python -m src.eval.evaluate`)
| Metric | Value |
|---|---|
| AUROC / AUPRC | TODO |
| Sensitivity / specificity at val-chosen threshold | TODO |
| ECE (calibration) | TODO |
| Dice on positives (box masks) | TODO |
| Grad-CAM pointing-game accuracy | TODO |
| AUROC by sex, by age band | TODO |

## Limitations and bias (be specific in the final write-up)
1. Single source (RSNA, drawn from NIH ChestX-ray14, US hospitals): no external validation, so performance on other scanners, populations, or countries is unknown.
2. "Positive" means a radiologist-marked lung opacity, which is not specific to pneumonia (edema, atelectasis, cancer and others look similar).
3. Segmentation masks are rectangles, so Dice is capped and the U-Net outlines boxes, not true lesion borders.
4. Mixed AP/PA views and portable-machine artifacts (tubes, text markers) can drive predictions; check Grad-CAM failure cases in `reports/figures`.
5. Subgroup performance (sex, age) may differ; see table above. Small subgroups give wide uncertainty.
6. Confidence is calibrated on one validation set only; it can be overconfident on out-of-distribution images.
7. Not evaluated for adversarial robustness, image compression, or dose differences.
