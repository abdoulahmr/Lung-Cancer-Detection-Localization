# Lung Cancer Detection & Localization

A two-stage deep learning pipeline for detecting, localizing, and classifying
lung tumors (nodules) in medical images — built to solve a specific
false-positive problem that a detector-only approach ran into.

## Overview

Object detectors are good at finding *candidate* regions but tend to flag a
lot of false positives — boxes around tissue that only looks tumor-like.
This project fixes that with a **two-step pipeline**:

1. **Detector** — Faster R-CNN with a ResNet-152 + FPN backbone proposes
   bounding boxes for suspicious regions.
2. **Classifier** — a ResNet-50 binary classifier re-inspects every box the
   detector proposes and decides whether it's a real finding (and whether
   it's benign or malignant), rejecting the ones that aren't.

At inference time, both models use an **adaptive confidence threshold**:
detection/classification thresholds start high (0.9) and relax in steps of
0.05 down to a floor of 0.05 until at least one confident detection
survives. This keeps precision high on easy cases while still recovering
faint findings on harder ones.

## Results

Evaluated on a held-out validation set (IoU ≥ 0.5 counts as a match):

| Metric | Value |
|---|---|
| Precision | **0.8734** |
| Recall | **0.9979** |
| F1-Score | **0.9315** |
| Mean IoU | **0.8104** |

Recall near 1.0 confirms the detector almost never misses a true nodule. The
classifier stage is what lifts precision from a false-positive-heavy
detector-only baseline up to ~0.87.

## Pipeline

```
Image ──► Faster R-CNN (ResNet-152 + FPN) ──► candidate boxes
                                                     │
                                                     ▼
                                        crop each box, resize to 224×224
                                                     │
                                                     ▼
                                     ResNet-50 classifier ──► keep / reject
                                                     │
                                                     ▼
                                   final boxes + benign/malignant labels
```

## Repository structure

```
.
├── lung_cancer_detection_pipeline.ipynb   # full pipeline: data, models, training, inference, evaluation
├── detector.ckpt                           # trained Faster R-CNN weights (not included — see below)
└── classifier.ckpt                         # trained ResNet-50 weights (not included — see below)
```

> Model checkpoints are not tracked in this repo due to size. Train your own
> with the notebook, or point the config paths at your own `.ckpt` files.

## Requirements

```
torch
torchvision
pytorch-lightning
xmltodict
Pillow
matplotlib
tqdm
```

Install with:

```bash
pip install torch torchvision pytorch-lightning xmltodict pillow matplotlib tqdm
```

## Data format

Expects PASCAL VOC–style annotations:

```
dataset/
├── JPEGImages/
│   ├── 0001.png
│   └── ...
└── Annotations/
    ├── 0001.xml
    └── ...
```

Each `<object>` in an annotation is labeled `cancer` (mapped to detector
class `1`) or anything else (mapped to class `2`).

## Usage

Open `lung_cancer_detection_pipeline.ipynb` and run top to bottom. It covers:

1. Config — set your data/checkpoint paths
2. Dataset — `VOCDataset` for loading images + annotations
3. Model definitions — `Detector` and `TumorClassifier`
4. Training — Lightning `Trainer` setup for both stages
5. Two-stage inference — `detect_and_classify()` + `draw_detections()`
6. Full validation loop — computes Precision / Recall / F1 / Mean IoU

### Quick inference example

```python
detector = Detector.load_from_checkpoint("detector.ckpt").to(DEVICE).eval()
classifier = TumorClassifier.load_from_checkpoint("classifier.ckpt").to(DEVICE).eval()

image = Image.open("sample.png").convert("RGB")
boxes, labels, probs = detect_and_classify(image, detector, classifier)

for box, label, p in zip(boxes, labels, probs):
    print(f"{label}: {p:.2f} at {box}")
```
**Results:**
Case courtesy of Ashesh Ishwarlal Ranchod, <a href="https://radiopaedia.org/?lang=us">Radiopaedia.org</a>. From the case <a href="https://radiopaedia.org/cases/222172?lang=us">rID: 222172</a>
![input](samples/tumor.jpeg)
![output](samples/output.png)
## Notes

- The detector's class labels (`cancer` vs. other) and the classifier's
  labels (`benign` vs. `malignant`) are two separate label spaces — the
  detector decides *where* something is, the classifier decides *what* it
  is (and whether it's real).
- Thresholds (`INITIAL_DET_THRESH`, `INITIAL_CLS_THRESH`, `STEP`,
  `MIN_THRESH`, `NMS_IOU_THRESHOLD`, `IOU_THRESHOLD`) are all defined in one
  place in the notebook's config cell for easy tuning.

## License

Add a license of your choice (e.g. MIT) before publishing.
