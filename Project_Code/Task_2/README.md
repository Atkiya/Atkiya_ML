# TinyMyo — EMG/IMU Gesture Recognition Benchmark

> A comprehensive supervised-learning benchmark for 17-class hand-gesture classification using surface EMG and accelerometer signals. Four models are evaluated across **9 train/test splits** (10 % to 90 % training data) to characterise both peak accuracy and data-efficiency.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Models](#models)
4. [Results — All 9 Splits per Model](#results--all-9-splits-per-model)
   - [TinyMyo](#1-tinymyo)
   - [PaperDNN (Feed-Forward DNN)](#2-paperdnn-feed-forward-dnn)
   - [DeepCNN](#3-deepcnn)
   - [MKA 1D-CNN (Multi-Kernel Attention)](#4-mka-1d-cnn-multi-kernel-attention)
5. [Cross-Model Comparison](#cross-model-comparison)
6. [Failure Mode Analysis](#failure-mode-analysis)
7. [Outputs & Artifacts](#outputs--artifacts)

---

## Project Overview

TinyMyo benchmarks four deep-learning architectures on a windowed EMG+accelerometer dataset covering 17 distinct hand/wrist gestures. Each model is trained and evaluated at nine different data-split ratios — from 90 % training data down to 10 % — enabling a rigorous study of generalisation, data efficiency, and overfitting behaviour under low-resource conditions.

All experiments were executed on Kaggle (GPU environment). Metrics reported are computed on held-out test windows that are never seen during training or validation.

---

## Dataset

| Property | Value |
|---|---|
| Total windows | 87,614 |
| Input channels | 48 (EMG = 12, ACC = 36) |
| Window size | 400 samples (200 ms) |
| Number of classes | 17 |
| Class labels | G13 – G29 |

### Per-Class Window Counts

| Gesture | Windows | Share |
|---|---|---|
| G13 | 4,701 | 5.4 % |
| G14 | 4,298 | 4.9 % |
| G15 | 4,150 | 4.7 % |
| G16 | 5,088 | 5.8 % |
| G17 | 4,374 | 5.0 % |
| G18 | 4,898 | 5.6 % |
| G19 | 5,070 | 5.8 % |
| G20 | 4,873 | 5.6 % |
| G21 | 5,090 | 5.8 % |
| G22 | 5,348 | 6.1 % |
| G23 | 5,248 | 6.0 % |
| G24 | 6,457 | 7.4 % |
| G25 | 5,463 | 6.2 % |
| G26 | 6,465 | 7.4 % |
| G27 | 5,225 | 6.0 % |
| G28 | 5,399 | 6.2 % |
| G29 | 5,467 | 6.2 % |

The dataset is approximately balanced across gestures, with the smallest class (G15, 4,150 windows) being only 1.6× smaller than the largest (G26, 6,465 windows).

---

## Models

### TinyMyo
A lightweight, compact architecture designed for efficient on-device inference. Only evaluated at the 90:10 split as a reference point.

### PaperDNN (Feed-Forward DNN)
A paper-exact feed-forward deep neural network that operates on hand-crafted feature vectors extracted from the raw signal windows.

| Property | Value |
|---|---|
| Architecture | Multi-layer feed-forward DNN |
| Parameters | 376,081 |
| Feature vector size | 336 |
| Early stopping | Patience 20 (at epoch 100 across most splits) |

### DeepCNN
A large convolutional neural network that processes raw windowed signals directly without manual feature extraction. Trades training time for high raw accuracy.

| Property | Value |
|---|---|
| Architecture | Deep 1D CNN |
| Parameters | 5,941,649 |
| Input format | 48 × 400 (channels × samples) |
| Early stopping | Patience 20, max 150 epochs |

### MKA 1D-CNN (Multi-Kernel Attention 1D CNN)

A 1D convolutional network with multi-kernel branches and an attention mechanism, offering a strong balance between parameter efficiency and classification accuracy.

| Property | Value |
|---|---|
| Architecture | Multi-Kernel Attention 1D CNN |
| Parameters | 515,329 |
| Input format | 48 × 400 (channels × samples) |
| Early stopping | Patience 20, max 150 epochs |

---

## Results — All 9 Splits per Model

All metrics are computed on the held-out test set. **Accuracy**, **Precision (W)**, **Recall (W)**, and **F1 (W)** are weighted averages across the 17 classes. **ROC-AUC (W)** is the weighted one-vs-rest area under the ROC curve. Train/Test wall times are in seconds.

---

### 1. TinyMyo

> Only evaluated at the 90:10 split.

| Split | Train Size | Test Size | Accuracy | Precision (W) | Recall (W) | F1 (W) | ROC-AUC (W) | Epochs |
|:---:|---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 90:10 | 70,966 | 8,762 | **0.9781** | **0.9783** | **0.9781** | **0.9781** | **0.9997** | 173 |
| 80:20 | — | — | — | — | — | — | — | — |
| 70:30 | — | — | — | — | — | — | — | — |
| 60:40 | — | — | — | — | — | — | — | — |
| 50:50 | — | — | — | — | — | — | — | — |
| 40:60 | — | — | — | — | — | — | — | — |
| 30:70 | — | — | — | — | — | — | — | — |
| 20:80 | — | — | — | — | — | — | — | — |
| 10:90 | — | — | — | — | — | — | — | — |

---

### 2. PaperDNN (Feed-Forward DNN)

**Best split: 90:10** — Weighted F1 = 0.9283 | Balanced Accuracy = 0.9284

| Split | Train Size | Test Size | Accuracy | Balanced Acc | Precision (W) | Recall (W) | F1 (W) | ROC-AUC (W) | Train (s) | Test (s) |
|:---:|---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|---:|---:|
| **90:10** | 70,966 | 8,762 | **0.9283** | **0.9284** | **0.9287** | **0.9283** | **0.9283** | **0.9982** | 185.9 | 0.37 |
| 80:20 | 63,081 | 17,523 | 0.9230 | 0.9232 | 0.9233 | 0.9230 | 0.9230 | 0.9982 | 157.2 | 0.40 |
| 70:30 | 55,196 | 26,285 | 0.9113 | 0.9117 | 0.9117 | 0.9113 | 0.9113 | 0.9975 | 152.8 | 0.50 |
| 60:40 | 47,311 | 35,046 | 0.8963 | 0.8961 | 0.8976 | 0.8963 | 0.8964 | 0.9968 | 131.5 | 0.58 |
| 50:50 | 39,426 | 43,807 | 0.8848 | 0.8851 | 0.8859 | 0.8848 | 0.8849 | 0.9960 | 120.9 | 0.68 |
| 40:60 | 31,540 | 52,569 | 0.8604 | 0.8611 | 0.8612 | 0.8604 | 0.8605 | 0.9943 | 96.0 | 0.83 |
| 30:70 | 23,655 | 61,330 | 0.8288 | 0.8295 | 0.8298 | 0.8288 | 0.8289 | 0.9913 | 80.1 | 0.84 |
| 20:80 | 15,769 | 70,092 | 0.7667 | 0.7686 | 0.7676 | 0.7667 | 0.7668 | 0.9837 | 64.5 | 0.93 |
| 10:90 | 7,884 | 78,853 | 0.6844 | 0.6867 | 0.6877 | 0.6844 | 0.6841 | 0.9672 | 49.3 | 1.05 |

**Key observations:**
- Performance degrades monotonically as training data decreases.
- Even with only 10 % training data (7,884 samples), the model achieves 68.4 % accuracy.
- The gap between 90:10 and 80:20 is small (~0.5 pp), while the drop from 30:70 to 10:90 is dramatic (~14.5 pp).
- Train time scales almost linearly with dataset size, staying under 3 minutes even at 90:10.

---

### 3. DeepCNN

**Best split: 90:10** — Weighted F1 = 0.9803 | ROC-AUC = 0.9998

| Split | Train Size | Test Size | Accuracy | Precision (W) | Recall (W) | F1 (W) | ROC-AUC (W) | Train (s) | Test (s) |
|:---:|---:|---:|:---:|:---:|:---:|:---:|:---:|---:|---:|
| **90:10** | 70,966 | 8,762 | **0.9803** | **0.9804** | **0.9803** | **0.9803** | **0.9998** | 6,657.3 | 1.86 |
| 80:20 | 63,081 | 17,523 | 0.9756 | 0.9757 | 0.9756 | 0.9756 | 0.9998 | 5,889.7 | 3.03 |
| 70:30 | 55,196 | 26,285 | 0.9738 | 0.9740 | 0.9738 | 0.9738 | 0.9997 | 4,866.2 | 4.28 |
| 60:40 | 47,311 | 35,046 | 0.9659 | 0.9661 | 0.9659 | 0.9659 | 0.9995 | 4,162.4 | 5.46 |
| 50:50 | 39,426 | 43,807 | 0.9407 | 0.9419 | 0.9407 | 0.9407 | 0.9989 | 3,528.3 | 6.85 |
| 40:60 | 31,540 | 52,569 | 0.9536 | 0.9540 | 0.9536 | 0.9536 | 0.9991 | 2,904.3 | 8.41 |
| 30:70 | 23,655 | 61,330 | 0.9331 | 0.9337 | 0.9331 | 0.9331 | 0.9983 | 1,758.0 | 9.65 |
| 20:80 | 15,769 | 70,092 | 0.9061 | 0.9073 | 0.9061 | 0.9061 | 0.9968 | 1,134.7 | 11.34 |
| 10:90 | 7,884 | 78,853 | 0.8515 | 0.8537 | 0.8515 | 0.8513 | 0.9926 | 595.6 | 12.84 |

**Key observations:**
- DeepCNN achieves 98 %+ accuracy at 90:10, the highest among all models at that split.
- Even at 10 % training data it reaches 85 % accuracy — significantly better than PaperDNN at the same split.
- The 40:60 split slightly outperforms 50:50, likely due to training instability at 50:50 (noisy early-stopping with erratic val_loss).
- Training is very expensive: the 90:10 split requires ~1.85 hours on GPU. Not recommended for rapid prototyping.

---

### 4. MKA 1D-CNN (Multi-Kernel Attention)

**Best split: 80:20** — Weighted F1 = 0.9876 | ROC-AUC = 0.9999

| Split | Train Size | Test Size | Accuracy | Precision (W) | Recall (W) | F1 (W) | ROC-AUC (W) | Train (s) | Test (s) |
|:---:|---:|---:|:---:|:---:|:---:|:---:|:---:|---:|---:|
| 90:10 | 70,966 | 8,762 | 0.9872 | 0.9873 | 0.9872 | 0.9872 | 0.9999 | 2,366.6 | 1.10 |
| **80:20** | 63,081 | 17,523 | **0.9876** | **0.9876** | **0.9876** | **0.9876** | **0.9999** | 2,068.9 | 1.65 |
| 70:30 | 55,196 | 26,285 | 0.9793 | 0.9794 | 0.9793 | 0.9793 | 0.9998 | 798.7 | 2.11 |
| 60:40 | 47,311 | 35,046 | 0.9767 | 0.9768 | 0.9767 | 0.9767 | 0.9997 | 833.6 | 2.65 |
| 50:50 | 39,426 | 43,807 | 0.9550 | 0.9554 | 0.9550 | 0.9550 | 0.9992 | 584.6 | 3.28 |
| 40:60 | 31,540 | 52,569 | 0.9615 | 0.9616 | 0.9615 | 0.9615 | 0.9993 | 499.5 | 4.07 |
| 30:70 | 23,655 | 61,330 | 0.9390 | 0.9393 | 0.9390 | 0.9390 | 0.9985 | 615.8 | 4.87 |
| 20:80 | 15,769 | 70,092 | 0.9308 | 0.9308 | 0.9308 | 0.9307 | 0.9981 | 305.2 | 6.11 |
| 10:90 | 7,884 | 78,853 | 0.8875 | 0.8882 | 0.8875 | 0.8876 | 0.9951 | 187.0 | 5.81 |

**Key observations:**
- MKA 1D-CNN is the **overall best model**, reaching 98.76 % accuracy at 80:20 — beating even DeepCNN's 90:10 result.
- Exceptionally data-efficient: at 10:90 it achieves 88.75 %, compared to 85.15 % for the much larger DeepCNN.
- Training time is reasonable: 80:20 takes ~34 minutes vs ~1.85 hours for DeepCNN at 90:10.
- The 80:20 split marginally outperforms 90:10 (98.76 % vs 98.72 %), suggesting 80:20 is the sweet spot.
- **Recommended as the primary supervised baseline for downstream SSL experiments.**

---

## Cross-Model Comparison

Summary of the **best split** results for each model:

| Model | Best Split | Accuracy | F1 (W) | ROC-AUC (W) | Parameters | Train (s) |
|---|:---:|:---:|:---:|:---:|---:|---:|
| TinyMyo | 90:10 | 0.9781 | 0.9781 | 0.9997 | — | — |
| PaperDNN | 90:10 | 0.9283 | 0.9283 | 0.9982 | 376,081 | 185.9 |
| DeepCNN | 90:10 | 0.9803 | 0.9803 | 0.9998 | 5,941,649 | 6,657.3 |
| **MKA 1D-CNN** | **80:20** | **0.9876** | **0.9876** | **0.9999** | **515,329** | **2,068.9** |

### F1 (Weighted) Across All 9 Splits

| Split | PaperDNN | DeepCNN | MKA 1D-CNN |
|:---:|:---:|:---:|:---:|
| 90:10 | 0.9283 | 0.9803 | 0.9872 |
| 80:20 | 0.9230 | 0.9756 | **0.9876** |
| 70:30 | 0.9113 | 0.9738 | 0.9793 |
| 60:40 | 0.8964 | 0.9659 | 0.9767 |
| 50:50 | 0.8849 | 0.9407 | 0.9550 |
| 40:60 | 0.8605 | 0.9536 | 0.9615 |
| 30:70 | 0.8289 | 0.9331 | 0.9390 |
| 20:80 | 0.7668 | 0.9061 | 0.9307 |
| 10:90 | 0.6841 | 0.8513 | 0.8876 |

MKA 1D-CNN consistently outperforms both PaperDNN and DeepCNN at every split. DeepCNN is superior to PaperDNN across all conditions, while PaperDNN has an advantage in extremely fast training times.

---

## Failure Mode Analysis

Across all models and splits, the gesture classes that are most consistently misclassified are:

| Difficulty | Classes |
|---|---|
| Hardest | G19, G17, G16, G20, G25, G28 |
| Medium | G26, G29, G22, G23, G27 |
| Easiest | G15, G13, G24, G18 |

**G15** is the easiest class to classify across all models and splits (typically >95 % per-class accuracy). **G19** is the most frequently worst-performing class, especially at low-data splits where it can fall below 70 % with PaperDNN (0.705 at 10:90).

> These findings suggest that certain gestures (G16, G17, G19, G20) produce overlapping feature representations and are natural targets for targeted data augmentation or hard-negative mining in future work.

---

## Outputs & Artifacts

Each training run saves the following files to the Kaggle working directory:

| Artifact | Description |
|---|---|
| `curves_<split>.png` | Training/validation loss and accuracy curves |
| `cm_<split>.png` | Confusion matrix on the test set |
| `roc_<split>.png` | Per-class ROC curves |
| `per_class_acc_<split>.png` | Per-class accuracy bar chart |
| `summary_all_splits.png` | Cross-split summary chart |
| `summary_all_splits.csv` | Machine-readable results table |
| `<model>_<split>.pt` | Saved model checkpoint (PyTorch) |

### Directory Layout

```
/kaggle/working/
├── results_dnn_paper/          # PaperDNN — splits 90:10, 80:20
│   ├── summary_all_splits.csv
│   └── ...
├── results_dnn_paper_B/        # PaperDNN — splits 70:30 to 50:50
├── results_dnn_paper_C/        # PaperDNN — splits 40:60 to 10:90
├── plots_dnn_paper/
├── ckpts_dnn_paper/
├── results_deepcnn_AB/         # DeepCNN — all splits
├── plots_deepcnn_AB/
├── ckpts_deepcnn_AB/
├── results_ALR/                # MKA 1D-CNN — all splits
├── plots_ALR/
└── ckpts_ALR/
```

---

## Recommended SSL Baselines

Based on the best-split analysis, the following supervised checkpoints are recommended as reference baselines for downstream Self-Supervised Learning (SSL) experiments:

| Model | Recommended Split | F1 (W) |
|---|:---:|:---:|
| PaperDNN | 90:10 | 0.9283 |
| DeepCNN | 90:10 | 0.9803 |
| MKA 1D-CNN | 80:20 | 0.9876 |
