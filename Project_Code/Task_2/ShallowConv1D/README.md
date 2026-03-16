# ShallowConv1D — Supervised EMG Gesture Recognition Baseline

A compact, fully-supervised 1D convolutional neural network for hand gesture classification from surface EMG and accelerometer signals. Trained and evaluated on **NinaPro DB7 Exercise B** (17-class, 22 subjects) across nine train/test split ratios. Designed as a reproducible **supervised baseline** for future self-supervised learning (SSL) experiments.

---

## What this is

The notebook implements a complete pipeline:

1. **Data loading** — reads per-subject `.mat` files from NinaPro DB7, extracts EMG (12 ch, 2000 Hz) and ACC signals, resamples ACC to 2000 Hz, and filters to Exercise B gesture labels (G13–G29, 17 classes).
2. **Windowing** — sliding windows of 200 ms with 50% overlap (100 ms step), producing fixed-size tensors of shape `(channels × 400)`.
3. **Normalisation** — per-channel z-score, fit on train only and applied to val/test.
4. **Model** — `ShallowConv1D`, a lightweight 1D VGG-style CNN (described below).
5. **Training** — Adam + Cosine Annealing LR, early stopping (patience 20) after a minimum of 100 epochs, max 200.
6. **Evaluation** — across nine split ratios from 90:10 to 10:90 (train:test), reporting Accuracy, Weighted Precision/Recall/F1, ROC-AUC, wall-clock time, and GFLOPs per window.
7. **Failure mode analysis** — per-class mean accuracy across splits, identifies the best split by weighted F1 as the reference for SSL comparisons.

---

## Model Architecture — `ShallowConv1D`

```
Input  (B, C, 400)          C = 12 EMG + n_acc channels

Block 1   Conv1d(C,  32, k=3) → BN → ReLU
          Conv1d(32, 32, k=3) → BN → ReLU → MaxPool(2)     → (B, 32, 200)

Block 2   Conv1d(32, 64, k=3) → BN → ReLU
          Conv1d(64, 64, k=3) → BN → ReLU → MaxPool(2)     → (B, 64, 100)

Block 3   Conv1d(64,  128, k=3) → BN → ReLU
          Conv1d(128, 128, k=3) → BN → ReLU                → (B, 128, 100)

Global Average Pooling                                      → (B, 128)
Dropout(0.3)
Linear(128, n_classes)                                      → (B, 17)
```

The design follows the VGGNet principle (Simonyan & Zisserman, 2015) adapted to 1D time-series: small kernels stacked in pairs, batch normalisation after every convolution, and Global Average Pooling in place of large fully-connected layers to reduce parameter count and overfitting.

---

## Dataset

**NinaPro DB7**
- 22 able-bodied subjects
- 12-channel sEMG + ACC, sampled at 2000 Hz / 148 Hz
- Exercise B: 17 hand and wrist gestures (labels 13–29)
- Labels taken from `restimulus` (movement-onset corrected)

> Kanzler, C.M., Muheim, J., Rinderknecht, M.D., Kulic, D., Lambercy, O., Gassert, R. (2017). **NinaPro DB7: A dataset for sEMG-based hand gesture recognition on upper limb amputees and intact subjects.** *Proceedings of the IEEE EMBC.*

Dataset available at: https://ninapro.hevs.ch

---

## Configuration

| Parameter | Value |
|---|---|
| Sampling rate | 2000 Hz |
| Window length | 200 ms (400 samples) |
| Window step | 100 ms (50% overlap) |
| Batch size | 256 |
| Optimizer | Adam (lr=1e-3, wd=1e-4) |
| Scheduler | CosineAnnealingLR |
| Min / Max epochs | 100 / 200 |
| Early stop patience | 20 epochs |
| Val fraction | 10% of train |
| Seed | 42 |

---

## Split Ratios Evaluated

| Train | Test |
|---|---|
| 90% | 10% |
| 80% | 20% |
| 70% | 30% |
| 60% | 40% |
| 50% | 50% |
| 40% | 60% |
| 30% | 70% |
| 20% | 80% |
| 10% | 90% |

The best split by weighted F1 is flagged as the reference supervised baseline for SSL experiments.

---

## Outputs

| Path | Contents |
|---|---|
| `ckpts_exB/` | Best model checkpoint per split (`.pt`) |
| `results_exB/` | Per-split metrics JSON + summary CSV |
| `plots_exB/` | Accuracy & F1 bar chart across splits |

---

## Requirements

```
torch >= 2.0
numpy
pandas
scipy
scikit-learn
matplotlib
seaborn
tqdm
thop        # optional — for GFLOPs measurement
```

---

## How to Run

Upload to Kaggle with NinaPro DB7 as the input dataset, then run all cells. The notebook auto-discovers the subject folders under `/kaggle/input`.

```python
if __name__ == '__main__':
    main()
```

---

## References

**Architecture**

1. Simonyan, K., & Zisserman, A. (2015). **Very Deep Convolutional Networks for Large-Scale Image Recognition.** *ICLR 2015.* https://arxiv.org/abs/1409.1556
   *(VGGNet — the double-conv-BN-ReLU-pool design pattern this model adapts to 1D)*

2. Lin, M., Chen, Q., & Yan, S. (2014). **Network In Network.** *ICLR 2014.* https://arxiv.org/abs/1312.4400
   *(Introduced Global Average Pooling as a replacement for large FC layers)*

3. Ioffe, S., & Szegedy, C. (2015). **Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.** *ICML 2015.* https://arxiv.org/abs/1502.03167

**EMG / sEMG deep learning — closest related work**

4. Côté-Allard, U., Fall, C.L., Drouin, A., Campeau-Lecours, A., Gosselin, C., Glette, K., Laviolette, F., & Gosselin, B. (2019). **Deep learning for electromyographic hand gesture signal classification using transfer learning.** *IEEE Transactions on Neural Systems and Rehabilitation Engineering, 27*(4), 760–771. https://doi.org/10.1109/TNSRE.2019.2896269
   *(Closest match — stacked conv-BN-ReLU-pool blocks applied directly to raw EMG time-series windows on NinaPro, same input modality and structural pattern as ShallowConv1D)*

5. Sun, Y., et al. (2020). **Hand Gesture Recognition Using Compact CNN via Surface Electromyography Signals.** *Sensors, 20*(3), 672. https://doi.org/10.3390/s20030672 (PMC7039218)
   *(Related design philosophy — kernel-3 throughout, GAP in place of FC layers, compact parameter count, NinaPro evaluation — but operates on 2D wavelet/spectrogram images rather than raw 1D signals)*

6. Zhai, X., Jelfs, B., Chan, R.H.M., & Tin, C. (2017). **Self-recalibrating surface EMG pattern recognition for neuroprosthesis control based on convolutional neural network.** *Frontiers in Neuroscience, 11*, 379. https://doi.org/10.3389/fnins.2017.00379

**Dataset**

7. Kanzler, C.M., Muheim, J., Rinderknecht, M.D., Kulic, D., Lambercy, O., & Gassert, R. (2017). **NinaPro DB7.** *39th Annual International Conference of the IEEE EMBC.* https://ninapro.hevs.ch

8. Atzori, M., et al. (2014). **Electromyography data for non-invasive naturally-controlled robotic hand prostheses.** *Scientific Data, 1*, 140053. https://doi.org/10.1038/sdata.2014.53
   *(Original NinaPro database paper)*
