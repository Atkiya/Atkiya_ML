# ShallowConv1D — Supervised EMG Gesture Recognition Baseline

A compact, fully-supervised 1D convolutional neural network for hand gesture classification
from surface EMG and accelerometer signals. Trained and evaluated on **NinaPro DB7 Exercise B**
(17-class, 22 subjects) across nine train/test split ratios. Designed as a reproducible
**supervised baseline** for future self-supervised learning (SSL) experiments.

---

## What this is

The notebook implements a complete end-to-end pipeline:

1. **Data loading** — reads per-subject `.mat` files from NinaPro DB7, extracts EMG (12 ch,
   2000 Hz) and ACC signals, resamples ACC to 2000 Hz, and filters to Exercise B gesture labels
   (G13–G29, 17 classes).
2. **Windowing** — sliding windows of 200 ms with 50% overlap (100 ms step), producing
   fixed-size tensors of shape `(channels × 400)`.
3. **Normalisation** — per-channel z-score, fit on train only and applied to val/test.
4. **Model** — `ShallowConv1D`, a lightweight 1D VGG-style CNN (described below).
5. **Training** — Adam + Cosine Annealing LR, early stopping (patience 20) after a minimum
   of 100 epochs, max 200.
6. **Evaluation** — across nine split ratios from 90:10 to 10:90 (train:test), reporting
   Accuracy, Weighted Precision/Recall/F1, ROC-AUC, wall-clock time, and GFLOPs per window.
7. **Failure mode analysis** — per-class mean accuracy across splits; identifies the best
   split by weighted F1 as the reference for future SSL comparisons.

---

## Model Architecture — `ShallowConv1D`

```
Input  (B, C, 400)            C = 12 EMG + n_acc channels

Block 1   Conv1d(C,   32, k=3) → BN → ReLU
          Conv1d(32,  32, k=3) → BN → ReLU → MaxPool(2)   → (B,  32, 200)

Block 2   Conv1d(32,  64, k=3) → BN → ReLU
          Conv1d(64,  64, k=3) → BN → ReLU → MaxPool(2)   → (B,  64, 100)

Block 3   Conv1d(64,  128, k=3) → BN → ReLU
          Conv1d(128, 128, k=3) → BN → ReLU               → (B, 128, 100)

Global Average Pooling                                     → (B, 128)
Dropout(0.3)
Linear(128, n_classes)                                     → (B, 17)
```

The design follows the VGGNet principle (Simonyan & Zisserman, 2015) adapted to 1D
time-series: small kernels stacked in pairs, batch normalisation after every convolution,
and Global Average Pooling in place of large fully-connected layers to reduce parameter
count and overfitting risk.

---

## Dataset

**NinaPro DB7**
- 22 able-bodied subjects
- 12-channel sEMG + ACC, sampled at 2000 Hz / 148 Hz
- Exercise B: 17 hand and wrist gestures (labels 13–29)
- Labels taken from `restimulus` (movement-onset corrected)

> Kanzler et al. (2017). *NinaPro DB7.* IEEE EMBC. https://ninapro.hevs.ch

---

## Configuration

| Parameter          | Value                        |
|--------------------|------------------------------|
| Sampling rate      | 2000 Hz                      |
| Window length      | 200 ms (400 samples)         |
| Window step        | 100 ms (50 % overlap)        |
| Batch size         | 256                          |
| Optimizer          | Adam (lr=1e-3, wd=1e-4)      |
| Scheduler          | CosineAnnealingLR            |
| Min / Max epochs   | 100 / 200                    |
| Early stop patience| 20 epochs                    |
| Val fraction       | 10 % of train                |
| Seed               | 42                           |

---

## Split Ratios Evaluated

| Train | Test | Purpose                              |
|-------|------|--------------------------------------|
| 90 %  | 10 % | Data-rich upper bound                |
| 80 %  | 20 % | Standard split                       |
| 70 %  | 30 % |                                      |
| 60 %  | 40 % |                                      |
| 50 %  | 50 % | Equal split                          |
| 40 %  | 60 % |                                      |
| 30 %  | 70 % |                                      |
| 20 %  | 80 % | Low-data regime                      |
| 10 %  | 90 % | Extreme low-data / SSL motivation    |

The best split by weighted F1 is flagged as the reference supervised baseline for
SSL experiments.

---

## Outputs

| Path            | Contents                                    |
|-----------------|---------------------------------------------|
| `ckpts_exB/`    | Best model checkpoint per split (`.pt`)     |
| `results_exB/`  | Per-split metrics JSON + summary CSV        |
| `plots_exB/`    | Accuracy & F1 bar chart across all splits   |

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
thop          # optional — for GFLOPs measurement
```

---

## How to Run

Upload to Kaggle with NinaPro DB7 as the input dataset, then run all cells.
The notebook auto-discovers the subject folders under `/kaggle/input`.

```python
if __name__ == '__main__':
    main()
```

---

## Related Work — Pipeline Comparisons

Each subsection below compares our pipeline stage-by-stage against one related work.
The symbol ✓ marks stages that are shared or equivalent; ✗ marks meaningful divergences.

---

### 1. Côté-Allard et al. (2019) — *Closest structural match*

> Côté-Allard, U. et al. "Deep learning for electromyographic hand gesture signal
> classification using transfer learning." *IEEE TNSRE 27*(4), 760–771.
> https://doi.org/10.1109/TNSRE.2019.2896269

**Summary.** This paper tests three ConvNet variants — one on raw EMG, one on STFT
spectrograms, one on CWT images — augmented with inter-subject transfer learning via a
Myo Armband dataset and NinaPro DB5. The raw-EMG ConvNet is architecturally the closest
ancestor to ShallowConv1D.

| Pipeline stage       | ShallowConv1D (ours)                             | Côté-Allard et al. (2019)                              | Match |
|----------------------|--------------------------------------------------|--------------------------------------------------------|-------|
| **Hardware**         | NinaPro DB7 — 12-ch gel EMG + ACC, 2000 Hz       | Myo Armband — 8-ch dry EMG, 200 Hz                     | ✗     |
| **Dataset**          | 22 subjects, 17 gestures (Exercise B)            | 36 subjects (own) + NinaPro DB5, 7 or 18 gestures      | ✗     |
| **Windowing**        | 200 ms window, 100 ms step, 50 % overlap         | 260 ms window; no explicit overlap reported             | ✓ ~   |
| **Input modality**   | Raw 1D EMG + ACC, tensor (C × 400)               | Three paths: raw EMG (1D), spectrogram (2D), CWT (2D)   | ✓ partial |
| **Preprocessing**    | Per-channel z-score normalisation                | Not explicitly described for raw path                  | ✗     |
| **Model structure**  | Stacked double Conv1D-BN-ReLU-Pool, GAP          | Stacked Conv1D-BN-ReLU-Pool, FC head (no GAP)           | ✓ ~   |
| **Model head**       | GAP → Dropout → Linear                           | Flatten → FC layers                                    | ✗     |
| **Training strategy**| Fully supervised from scratch per split          | Pre-train on source subjects, fine-tune on target      | ✗     |
| **Optimiser**        | Adam + CosineAnnealingLR, early stopping         | Not specified in detail                                | —     |
| **Evaluation scope** | 9 train:test ratios, offline                     | Single offline split + 14-day real-time use-case study | ✗     |
| **Metrics**          | Acc, weighted F1, ROC-AUC, GFLOPs               | Offline accuracy only                                  | ✗     |

**Key difference.** The central contribution of Côté-Allard et al. is *transfer learning*
across subjects to reduce annotation cost. Our pipeline deliberately avoids this: we train
from scratch per split precisely to measure how much labelled data the model needs in a
fully supervised regime — establishing the ceiling that a future SSL method should approach.

---

### 2. Sun et al. (2020) — *Compact-CNN design philosophy*

> Sun, Y. et al. "Hand Gesture Recognition Using Compact CNN via Surface
> Electromyography Signals." *Sensors 20*(3), 672.
> https://doi.org/10.3390/s20030672

**Summary.** Proposes a compact 2D CNN for EMG gesture recognition. Signals are first
converted to 2D time-frequency images (short-time Fourier or wavelet), then classified
with a shallow ConvNet emphasising small kernels and low parameter count — evaluated on
NinaPro. The architecture philosophy (kernel-3 throughout, GAP) overlaps with ours, but
the input domain is fundamentally different.

| Pipeline stage       | ShallowConv1D (ours)                             | Sun et al. (2020)                                       | Match |
|----------------------|--------------------------------------------------|----------------------------------------------------------|-------|
| **Hardware**         | NinaPro DB7 — 12-ch gel EMG, 2000 Hz             | NinaPro DB1/DB2 — 10-ch gel EMG, 100 Hz                  | ✓ ~   |
| **Dataset**          | 22 subjects, NinaPro DB7 Exercise B              | NinaPro DB1 / DB2, multiple exercises                   | ✓ ~   |
| **Input modality**   | Raw 1D EMG time-series                           | 2D time-frequency image (STFT or CWT)                   | ✗     |
| **Preprocessing**    | Per-channel z-score → sliding window             | Signal → STFT/CWT → image resize/normalise              | ✗     |
| **Windowing**        | 200 ms sliding window, 50 % overlap              | Frame-based segmentation per image                      | ✓ ~   |
| **Model type**       | 1D CNN (temporal convolutions)                   | 2D CNN (spatial convolutions over image)                | ✗     |
| **Kernel size**      | k=3 throughout                                   | Small kernels (k=3 or 5) throughout                     | ✓     |
| **Pooling**          | MaxPool × 2 + Global Average Pooling             | MaxPool + Global Average Pooling                        | ✓     |
| **Model head**       | GAP → Dropout → Linear                           | GAP → Softmax                                           | ✓     |
| **Parameter count**  | ~50–80 K (estimated)                             | <100 K (compact by design)                              | ✓     |
| **Training strategy**| Fully supervised from scratch                    | Fully supervised from scratch                           | ✓     |
| **Evaluation scope** | 9 split ratios, offline                          | Single split, offline                                   | ✗     |
| **Metrics**          | Acc, weighted F1, ROC-AUC, GFLOPs               | Accuracy, parameter count, inference time               | ✓ ~   |

**Key difference.** Sun et al. solve a 2D image classification problem — the EMG signal is
discarded in favour of its time-frequency representation before it ever reaches the network.
ShallowConv1D operates directly on the raw signal. This means our model can in principle run
at much lower latency (no STFT/CWT preprocessing step) and generalises the compact-kernel +
GAP design to the temporal domain.

---

### 3. Zhai et al. (2017) — *Self-recalibrating CNN for neuroprosthetics*

> Zhai, X., Jelfs, B., Chan, R.H.M., & Tin, C. "Self-recalibrating surface EMG pattern
> recognition for neuroprosthesis control based on convolutional neural network."
> *Frontiers in Neuroscience 11*, 379.
> https://doi.org/10.3389/fnins.2017.00379

**Summary.** Focuses on the practical deployment problem: EMG classifiers degrade over time
as electrode contact shifts and muscle fatigue accumulates. Proposes an online
self-recalibration strategy built around a CNN trained on raw multi-channel EMG windows.
The baseline classifier is architecturally simple (shallow CNN), making it a useful
comparison for our model in terms of structure and problem setting, even though our
pipeline does not address session drift.

| Pipeline stage       | ShallowConv1D (ours)                             | Zhai et al. (2017)                                      | Match |
|----------------------|--------------------------------------------------|----------------------------------------------------------|-------|
| **Hardware**         | NinaPro DB7 — 12-ch gel EMG, 2000 Hz             | Custom setup — 8-ch gel EMG, ~1000 Hz                   | ✓ ~   |
| **Dataset**          | 22 subjects (inter-subject pooled), offline      | Small per-subject dataset, multiple sessions            | ✗     |
| **Input modality**   | Raw 1D multi-channel EMG + ACC                   | Raw 1D multi-channel EMG                                | ✓     |
| **Preprocessing**    | Per-channel z-score                              | Per-channel mean removal                                | ✓ ~   |
| **Windowing**        | 200 ms sliding window, 50 % overlap              | 150–200 ms sliding window, varying overlap              | ✓     |
| **Model type**       | 1D CNN, stacked double-conv blocks               | Shallow 1D CNN, single-conv blocks                      | ✓ ~   |
| **Model depth**      | 3 double-conv blocks (6 conv layers total)       | 2–3 single-conv blocks                                  | ✓ ~   |
| **Pooling**          | MaxPool × 2 + Global Average Pooling             | MaxPool + Flatten + FC                                  | ✗     |
| **Training strategy**| Fully supervised, fixed offline train/test split | Initial supervised training + online self-recalibration | ✗     |
| **Session handling** | Single-session, no drift correction              | Multi-session, explicit drift correction loop           | ✗     |
| **Evaluation scope** | 9 split ratios, offline benchmark                | Per-session accuracy over time (online)                 | ✗     |
| **Metrics**          | Acc, F1, ROC-AUC, GFLOPs                        | Per-session accuracy, recalibration improvement         | ✗     |

**Key difference.** Zhai et al. address *temporal generalisation* — what happens to accuracy
after minutes or hours of use. Our pipeline addresses *data efficiency* — how accuracy
degrades as the size of the labelled training set shrinks. The two questions are orthogonal:
a model could perform well in our evaluation and still degrade badly over a session, and
vice versa. Zhai's recalibration strategy would be a meaningful extension to build on top of
our supervised baseline.

---

### 4. Simonyan & Zisserman (2015) — *Architectural ancestor*

> Simonyan, K., & Zisserman, A. "Very Deep Convolutional Networks for Large-Scale Image
> Recognition." *ICLR 2015.* https://arxiv.org/abs/1409.1556

**Summary.** VGGNet is the direct design ancestor of ShallowConv1D. We adapt its core
principle — repeated blocks of small (3×3) convolutions with batch normalisation, separated
by pooling — from 2D image classification to 1D EMG time-series. This is not an EMG paper;
the comparison below is architectural only.

| Pipeline stage       | ShallowConv1D (ours)                             | VGGNet (Simonyan & Zisserman, 2015)                     | Match |
|----------------------|--------------------------------------------------|---------------------------------------------------------|-------|
| **Domain**           | 1D bio-signal (sEMG + ACC)                       | 2D natural images (ImageNet)                            | ✗     |
| **Convolution type** | Conv1d, temporal                                 | Conv2d, spatial                                         | ✗     |
| **Kernel size**      | k=3 throughout                                   | 3×3 throughout (core insight of the paper)              | ✓     |
| **Block structure**  | Double conv-BN-ReLU per block                    | Double or triple conv-ReLU per block (no BN in orig.)   | ✓ ~   |
| **Batch normalisation** | After every convolution                       | Not in original; added in later reproductions           | ✓ ~   |
| **Pooling**          | MaxPool × 2 → Global Average Pooling             | MaxPool × 5 → three large FC layers                     | ✗     |
| **Head**             | GAP → Dropout(0.3) → Linear(128, 17)             | FC(4096) → FC(4096) → FC(1000)                          | ✗     |
| **Depth**            | 6 conv layers (shallow)                          | 11–19 conv layers (deep — hence "VGG")                  | ✗     |
| **Parameter count**  | ~50–80 K                                         | 138 M (VGG-16)                                          | ✗     |
| **Training data**    | ~few hundred thousand windows                    | 1.2 M ImageNet images                                   | ✗     |

**Key difference.** ShallowConv1D borrows the *micro-architecture* of VGGNet (the 3×3
double-conv-BN-ReLU block) but discards the macro-architecture (depth, large FC head).
GAP replaces the three FC layers, reducing parameters by several orders of magnitude and
making the model appropriate for bio-signal windows rather than megapixel images.

---

## Consolidated Comparison Table

The table below summarises the four comparisons on the dimensions most relevant to
reproducibility and future SSL work.

| Dimension             | ShallowConv1D    | Côté-Allard 2019  | Sun 2020          | Zhai 2017         |
|-----------------------|------------------|-------------------|-------------------|-------------------|
| Input modality        | Raw 1D           | Raw 1D + 2D ×2    | 2D image          | Raw 1D            |
| Channels / rate       | 12 ch · 2000 Hz  | 8 ch · 200 Hz     | 10 ch · 100 Hz    | 8 ch · ~1000 Hz   |
| Window length         | 200 ms           | 260 ms            | ~200 ms           | 150–200 ms        |
| Normalisation         | z-score          | —                 | image-level       | mean removal      |
| Conv type             | 1D               | 1D                | 2D                | 1D                |
| Kernel size           | 3                | 3                 | 3–5               | 3–5               |
| Head                  | GAP + Linear     | Flatten + FC      | GAP + Softmax     | Flatten + FC      |
| Depth (conv layers)   | 6                | ~4–6              | ~4–6              | ~2–4              |
| Training strategy     | Supervised       | Transfer learning | Supervised        | Supervised + online recal. |
| Multi-split eval      | ✓ (9 ratios)     | ✗                 | ✗                 | ✗                 |
| Real-time / online    | ✗                | ✓ (use-case)      | ✗                 | ✓ (core contribution) |
| GFLOPs reported       | ✓                | ✗                 | ✓                 | ✗                 |
| NinaPro dataset       | DB7              | DB5               | DB1 / DB2         | Custom            |

---

## References

**Architecture**

1. Simonyan, K., & Zisserman, A. (2015). **Very Deep Convolutional Networks for Large-Scale
   Image Recognition.** *ICLR 2015.* https://arxiv.org/abs/1409.1556

2. Lin, M., Chen, Q., & Yan, S. (2014). **Network In Network.** *ICLR 2014.*
   https://arxiv.org/abs/1312.4400
   *(introduced Global Average Pooling as a replacement for large FC layers)*

3. Ioffe, S., & Szegedy, C. (2015). **Batch Normalization: Accelerating Deep Network
   Training by Reducing Internal Covariate Shift.** *ICML 2015.*
   https://arxiv.org/abs/1502.03167

**EMG / sEMG deep learning**

4. Côté-Allard, U., Fall, C.L., Drouin, A., Campeau-Lecours, A., Gosselin, C., Glette, K.,
   Laviolette, F., & Gosselin, B. (2019). **Deep learning for electromyographic hand gesture
   signal classification using transfer learning.** *IEEE Transactions on Neural Systems and
   Rehabilitation Engineering, 27*(4), 760–771.
   https://doi.org/10.1109/TNSRE.2019.2896269
   *(closest structural match — raw 1D EMG, stacked conv-BN-ReLU-pool, NinaPro)*

5. Sun, Y., et al. (2020). **Hand Gesture Recognition Using Compact CNN via Surface
   Electromyography Signals.** *Sensors, 20*(3), 672.
   https://doi.org/10.3390/s20030672
   *(compact design philosophy — kernel-3, GAP — but on 2D time-frequency images)*

6. Zhai, X., Jelfs, B., Chan, R.H.M., & Tin, C. (2017). **Self-recalibrating surface EMG
   pattern recognition for neuroprosthesis control based on convolutional neural network.**
   *Frontiers in Neuroscience, 11*, 379.
   https://doi.org/10.3389/fnins.2017.00379
   *(shallow 1D CNN baseline + online recalibration for session drift)*

**Dataset**

7. Kanzler, C.M., Muheim, J., Rinderknecht, M.D., Kulic, D., Lambercy, O., & Gassert, R.
   (2017). **NinaPro DB7.** *39th Annual International Conference of the IEEE EMBC.*
   https://ninapro.hevs.ch

8. Atzori, M., et al. (2014). **Electromyography data for non-invasive naturally-controlled
   robotic hand prostheses.** *Scientific Data, 1*, 140053.
   https://doi.org/10.1038/sdata.2014.53
   *(original NinaPro database paper)*
