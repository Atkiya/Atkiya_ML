# TinyMyo Supervised Baseline — NinaPro DB7 Exercise B

> **Establishing a fully-supervised reference baseline for EMG gesture classification using the TinyMyo transformer architecture across nine train/test split regimes.**

---

## Table of Contents

- [Overview](#overview)
- [What is TinyMyo?](#what-is-tinymyo)
- [Full TinyMyo vs. This Implementation](#full-tinymyo-vs-this-implementation)
- [Why Supervised-Only? The Design Justification](#why-supervised-only-the-design-justification)
- [Architecture Deep Dive](#architecture-deep-dive)
- [Dataset](#dataset)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [Experimental Design: Nine Split Ratios](#experimental-design-nine-split-ratios)
- [Compute Budget Justification](#compute-budget-justification)
- [Results Interpretation](#results-interpretation)
- [How This Feeds Into SSL Experiments](#how-this-feeds-into-ssl-experiments)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Citation](#citation)

---

## Overview

This notebook implements a **fully supervised** gesture classification pipeline on the [NinaPro DB7](https://ninapro.hevs.ch/) dataset using the transformer backbone from the **TinyMyo** architecture. It is deliberately *not* the full TinyMyo self-supervised learning (SSL) pipeline.

The purpose of this experiment is to:

1. Establish a **supervised upper bound** — the best accuracy achievable when labels are freely available
2. Expose **where supervised-only learning degrades** — specifically at low label regimes (10%–30% training data)
3. Produce a **reference baseline** against which a subsequent SSL experiment can be compared

This is **Phase 1** of a two-phase research project. Phase 2 will introduce full TinyMyo pretraining and measure the accuracy gap that SSL closes in data-scarce conditions.

---

## What is TinyMyo?

TinyMyo is a **self-supervised learning framework** designed specifically for surface EMG signals. Its core idea is borrowed from masked autoencoders (MAE): randomly mask a large fraction (~75%) of signal patches, then train the model to reconstruct the missing patches from context alone — all without any labels.

The result is a transformer backbone that learns **rich, generalizable representations of EMG structure** purely from the signal itself. Once pretrained, only a small labeled dataset is needed to fine-tune a classification head on top.

### Full TinyMyo Pipeline (as designed)

```
Unlabeled EMG data (large corpus)
         │
         ▼
 ┌─────────────────────────────────────┐
 │         STAGE 1: PRETRAINING        │
 │                                     │
 │  Patch Embedding                    │
 │         ↓                           │
 │  Random Masking (~75% of patches)   │
 │         ↓                           │
 │  Transformer Encoder                │
 │   (8 layers, RoPE attention)        │
 │         ↓                           │
 │  PatchReconstructionHead            │
 │         ↓                           │
 │  Reconstruct masked patches         │
 │  Loss: MSE / L1 on masked patches   │
 └─────────────────────────────────────┘
         │
         │  Pretrained backbone weights
         ▼
 ┌─────────────────────────────────────┐
 │       STAGE 2: FINE-TUNING          │
 │                                     │
 │  Freeze or fine-tune backbone       │
 │         ↓                           │
 │  Attach EMGClassificationHead       │
 │         ↓                           │
 │  Train with small labeled set       │
 │  Loss: CrossEntropyLoss             │
 │         ↓                           │
 │  Gesture prediction (17 classes)    │
 └─────────────────────────────────────┘
```

**The key promise of full TinyMyo:** even with very few labels, the pretrained backbone already understands EMG signal structure, so fine-tuning converges fast and generalizes well.

---

## Full TinyMyo vs. This Implementation

| Aspect | Full TinyMyo (original) | This Notebook |
|---|---|---|
| **Training strategy** | SSL pretraining → supervised fine-tune | Supervised only, end-to-end |
| **Stage 1** | Masked patch reconstruction (no labels) | ✗ Skipped entirely |
| **Stage 2** | Fine-tune with few labels | Full supervised training with all labels |
| **Loss function** | Reconstruction loss → CrossEntropyLoss | CrossEntropyLoss only |
| **PatchReconstructionHead** | Core component of pretraining | Defined in code, never invoked |
| **Masking** | ~75% patch masking | None — full signal seen at every step |
| **Initial backbone weights** | Learned from unlabeled EMG structure | Random (Xavier uniform) initialization |
| **Label efficiency** | High — works well with 10% labels | Low — degrades sharply with few labels |
| **Data requirement** | Large unlabeled corpus + small labeled set | Fully labeled dataset required |
| **Purpose** | Production EMG system with minimal annotation | Supervised reference baseline |
| **Expected behavior at 10:90 split** | Relatively robust (pretrained representations) | Significant accuracy drop (expected) |

### The Critical Difference

```
Full TinyMyo:
─────────────────────────────────────────────────────────────────
Unlabeled EMG (large) ──► Pretrain ──► Rich representations
                                               │
Labeled EMG (tiny)    ──────────────► Fine-tune ──► High accuracy
                                                     even at 10% labels

This Notebook:
─────────────────────────────────────────────────────────────────
Labeled EMG (90%)  ──► Train from scratch ──► High accuracy  ✓
Labeled EMG (10%)  ──► Train from scratch ──► Low accuracy   ✗
                               ▲
                    This degradation is what we are measuring.
                    It defines the problem SSL must solve.
```

---

## Why Supervised-Only? The Design Justification

This is the question most likely to arise when reviewing this work. The answer is multi-layered.

### 1. You cannot evaluate SSL without a supervised baseline

A self-supervised system is only meaningful if you can quantify what it *improves* over a system that uses labels directly. This notebook produces that reference point. Without it, claims about SSL performance have no anchor.

### 2. Identifying the low-label breakpoint

The nine split ratios (90:10 down to 10:90) are designed to find the **exact training-data threshold** where supervised learning stops being sufficient. Once identified, SSL pretraining can be targeted precisely at those regimes. Running the full SSL pipeline blindly without this knowledge would be scientifically unjustified.

### 3. Confirming the architecture works on this dataset

NinaPro DB7 Exercise B contains **17 wrist and hand gestures** from 22 subjects, sampled at 2000 Hz — a specific domain that TinyMyo has not necessarily been validated on. This notebook confirms that the transformer backbone is correctly wired, the data loading is correct, windowing is appropriate, and the model can train to convergence at all before SSL complexity is introduced.

### 4. Isolating variables for fair comparison

When Phase 2 introduces SSL pretraining, the **only change** will be the training strategy. Architecture, optimizer, learning rate, windowing, normalization, and evaluation code will remain identical. This is only possible because Phase 1 is established first with everything else fixed.

### 5. Compute realism

Running nine full train-test cycles is already a substantial compute job on Kaggle TPU V5e8. Adding SSL pretraining (which itself requires many epochs) on top of each of the nine splits would be computationally unrealistic in a single session. The supervised baseline is therefore also a **practical necessity** for iterative research.

---

## Architecture Deep Dive

The model used is `TinyMyo` instantiated in classification mode. Here is what each component does:

### Patch Embedding — `PatchEmbedWaveformKeepChans`

```
Input:  (B, C, T)  — batch × channels × time samples
                      e.g. (256, 12, 400)  for 12 EMG channels, 200 ms window

Step 1: Add dummy spatial dim → (B, 1, C, T)
Step 2: Conv2d(1, embed_dim, kernel=(1, patch_size), stride=(1, patch_size))
         → (B, embed_dim, C, T//patch_size)
         → Each channel gets its own non-overlapping temporal patches
Step 3: Rearrange "B D C t → B (C t) D"
         → (B, num_patches, embed_dim)
         → num_patches = C × (T // patch_size) = 12 × 20 = 240

Output: (B, 240, 192)  — sequence of 240 patch tokens, each 192-dim
```

**Why keep channels separate?** Cross-channel interaction is handled by the attention mechanism, not conflated at the patch level. This preserves per-channel signal identity.

### Rotary Positional Embeddings (RoPE)

Standard absolute positional embeddings inject position as a fixed additive vector. RoPE instead **rotates** query and key vectors in attention by an angle proportional to their position:

```
q_rotated = RoPE(q, position)
k_rotated = RoPE(k, position)

The dot product q·k then naturally encodes relative distance between positions,
making attention position-aware without learned position parameters.
```

RoPE is particularly well suited to EMG because:
- EMG gesture onset timing varies between repetitions — relative position matters more than absolute
- It generalizes better to sequence lengths not seen during training
- No additional parameters are introduced

### Rotary Transformer Block

Each of the 8 blocks follows the standard pre-norm transformer architecture:

```
x = x + DropPath(RotarySelfAttention(LayerNorm(x)))
x = x + DropPath(MLP(LayerNorm(x)))
```

With:
- `embed_dim = 192`, `num_heads = 3`, `head_dim = 64`
- `mlp_ratio = 4` → hidden MLP dim = 768
- `attn_drop = proj_drop = drop_path = 0.1`

Weight rescaling (`fix_init_weight`) divides each layer's projection weights by `√(2 × layer_id)` to stabilize gradients in deep networks.

### Classification Head — `EMGClassificationHead` (concat reduction)

```
Input:  (B, num_patches, embed_dim) = (B, 240, 192)

Step 1: Rearrange "b (c p) d → b p (c d)"
         → (B, 20, 12×192) = (B, 20, 2304)
         Temporal patches, with all channels concatenated per patch

Step 2: Mean over temporal patches → (B, 2304)
         Single feature vector per window

Step 3: Linear(2304, 17) → (B, 17) logits

Output: Class probabilities over 17 gestures
```

The `concat` reduction is chosen over `mean` because it preserves inter-channel relationships in the feature vector. For multi-channel EMG, channel co-activation patterns are discriminative and should not be averaged away.

### Model Scale

| Hyperparameter | Value | Rationale |
|---|---|---|
| `embed_dim` | 192 | ViT-Tiny standard; sufficient for 17-class task |
| `n_layer` | 8 | Enough depth for temporal-spatial reasoning |
| `n_head` | 3 | head_dim=64, well-established for stability |
| `patch_size` | 20 | 10 ms per patch at 2000 Hz — captures motor unit firing |
| `img_size` | 400 | 200 ms window at 2000 Hz |
| `in_chans` | 12 (+ ACC) | NinaPro DB7 EMG electrode count |

---

## Dataset

**NinaPro DB7 — Exercise B**

| Property | Value |
|---|---|
| Subjects | 22 |
| EMG channels | 12 |
| ACC channels | variable per subject |
| Sampling rate | 2000 Hz (EMG), 148 Hz (ACC, upsampled) |
| Gesture range | Labels 13–29 (Exercise B) |
| Number of classes | 17 |
| Window length | 200 ms (400 samples) |
| Window step | 100 ms (50% overlap) |
| Label strategy | Majority vote within window |

Exercise B focuses on **wrist and hand gestures** including flexion, extension, radial/ulnar deviation, and combined movements. These are among the most clinically relevant gestures for prosthetic control.

### Preprocessing Steps

1. Load `.mat` files per subject
2. Extract EMG and ACC arrays; upsample ACC to match EMG length via `resample_poly`
3. Filter rows to Exercise B gesture range (labels 13–29)
4. Slide 400-sample windows with 200-sample steps
5. Assign window label by majority vote across timesteps
6. Per-channel z-score normalization fit on training split only

---

## Pipeline Walkthrough

```
┌──────────────────────────────────────────────────────────────────┐
│  STEP 1 — Data Loading                                           │
│                                                                  │
│  SubjectLoader iterates over 22 subjects                         │
│  EMGAccPreprocessor loads .mat, resamples ACC, filters Ex.B      │
│  Results cached as .npz per subject for reuse across splits      │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│  STEP 2 — Windowing                                              │
│                                                                  │
│  Slide 400-sample windows, 200-sample step                       │
│  X: (N, C, 400)   y: (N,)  zero-indexed class labels            │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│  STEP 3 — Split (for each of 9 ratios)                           │
│                                                                  │
│  Stratified train/test split                                     │
│  Val carved from train (10% of train)                            │
│  ChannelNormalizer fit on train, applied to val + test           │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│  STEP 4 — Training                                               │
│                                                                  │
│  TinyMyo (task="classification") initialized with random weights │
│  Adam optimizer, lr=1e-3, weight_decay=1e-4                      │
│  CosineAnnealingLR scheduler                                     │
│  Min 100 epochs, max 200, early stop patience=20                 │
│  Best checkpoint saved by validation loss                        │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│  STEP 5 — Evaluation                                             │
│                                                                  │
│  Load best checkpoint                                            │
│  Compute: Accuracy, Precision, Recall, F1 (weighted), ROC-AUC    │
│  Per-class accuracy breakdown                                    │
│  Confusion matrix (count + normalised), ROC curves              │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│  STEP 6 — Failure Mode Analysis                                  │
│                                                                  │
│  Aggregate per-class accuracy across all 9 splits               │
│  Identify worst-performing gesture classes                       │
│  Identify best split by weighted F1                              │
│  Output: reference split for SSL Phase 2                         │
└──────────────────────────────────────────────────────────────────┘
```

---

## Experimental Design: Nine Split Ratios

The nine train:test ratios are:

| Split | Train % | Test % | Purpose |
|---|---|---|---|
| 90:10 | 90% | 10% | Near-maximum label availability |
| 80:20 | 80% | 20% | Standard ML benchmark split |
| 70:30 | 70% | 30% | Common research split |
| 60:40 | 60% | 40% | Moderate label reduction |
| 50:50 | 50% | 50% | Symmetric split |
| 40:60 | 40% | 60% | Low label regime begins |
| 30:70 | 30% | 70% | Low label regime |
| 20:80 | 20% | 80% | Very low label regime |
| 10:90 | 10% | 90% | Extreme label scarcity |

This sweep serves two purposes:

1. **Maps the accuracy curve** as a function of label availability — the shape of this curve tells us how data-hungry pure supervised learning is on this task
2. **Pinpoints the crossover** — the split ratio below which supervised accuracy drops sharply enough that SSL pretraining becomes worthwhile

---

## Compute Budget Justification

### Why tiny and not full TinyMyo?

A full TinyMyo variant would typically use `embed_dim=384` or `embed_dim=768` with 12+ layers. The cost scaling is:

```
Attention cost   ∝  O(n² × d)   where d = embed_dim
MLP cost         ∝  O(n × d²)

Going from embed_dim=192 → 384:
  Attention: 2× per layer
  MLP:       4× per layer
  Total per forward pass: ~3-4×

9 splits × 200 epochs × 3-4× cost = guaranteed TPU session timeout
```

The tiny configuration (`embed_dim=192`, 8 layers) has sufficient capacity for a 17-class problem on 400-sample windows. Scaling up adds compute cost without meaningful accuracy gain at this task complexity — and would prevent completing the full split sweep, which is the entire point of this experiment.

---

## Results Interpretation

After running all nine splits, the `analyse_failure_modes` function produces:

- **Per-class mean accuracy** ranked worst to best — reveals which gestures are inherently confusable (e.g. radial vs. ulnar deviation which share similar muscle activation patterns)
- **Best split by weighted F1** — this becomes the fixed train/test partition used in all subsequent SSL experiments
- **Summary bar chart** — accuracy and F1 across all splits, showing the degradation curve

The degradation curve shape is the key scientific output. A steep drop at low label fractions confirms that **SSL pretraining is warranted** and motivates Phase 2.

---

## How This Feeds Into SSL Experiments

```
This Notebook                    Phase 2 (SSL)
─────────────────────────────────────────────────────────────
Supervised accuracy at           SSL accuracy at
each split ratio         ──►     same split ratios
         │                               │
         └────────────── GAP ────────────┘
                           │
                    This gap is the
                  research contribution.
                  
Largest gap expected at:
  10:90, 20:80, 30:70 splits
  → SSL closes the gap because pretrained
    representations don't need many labels
```

The best split identified here (by weighted F1) will be used as the **fixed evaluation condition** in all SSL ablation studies, ensuring a controlled comparison.

---

## Project Structure

```
├── test-tinymyo-1to2-updated.ipynb   # This notebook
├── README.md                         # This file
└── /kaggle/working/
    ├── cache_exB/                    # Per-subject .npz caches
    ├── ckpts_exB/                    # Best model checkpoints per split
    ├── plots_exB/                    # Confusion matrices, ROC curves, learning curves
    └── results_exB/                  # Per-split JSON metrics + summary CSV
```

---

## Dependencies

```
torch
timm
einops
numpy
pandas
scipy
scikit-learn
matplotlib
seaborn
tqdm
thop          # optional, for GFLOPs measurement
```

Accelerator: **Kaggle TPU V5e8**
Python: 3.12

---

## Citation

If you use this baseline or the TinyMyo architecture in your work, please cite the original TinyMyo paper and the NinaPro DB7 dataset.

```bibtex
@dataset{ninapro_db7,
  title   = {NinaPro Database 7},
  author  = {Krasoulis, Agamemnon and others},
  url     = {https://ninapro.hevs.ch/}
}
```

---

*This repository contains Phase 1 of a two-phase EMG gesture classification study. Phase 2 (SSL pretraining with masked patch reconstruction) will be released separately.*
