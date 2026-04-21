# BYOL Self-Supervised Learning — Classification Accuracy Across Train/Test Splits

**Backbone:** Custom CNN (513,392 params) | **SSL Epochs:** 150 

## Accuracy Results by Evaluation Method and Train:Test Split

| Evaluation     | 90:10  | 80:20  | 70:30  | 60:40  | 50:50  | 40:60  | 30:70  | 20:80  | 10:90  |
|----------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| Linear Probe   | 0.6545 | 0.6486 | 0.6471 | 0.6332 | 0.6374 | 0.6433 | 0.6266 | 0.6135 | 0.5887 |
| MLP            | 0.7802 | 0.8101 | 0.7856 | 0.7995 | 0.7716 | 0.7776 | 0.7301 | 0.7467 | 0.6576 |
| SVM            | 0.5968 | 0.5938 | 0.6043 | 0.5862 | 0.5869 | 0.5856 | 0.5759 | 0.5655 | 0.5452 |
| DT             | 0.7658 | 0.7713 | 0.7325 | 0.7144 | 0.7188 | 0.7221 | 0.6887 | 0.6430 | 0.5555 |
| RF             | 0.8713 | 0.8715 | 0.8711 | 0.8626 | 0.8552 | 0.8457 | 0.8255 | 0.8079 | 0.7560 |
| K = 1          | 0.8374 | 0.8387 | 0.8431 | 0.8366 | 0.8306 | 0.8232 | 0.8076 | 0.7887 | 0.7414 |
| K = 5          | 0.8148 | 0.8160 | 0.8126 | 0.8065 | 0.7960 | 0.7812 | 0.7597 | 0.7334 | 0.6747 |
| K = 20         | 0.7625 | 0.7571 | 0.7508 | 0.7380 | 0.7231 | 0.7048 | 0.6713 | 0.6396 | 0.5501 |
| Fine Tune      | 0.9872 | 0.9845 | 0.9785 | 0.9693 | 0.9671 | 0.9549 | 0.9351 | 0.9216 | 0.8656 |


---

### Key Observations

- **Fine-tuning** consistently achieves the highest accuracy across all splits, peaking at **98.72%** on the 90:10 split.
- **Random Forest** is the best-performing frozen-feature head, ranging from 87.15% (80:20) down to 75.60% (10:90), and remains remarkably stable across the larger splits.
- **k-NN (K=1)** performs competitively with RF at larger splits, reaching 84.31% at 70:30, suggesting the BYOL embedding space is locally well-structured.
- **Linear Probe** accuracy stays within a narrow band (~59–65%) across all splits, reflecting that the 128-dim frozen features are not linearly separable for all 17 classes.
- **SVM** is consistently the weakest shallow head, indicating non-linear decision boundaries in the embedding space.
- Performance degrades gracefully as training data shrinks, with Fine Tune still achieving **86.56%** at the extreme 10:90 split.
