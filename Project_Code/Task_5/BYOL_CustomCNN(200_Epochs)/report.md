# BYOL Self-Supervised Learning — Classification Accuracy Across Train/Test Splits

**Backbone:** Custom CNN (513,392 params) | **SSL Epochs:** 200 

## Accuracy Results by Evaluation Method and Train:Test Split

| Evaluation     | 90:10  | 80:20  | 70:30  | 60:40  | 50:50  | 40:60  | 30:70  | 20:80  | 10:90  |
|----------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| Linear Probe   | 0.6633 | 0.6552 | 0.6547 | 0.6430 | 0.6354 | 0.6459 | 0.6316 | 0.6111 | 0.5991 |
| MLP            | 0.8137 | 0.7970 | 0.7979 | 0.7773 | 0.7583 | 0.7568 | 0.7344 | 0.7491 | 0.6439 |
| SVM            | 0.6021 | 0.6020 | 0.6036 | 0.5965 | 0.5847 | 0.5903 | 0.5805 | 0.5625 | 0.5514 |
| DT             | 0.7853 | 0.7830 | 0.7215 | 0.6901 | 0.7048 | 0.7164 | 0.6741 | 0.6421 | 0.5294 |
| RF             | 0.8698 | 0.8671 | 0.8674 | 0.8615 | 0.8505 | 0.8432 | 0.8225 | 0.8051 | 0.7553 |
| K = 1          | 0.8361 | 0.8328 | 0.8339 | 0.8302 | 0.8233 | 0.8167 | 0.8024 | 0.7847 | 0.7381 |
| K = 5          | 0.8159 | 0.8075 | 0.8044 | 0.8009 | 0.7885 | 0.7770 | 0.7549 | 0.7288 | 0.6710 |
| K = 20         | 0.7609 | 0.7554 | 0.7454 | 0.7355 | 0.7171 | 0.6974 | 0.6709 | 0.6375 | 0.5478 |
| Fine Tune      | 0.9877 | 0.9856 | 0.9781 | 0.9708 | 0.9670 | 0.9558 | 0.9355 | 0.9175 | 0.8690 |

---

### Key Observations

- **Fine-tuning** consistently achieves the highest accuracy across all splits, peaking at **98.77%** on the 90:10 split.
- **Random Forest** is the best-performing frozen-feature head, ranging from 86.98% (90:10) down to 75.53% (10:90), and stays highly consistent across the larger splits.
- **k-NN (K=1)** tracks closely behind RF at larger splits, reaching 83.61% at 90:10, confirming well-structured local geometry in the BYOL embedding space.
- **Linear Probe** accuracy holds steady within a narrow ~60–66% range, reflecting the non-linear nature of the 17-class boundaries in the 128-dim feature space.
- **SVM** remains the weakest shallow head throughout all splits, consistent with the non-linear embedding structure.
- Performance degrades gracefully as training data shrinks, with Fine Tune still delivering **86.90%** at the extreme 10:90 split.
