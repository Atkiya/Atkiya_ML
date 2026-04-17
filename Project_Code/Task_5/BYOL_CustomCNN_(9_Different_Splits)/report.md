### BYOL Self-Supervised Learning — Classification Accuracy Across Train/Test Splits

**Backbone:** Custom CNN (513,392 params) 


| Evaluation     | 90:10  | 80:20  | 70:30  | 60:40  | 50:50  | 40:60  | 30:70  | 20:80  | 10:90  |
|----------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| Linear Probe   | 0.6645 | 0.6484 | 0.6499 | 0.6410 | 0.6458 | 0.6570 | 0.6361 | 0.6229 | 0.6034 |
| MLP            | 0.8135 | 0.8063 | 0.8145 | 0.8037 | 0.7712 | 0.7606 | 0.7290 | 0.7088 | 0.6425 |
| SVM            | 0.6172 | 0.5924 | 0.5962 | 0.5929 | 0.5900 | 0.6053 | 0.5803 | 0.5764 | 0.5523 |
| DT             | 0.7724 | 0.7728 | 0.6961 | 0.7559 | 0.6826 | 0.7037 | 0.6787 | 0.6759 | 0.5754 |
| RF             | 0.8659 | 0.8642 | 0.8641 | 0.8613 | 0.8485 | 0.8393 | 0.8198 | 0.8033 | 0.7533 |
| K = 1          | 0.8325 | 0.8301 | 0.8288 | 0.8270 | 0.8199 | 0.8113 | 0.7987 | 0.7824 | 0.7342 |
| K = 5          | 0.8137 | 0.8041 | 0.8025 | 0.7946 | 0.7844 | 0.7725 | 0.7528 | 0.7273 | 0.6695 |
| K = 20         | 0.7582 | 0.7506 | 0.7438 | 0.7314 | 0.7139 | 0.6954 | 0.6690 | 0.6360 | 0.5478 |
| Fine Tune      | 0.9892 | 0.9854 | 0.9799 | 0.9721 | 0.9678 | 0.9568 | 0.9366 | 0.9188 | 0.8601 |

---

### Key Observations

- **Fine-tuning** consistently achieves the highest accuracy across all splits, peaking at **98.92%** on the 90:10 split.
- **Random Forest** is the best-performing frozen-feature head, ranging from 86.59% (90:10) down to 75.33% (10:90).
- **Linear Probe** accuracy is relatively stable (~60–66%) across splits, indicating the frozen BYOL representations encode broadly useful but not perfectly linearly separable features.
- **SVM** is the weakest shallow head in all splits, suggesting the 128-dim embedding space has non-linear class boundaries.
- Performance degrades gracefully as training data shrinks, with Fine Tune still reaching **86.01%** even at the extreme 10:90 split.
