

# DeepCNN – Ninapro DB7 (Gestures 13–29) 🌸

This repository contains a **Deep Convolutional Neural Network (DeepCNN)** trained on **Ninapro DB7**.
The model was evaluated using **9 different train/test splits**.

---

## Dataset

* **Dataset:** Ninapro DB7
* **Subjects:** DB7 subset
* **Input signals:**

  * EMG
  * Accelerometer (ACC)
* **Classes:** 17 gestures (13–29)

---

## Model

* Architecture: DeepCNN
* Input: Windowed EMG + ACC signals
* Output: Gesture classification (17 classes)
* Metric: Accuracy, Precision (weighted), Recall (weighted), F1-score (weighted), ROC-AUC (weighted)

---

## Train/Test Split Comparison

| Split | Train Size | Test Size | Accuracy | Precision (W) | Recall (W) | F1 (W)   | ROC-AUC (W) | Train Time (s) | Test Time (s) |
| ----- | ---------- | --------- | -------- | ------------- | ---------- | -------- | ----------- | -------------- | ------------- |
| 90:10 | 70966      | 8762      | 0.980256 | 0.980425      | 0.980256   | 0.980259 | 0.999803    | 6657.30        | 1.860         |
| 80:20 | 63081      | 17523     | 0.975632 | 0.975729      | 0.975632   | 0.975626 | 0.999775    | 5889.70        | 3.028         |
| 70:30 | 55196      | 26285     | 0.973787 | 0.973978      | 0.973787   | 0.973785 | 0.999687    | 4866.19        | 4.283         |
| 60:40 | 47311      | 35046     | 0.965873 | 0.966095      | 0.965873   | 0.965874 | 0.999517    | 4162.36        | 5.456         |
| 50:50 | 39426      | 43807     | 0.940694 | 0.941861      | 0.940694   | 0.940696 | 0.998942    | 3528.29        | 6.851         |
| 40:60 | 31540      | 52569     | 0.953642 | 0.953963      | 0.953642   | 0.953639 | 0.999106    | 2904.34        | 8.412         |
| 30:70 | 23655      | 61330     | 0.933132 | 0.933693      | 0.933132   | 0.933084 | 0.998336    | 1758.02        | 9.655         |
| 20:80 | 15769      | 70092     | 0.906109 | 0.907322      | 0.906109   | 0.906133 | 0.996834    | 1134.70        | 11.344        |
| 10:90 | 7884       | 78853     | 0.851458 | 0.853671      | 0.851458   | 0.851253 | 0.992625    | 595.62         | 12.841        |

---

## Observations

* Best performance achieved with **90:10 split** (98.02% accuracy).
* Performance decreases as training data size decreases.
* ROC-AUC remains consistently high (>0.99) across most splits.
* Training time scales proportionally with dataset size.

---


