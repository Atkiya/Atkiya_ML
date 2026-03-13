

# EMG Gesture Classification using Deep Neural Network (NinaPro DB7)

This repository contains experiments for **hand gesture classification using EMG signals** from the **NinaPro DB7 dataset**.

A **feed-forward Deep Neural Network (DNN)** is implemented in PyTorch and evaluated under **multiple train–test split configurations** to analyze how training data size affects performance.

The project includes **feature extraction, signal preprocessing, model training, and evaluation using multiple metrics**.

---

# Dataset

Dataset used: **NinaPro Database 7 (DB7)**

Signal types used:

* EMG signals
* Accelerometer (ACC)
* Gesture labels

Experiment setup:

* Gesture classes: **13 – 29 (17 gestures)**
* Window size: **400 samples (200 ms)**
* Step size: **200 samples (100 ms)**

---

# Feature Extraction

Time-domain EMG features were extracted from each sliding window:

| Feature | Description               |
| ------- | ------------------------- |
| MAV     | Mean Absolute Value       |
| ZC      | Zero Crossing             |
| SSC     | Slope Sign Changes        |
| MAVSLP  | Mean Absolute Value Slope |

Configuration:

* **ZC / SSC threshold:** `1e-08`
* **MAVSLP segments:** `3`
* **Feature vector size:** `84`

---

# Model Architecture

The neural network used in the experiment:

```
Input (84 features)
   ↓
Linear (84 → 512)
BatchNorm
ReLU
Dropout (0.2)

Linear (512 → 256)
BatchNorm
ReLU
Dropout (0.2)

Linear (256 → 256)
BatchNorm
ReLU
Dropout (0.2)

Linear (256 → 17)
Output (gesture classes)
```

Training parameters:

| Parameter     | Value      |
| ------------- | ---------- |
| Optimizer     | Adam       |
| Learning rate | 0.005      |
| Weight decay  | 1e-5       |
| Dropout       | 0.2        |
| Device        | CUDA (GPU) |

---

# Evaluation Metrics

Model performance was evaluated using:

| Metric               | Description                           |
| -------------------- | ------------------------------------- |
| Accuracy             | Overall correct predictions           |
| Balanced Accuracy    | Average recall across classes         |
| Precision (Weighted) | Weighted precision across classes     |
| Recall (Weighted)    | Weighted recall                       |
| F1 Score             | Harmonic mean of precision and recall |
| ROC-AUC              | Multiclass ROC AUC                    |

---

# Experiment Results

Performance across **different train–test splits**:

| Train | Test | Accuracy   | Precision  | Recall     | F1 Score   | ROC-AUC    |
| ----- | ---- | ---------- | ---------- | ---------- | ---------- | ---------- |
| 10%   | 90%  | 0.6844     | 0.6877     | 0.6844     | 0.6841     | 0.9672     |
| 20%   | 80%  | 0.7667     | 0.7676     | 0.7667     | 0.7668     | 0.9837     |
| 30%   | 70%  | 0.8288     | 0.8298     | 0.8288     | 0.8289     | 0.9913     |
| 40%   | 60%  | 0.8604     | 0.8612     | 0.8604     | 0.8605     | 0.9943     |
| 50%   | 50%  | 0.8848     | 0.8859     | 0.8848     | 0.8849     | 0.9960     |
| 60%   | 40%  | 0.8963     | 0.8976     | 0.8963     | 0.8964     | 0.9968     |
| 70%   | 30%  | 0.9113     | 0.9117     | 0.9113     | 0.9113     | 0.9975     |
| 80%   | 20%  | 0.9230     | 0.9233     | 0.9230     | 0.9230     | 0.9982     |
| 90%   | 10%  | **0.9283** | **0.9287** | **0.9283** | **0.9283** | **0.9982** |

Observation:

* Performance improves as **training data increases**.
* Best accuracy achieved with **90–10 split (92.83%)**.
* The model maintains **very high ROC-AUC (>0.96)** across all splits.

---

# Visualizations

The notebooks generate:

* Per-class accuracy plots
* Confusion matrices
* ROC curves
* Performance comparison across splits


