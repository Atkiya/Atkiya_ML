## BYOL Self-Supervised Learning — Ablation Study on SSL Pre-training Epochs

**Backbone:** Custom CNN (513,392 params) | **Feature Dim:** 128 | **Classes:** 17 (G13–G29)

### Fine-Tune Accuracy Ablation: SSL Pre-training Epochs vs. Train:Test Split Ratio

| Split Ratio | 250 Epochs | 200 Epochs | 150 Epochs |
|-------------|------------|------------|------------|
| 90:10       | **0.9892** | **0.9877** | **0.9872** |
| 80:20       | 0.9854     | 0.9856     | 0.9845     |
| 70:30       | 0.9799     | 0.9781     | 0.9785     |
| 60:40       | 0.9721     | 0.9708     | 0.9693     |
| 50:50       | 0.9678     | 0.9670     | 0.9671     |
| 40:60       | 0.9568     | 0.9558     | 0.9549     |
| 30:70       | 0.9366     | 0.9355     | 0.9351     |
| 20:80       | 0.9188     | 0.9175     | 0.9216     |
| 10:90       | 0.8601     | 0.8690     | 0.8656     |

---

### Key Observations

- **More pre-training epochs generally help**, but gains diminish quickly: going from 150 → 250 epochs yields at most ~0.3 pp improvement at data-rich splits (90:10).
- **Split ratio has a far greater impact than epoch count.** Reducing training data from 90:10 to 10:90 costs ~13 pp of accuracy, while halving epochs (250 → 150) costs less than 1 pp at most splits.
- **At data-scarce splits (10:90)**, 200 epochs slightly outperforms 250 epochs (86.90% vs. 86.01%), suggesting that shorter pre-training may generalize better when labelled fine-tune data is very limited.
- **The 90:10 split with 250 epochs** yields the best overall result at **98.92%**, making it the recommended configuration when labelled data is abundant.
- **All three epoch settings remain viable** — the worst-case gap between 150 and 250 epochs across all splits is under 1.3 pp, confirming BYOL's robustness to moderate changes in pre-training duration.
