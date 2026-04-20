# DNN Result Comparison: With ACC vs Without ACC

## Overview

This comparison shows the effect of adding accelerometer (ACC) data to the DNN model.

- **With ACC**: DNN uses 48 input channels total  
  - EMG: 12 channels  
  - ACC: 36 channels  

- **Without ACC**: DNN uses 12 EMG channels only  

The goal is to compare classification accuracy between these two settings.

---

## Accuracy Comparison

| Setting        | Input Channels        | Split | Accuracy |
|----------------|----------------------|:-----:|---------:|
| With ACC       | 48 (EMG + ACC)       | 10:90 | 0.6844   |
| Without ACC    | 12 (EMG only)        | 10:90 | 0.4530   |

---

## Difference in Accuracy

- Absolute improvement from adding ACC: **0.2314**  
- This is approximately **23.14 percentage points higher accuracy**

This shows that the DNN performs much better when accelerometer information is included together with EMG.

---

## Conclusion

Using **ACC + EMG** clearly improves DNN performance compared to using **EMG only**.

- With ACC: **68.44% accuracy**  
- Without ACC: **45.30% accuracy**

For the 10:90 split, adding ACC provides a substantial performance gain and makes the DNN much more reliable for gesture classification.
