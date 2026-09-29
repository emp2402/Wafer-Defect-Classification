# Project Background
This project focuses on automated semiconductor quality control using deep learning to classify silicon wafer defect patterns. In semiconductor fabrication plants (fabs), wafer map patterns provide crucial diagnostics for manufacturing line health. Early detection and accurate classification of pattern defects prevent costly downstream yield losses.

The analysis evaluates a custom Deep Learning pipeline applied to the benchmark WM-811K dataset, comparing various Convolutional Neural Network (CNN) architectures and preprocessing strategies against a baseline model.

Insights and recommendations are provided on the following key areas:

- **Data Preprocessing & Pipeline Engineering:** Impact of one-hot encoding, fixed-size spatial resizing, nearest-neighbor interpolation, and D4 dihedral symmetry augmentation.
- **Model Architecture Exploration:** Performance comparison across Ported VGG-style CNNs, Global Max Pooling (GMP) networks, CoordConv, Squeeze-and-Excitation (SE), and ResNet architectures.
- **Class Imbalance & Re-balancing Dynamics:** Analysis of how combining undersampling, data augmentation, and class weighting affects precision-recall tradeoffs.
- **Metric Evaluation & Production Decision:** Evaluating models on true distribution test sets using Macro F1, Precision, and Recall rather than deceptive overall accuracy.

# Data Structure & Initial Checks

The semiconductor database structure consists of real silicon wafer maps from the WM-811K dataset, evaluated across an imbalanced test set of $n = 17,295$ wafer records[cite: 2]. The primary data components and classes include:

- **Background Channel (Value 0):** Non-wafer grid area surrounding the physical silicon wafer.
- **Passing Die Channel (Value 1):** Functional integrated circuits passing quality checks.
- **Defective Die Channel (Value 2):** Non-functional integrated circuits exhibiting defect patterns.
- **Pattern Classes (9 Total):** `none` (no defect, ~85% majority class) and 8 defect classes: `Center`, `Donut`, `Edge-Loc`, `Edge-Ring`, `Loc`, `Near-full`, `Random`, and `Scratch`.

### WM-811K Dataset Test Set Distribution (Total: 17,295)

| Class / Defect Type | Count | Share (%) |
| :--- | ---: | ---: |
| None | 14,743 | ~85.2% |
| Edge-Ring | 968 | ~5.6% |
| Edge-Loc | 519 | ~3.0% |
| Center | 429 | ~2.5% |
| Loc | 359 | ~2.1% |
| Scratch | 119 | ~0.7% |
| Random | 87 | ~0.5% |
| Donut | 56 | ~0.3% |
| Near-full | 15 | ~0.1% |
| **Total** | **17,295** | **100.0%** |


# Executive Summary

### Overview of Findings

Standard accuracy metrics are severely misleading for wafer defect classification because a naive model predicting `none` every time achieves 85% accuracy while failing completely on defect detection. Evaluation on the true imbalanced distribution ($n = 17,295$) reveals that model depth with residual connections (ResNet) achieves the highest Macro F1 score of 0.911 and Macro Precision of 0.940 without class weights. Furthermore, stacking class weighting on top of sampling and augmentation causes a "stacking paradox" that drastically degrades precision due to over-correction.

<img width="1082" height="505" alt="tab2" src="https://github.com/user-attachments/assets/251dc732-9124-400b-bf7f-ba134ac624a6" />

# Insights Deep Dive
### Data Preprocessing & Pipeline Engineering:

* **One-hot encoding prior to spatial resizing prevents categorical distortion.** Interpolating raw ternary pixel values ($0, 1, 2$) creates physically meaningless fractional values (e.g., $1.5$). Separating maps into three binary planes (`background`, `pass`, `defect`) via `np.eye(3)` followed by nearest-neighbor resizing preserves pristine data integrity.
  
* **Standardizing wafer map size to $54\times54$ balances information retention and computational memory.** Raw wafer maps range from a few dozen pixels up to $212\times204$. Padding smaller maps up to $54\times54$ preserves ~86% of wafers without information loss, avoiding a ~30x RAM overhead compared to sizing all maps to $212\times204$.
  
* **Deterministic D4 Dihedral Symmetry Augmentation outperforms stochastic augmentation.** Applying all 8 exact rotations and reflections to square wafer maps naturally expands minority classes while strictly preserving spatial relationships (e.g., edge defects remain edge defects).
  
* **Dropping background channels optimizes lean architectures.** In Global Max Pooling (GMP) experiments, using a Lambda slice to hand the network only the `pass` and `defect` channels prevented the model from wasting capacity modeling empty space.

### Model Architecture Exploration:

* **ResNet achieved top-tier performance with a Macro F1 of 0.911.** Adding residual skip connections and expanding channel depth to 128 filters allowed the network to learn rich feature hierarchies, significantly improving detection on hard geometric shapes like `Scratch` (F1 increased from 0.79 to 0.83).
  
* **Squeeze-and-Excitation (SE) channel attention ranked second with a Macro F1 of 0.906.** Dynamically re-weighting feature map importance per wafer enabled high performance, matching ResNet's overall accuracy at 0.981.
  
* **Flatten heads outperform Global Max Pooling (GMP) heads by preserving spatial layout.** The best GMP model reached a Macro F1 of only 0.830 because spatial pooling discards spatial location—information critical for detecting position-defined defect types such as `Center`, `Edge-Ring`, and `Edge-Loc`.
  
* **Simply increasing network depth without residual connections degrades performance.** A custom 9-layer Deep VGG CNN achieved a Macro F1 of only 0.840, trailing the baseline model (0.890) and ResNet (0.911) by up to 7 percentage points.

### Class Imbalance & Re-balancing Dynamics:

* **Evaluating on artificially balanced test sets creates a false impression of success.** Initial evaluations on re-balanced test data yielded a ~92% macro score. However, testing on the true imbalanced distribution ($n = 17,295$) exposed significant false-positive rates.
  
* **Applying class weights to pre-balanced data creates a "stacking paradox."** Combining undersampling, D4 augmentation, and class weighting causes double-correction. The model over-corrects toward minority classes, causing precision to collapse.
  
* **Class weighting severely damages precision for sparse classes.** In the baseline model with class weights, `Scratch` recall improved to 90.8%, but false alarms skyrocketed, causing `Scratch` precision to drop from 0.86 to 0.443.
  
* **Removing class weights consistently improves F1 across all architectures.** Removing class weights raised the baseline F1 from 0.859 to 0.890, ResNet F1 from 0.880 to 0.911, and SE F1 from 0.845 to 0.906.






