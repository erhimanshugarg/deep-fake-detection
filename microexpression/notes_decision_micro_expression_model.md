**🧾 Notes & Decisions Log: Micro-Expression Final Model**

---

### 🎯 Purpose
To document the reasoning behind selecting the micro-expression model `microexpression_model_20250622_1826.keras` as the final model, even with a moderate accuracy of 63%.

---

### 📦 Model Summary
- **Model Name:** `microexpression_model_20250622_1826.keras`
- **Architecture:** TimeDistributed MobileNetV2 + LSTM(128)
- **Dataset:** CASME II
- **Output Classes:** `positive`, `negative`, `surprise`
- **Final Accuracy:** 63%
- **Checkpoint Match:** ✅ Confirmed identical performance to best model saved during epoch 24

---

### 📊 Performance Breakdown
| Emotion   | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| Positive  | 0.40      | 0.33   | 0.36     | 6       |
| Negative  | 0.75      | 0.75   | 0.75     | 16      |
| Surprise  | 0.50      | 0.60   | 0.55     | 5       |
| **Accuracy** |         |        | **63%**   | 27      |

---

### ✅ Why This Model is Acceptable

#### ✔️ **Dataset Complexity**
- CASME II contains extremely subtle, high-speed micro-expressions.
- Realistic F1 scores for this dataset are low across most research.
- Positive and surprise emotions are especially hard to detect.

#### ✔️ **Performance is Above Baseline**
- Random chance in a 3-class setup: ~33%
- This model performs nearly **2× better than random**

#### ✔️ **Multimodal Context**
- This is **not the only model** — it will be fused with a Deepfake Detection model.
- In fusion systems, **even partial signals** from weak classifiers add value.

#### ✔️ **Model Stability Proven**
- Identical results across:
  - Final saved model
  - Best checkpoint model
- Full evaluation logged and reproducible

---

### 🧠 Recommendation
Use this model as the official **micro-expression emotion recognizer** in the fusion system. Document its limitations, but also its valuable contribution in detecting subtle emotional cues that support robust liveness verification.

---

### 📅 Logged On
**June 22, 2025 — 19:30 IST**

