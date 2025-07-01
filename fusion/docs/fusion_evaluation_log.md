## 📘 Fusion Evaluation Log
Logs of predictions made using `fusion_predict.py` and smart fusion classifier.
Tracks model outputs, decisions, and performance.

---

### 📅 June 22, 2025 — 11:33 PM
**🧪 Test Sample #1**  
**Input Sources:**
- 🔹 Deepfake: 1 face frame from `dataset/processed_data/real/...`
- 🔹 Micro-expression: 16-frame sequence from `dataset/microexpression_processed/...`

**Model Predictions:**
- 🎭 **Deepfake Model Output:** `[0.392237]` → **Classified as: Fake**
- 😐 **Micro-Expression Output:** `[0.07985234, 0.8542886, 0.06585906]` → **Classified as: Negative**

**🎯 Final Liveness Decision:** ❌ **SPOOF**

**🧠 Fusion Rule Triggered:**
- Deepfake classified as **Fake**, even with valid emotion → decision = `Spoof`

**📌 Observation:**
- Fusion logic worked as expected ✅
- System correctly treated this case as a spoof due to the deepfake flag

---

### 📅 July 1, 2025 — 10:37 PM
**🏷️ Smart Fusion Classifier — Evaluation Results**

🔍 **Model:** Logistic Regression Fusion Classifier  
📁 Dataset: `fusion_training_dataset.csv`

**📊 Classification Report:**
| Metric        | SPOOF | LIVE |
|---------------|--------|------|
| Precision     | 0.88   | 1.00 |
| Recall        | 1.00   | 0.92 |
| F1-score      | 0.93   | 0.96 |
| **Accuracy**  | **0.95** |

**📊 Confusion Matrix:**
```
              Predicted
             SPOOF   LIVE
Actual SPOOF     7      0
       LIVE      1     12
```

📌 **Files Saved:**
- 📦 Model: `fusion/fusion_classifier_model_20250701_2237.joblib`
- 🧪 Scaler: `fusion/fusion_scaler_20250701_2237.joblib`
- 📈 Confusion Matrix Plot: `fusion/plots/fusion_confusion_matrix_20250701_2237.png`

**🧠 Insight:**
- Classifier successfully learned liveness prediction using combined deepfake and microexpression scores
- Good generalization and balanced precision/recall for both classes

🏁 **Status:** Smart Fusion Classifier **Ready for integration** ✅

