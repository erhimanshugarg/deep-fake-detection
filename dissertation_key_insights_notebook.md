**🗒️ Dissertation Key Insights Notebook: Himanshu Garg**

---

### 🎯 Scope & Research Strategy

- **📘 Title:** `Enhanced Deepfake and Synthetic Media Detection for Robust Liveness Verification in Financial KYC using Multi-Modal and Micro-Expression Analysis`
- **🧩 Focus:** Combine visual forgery detection (**deepfake**) with behavioral authenticity (**micro-expressions**)
- **💡 Innovation:** Late fusion of two independent detection streams to ensure robust liveness detection in financial KYC scenarios

---

### 🏆 Key Achievements (Chronological)

#### 🔹 **June 15, 2025 — Scope Finalization**

- ✅ Defined a two-model approach: Deepfake Detection + Micro-Expression Recognition
- 🔄 Selected late fusion for integrating both models to keep system efficient and explainable

#### 🔹 **June 15, 2025 — Data Preparation**

- 🧼 Preprocessed FaceForensics++ dataset using MTCNN
- 📂 Created structured dataset: `processed_data/real/` and `processed_data/fake/`

#### 🔹 **June 15, 2025 — Baseline Deepfake Model Training**

- 🤖 Trained MobileNetV2 (frozen) on binary deepfake classification
- 💾 Model saved as: `mobilenet_deepfake_model.keras`
- ⚠️ Initial performance: **71% accuracy**, Real Recall = 0.45 (weak generalization)

#### 🔹 **June 17, 2025 — Dataset Expansion & Fine-Tuning**

- 📈 Increased data to 10 real + 10 fake videos
- 🔁 Applied full fine-tuning with augmentation
- 📊 Final Accuracy: **91%**
- 🔍 Confusion Matrix:

```
              Predicted Fake    Predicted Real
Actual Fake       419               22
Actual Real        66              446
```

📋 Comparison Table: 

|📊 Metric             | 🧪 Previous (3 videos/class) | 🚀 Now (10 videos/class) | ✅ Improvement                  |
|----------------------|------------------------------|--------------------------|----------------------------------|
| **Accuracy**         | 0.71                         | **0.91**                | ✅ +20%                           | 
| **Fake Recall**      | 1.00                         | 0.95                    | 🔽 Minor drop                    | 
| **Real Recall**      | 0.45                         | **0.87**                | ✅ Massive improvement           | 
| **F1-score (real)**  | 0.62                         | **0.91**                | ✅                                | 
| **Support**          | 213 samples                  | 953 samples             | 🟢 More robust                   |

---

### 🌟 Novelty & Contribution

- 🔧 Not in just training CNNs — but in **cross-modal integration** of image-based and behavioral-based signals
- 🏦 Target domain: **Financial KYC** — highly relevant, underexplored use case
- 🧠 Planned fusion logic: reliability-weighted voting or hybrid late fusion

---

### 🔜 Upcoming Milestones (as of June 17, 2025)

- 🧠 Begin micro-expression model setup (SMIC/CASME II)
- 🔬 Select architecture: 3D-CNN or CNN + LSTM
- 🔗 Implement late fusion strategy
- 🔍 Evaluate an entire system on spoofing scenarios (KYC test cases)

---

📝 \*Notebook last updated: \**`June 17, 2025`*

