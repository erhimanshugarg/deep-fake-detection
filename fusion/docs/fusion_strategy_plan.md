**🔗 Fusion Architecture Planning for Liveness Verification**

**📘 Objective:** To combine two independently trained models — Deepfake Detection and Micro-Expression Recognition — into a single robust binary liveness classification system.

---

**🧠 Models Involved:**

1. **Deepfake Detection Model**\
   📁 `mobilenet_deepfake_model_finetuned.keras`\
   🔍 Predicts: `real` vs `fake` face video (2-class)

2. **Micro-Expression Recognition Model**\
   📁 `microexpression_model_20250622_2034.keras`\
   🔍 Predicts: `positive`, `negative`, or `surprise` emotions (3-class)

---

**🎯 Fusion Method:** ✅ **Late Fusion (Score-Level Decision)**

- Rationale: Both models are pre-trained and independent
- Advantage: Simple, interpretable, modular

---

**📊 Decision Logic (Rule-Based):**

| Deepfake Output | Micro-expression Output  | Final Liveness Decision |
| --------------- | ------------------------ | ----------------------- |
| Real            | Valid Emotion (any of 3) | ✅ Live                  |
| Fake            | No/Invalid Emotion       | ❌ Spoof                 |
| Real            | Ambiguous/Low confidence | ⚠️ Treat as Fake        |
| Fake            | Valid Emotion            | ❌ Spoof                 |

> Thresholds can be adjusted based on confidence scores and F1 balance during evaluation.

---

**📐 Fusion Architecture (Conceptual Flow):**

```
               +------------------------+     +--------------------------+
               |  Deepfake Model (CNN)  |     |  Micro-Expression Model  |
               |  Input: Video frames   |     |  Input: 16-frame seq.    |
               +-----------+------------+     +-----------+--------------+
                           |                          |
                 Deepfake probability         Emotion probability
                           |                          |
                 +---------v--------------------------v--------+
                 |          Fusion Decision Logic              |
                 |  (Rule-based / Weighted average / ML-based) |
                 +--------------------+-------------------------+
                                      |
                              Final Liveness Decision
```

---

**📅 Timeline & Tasks:**

- 🔄 Implement fusion wrapper: 1–2 days
- 🧪 Create demo/test video KYC scenarios: 1–2 days
- 📊 Evaluate fusion model (accuracy, EER, ROC): 2–3 days

---

**✅ Chosen Approach:**

- ✅ **Fusion Type:** Rule-Based Late Fusion
- 🔧 Customizable Thresholds: YES
- 📤 Output: Binary `Live` or `Fake` label per input

> Next: Implement `fusion_predict.py` using this strategy.

