"""
fusion_dataset_builder.py (Updated to include all micro-expression probability scores)
"""

import os
import pandas as pd
import glob

batch_dir = os.path.join("fusion", "batch_results")
output_file = "fusion_training_dataset.csv"

csv_files = glob.glob(os.path.join(batch_dir, "fusion_batch_results_*.csv"))
if not csv_files:
    raise FileNotFoundError(f"❌ No fusion_batch_results_*.csv found in {batch_dir}")

latest_file = max(csv_files, key=os.path.getctime)
print(f"📄 Using latest batch file: {latest_file}")

df = pd.read_csv(latest_file)

required_cols = ["deep_score", "micro_score", "deep_pred", "micro_emotion", "fusion_result"]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"❌ Missing columns in CSV: {missing_cols}")

# Encode classes
df["deep_class"] = df["deep_pred"].map({"fake": 0, "real": 1})

unique_emotions = sorted(df["micro_emotion"].unique())
emotion_to_id = {emo: idx for idx, emo in enumerate(unique_emotions)}
df["micro_class"] = df["micro_emotion"].map(emotion_to_id)

print(f"🎭 Detected microexpression classes: {unique_emotions}")
print(f"🔢 Class mapping: {emotion_to_id}")

df["fusion_label"] = df["fusion_result"]

# Columns of micro-expression probability scores
micro_prob_cols = [f"micro_score_{emo}" for emo in unique_emotions]
missing_probs = [col for col in micro_prob_cols if col not in df.columns]
if missing_probs:
    raise ValueError(f"❌ Missing micro-expression probability columns: {missing_probs}")

# Build final dataset with deepfake score, deep class, all micro-expression prob columns, micro_class, fusion_label
final_cols = ["deep_score", "deep_class"] + micro_prob_cols + ["micro_class", "fusion_label"]

final_df = df[final_cols]
final_df.to_csv(output_file, index=False)
print(f"✅ Fusion training dataset saved to: {output_file}")
print(f"🎯 Label distribution:\n{final_df['fusion_label'].value_counts()}")
