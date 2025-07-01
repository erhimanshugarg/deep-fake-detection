"""
📦 fusion_dataset_builder.py
─────────────────────────────────────
Creates training dataset for smart fusion model by extracting
scores, predicted classes, and true fusion outcome from the
most recent batch prediction file.
"""

import os
import pandas as pd
import glob


df = pd.read_csv("batch_results/fusion_batch_results_20250627_1117.csv")
print(df[['deep_label', 'deep_pred']].value_counts())

# === Config ===
batch_dir = "batch_results"
output_file = "fusion_training_dataset.csv"

# === Step 1: Find latest batch file ===
csv_files = glob.glob(os.path.join(batch_dir, "fusion_batch_results_*.csv"))
if not csv_files:
    raise FileNotFoundError("❌ No fusion_batch_results_*.csv found in batch_results/")

latest_file = max(csv_files, key=os.path.getctime)
print(f"📄 Using batch file: {latest_file}")

# === Step 2: Load & filter columns ===
df = pd.read_csv(latest_file)

# Ensure required columns exist
required_cols = ['deep_score', 'micro_score', 'deep_pred', 'micro_emotion', 'fusion_result']
missing = [col for col in required_cols if col not in df.columns]
if missing:
    raise ValueError(f"Missing columns in batch file: {missing}")

# === Step 3: Derive numeric classes ===
df['deep_class'] = df['deep_pred'].map({'fake': 0, 'real': 1})
df['micro_class'] = df['micro_emotion'].map({'positive': 0, 'negative': 1, 'surprise': 2})
df['fusion_label'] = df['fusion_result']  # Keep as string ('SPOOF', 'LIVE')

# === Step 4: Save cleaned dataset ===
final_df = df[['deep_score', 'deep_class', 'micro_score', 'micro_class', 'fusion_label']]
final_df.to_csv(output_file, index=False)
print(f"✅ Fusion training dataset saved to: {output_file}")
print(f"🎯 Label distribution:\n{final_df['fusion_label'].value_counts()}")
