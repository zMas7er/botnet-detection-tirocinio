import os
import glob
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.preprocessing import clean_numeric

print("--- FASE 4: AUTOPSIA DEL MODELLO (SHAP) ---")
ctu_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/raw/ctu13")

TARGET_ZERO_DAY = "Rbot"

FAMILY_MAP = {"42": "Neris", "44": "Rbot", "46": "Virut", "47": "Menti"}


def smart_load_ctu(directory, fraction_benign=0.03):
    files = glob.glob(os.path.join(directory, "**", "*.binetflow"), recursive=True)
    chunks_list = []
    for f in files:
        print(f"Lettura a blocchi di: {os.path.basename(f)}...")
        for chunk in pd.read_csv(f, chunksize=100000, low_memory=False):
            label_col = "Label" if "Label" in chunk.columns else chunk.columns[-1]
            is_botnet = chunk[label_col].astype(str).str.contains("Botnet", case=False, na=False)

            botnet_chunk = chunk[is_botnet].copy()
            if not botnet_chunk.empty:
                scenario_match = botnet_chunk[label_col].astype(str).str.extract(r'V(\d+)')[0]
                botnet_chunk['scenario'] = scenario_match
                botnet_chunk['family'] = botnet_chunk['scenario'].map(FAMILY_MAP).fillna("Unknown_Botnet")
                botnet_chunk['label'] = 1

            benign_chunk = chunk[~is_botnet].sample(frac=fraction_benign, random_state=42).copy()
            if not benign_chunk.empty:
                benign_chunk['family'] = "benign"
                benign_chunk['scenario'] = "0"
                benign_chunk['label'] = 0

            chunks_list.append(botnet_chunk)
            chunks_list.append(benign_chunk)

    df = pd.concat(chunks_list, ignore_index=True)
    leaky = ["StartTime", "SrcAddr", "DstAddr", "Sport", "Dport", "Label"]
    return df.drop(columns=[c for c in leaky if c in df.columns])


print("Caricamento e pulizia...")
df = smart_load_ctu(ctu_path)
df_clean = clean_numeric(df)

feature_cols = [c for c in df_clean.columns
                if c not in ['label', 'family', 'source_dataset', 'scenario']]

df_benign_all = df_clean[df_clean['family'] == 'benign']
df_botnet_all = df_clean[df_clean['family'] != 'benign']

df_benign_train, df_benign_test_pool = train_test_split(
    df_benign_all, test_size=0.2, random_state=42
)

df_train = pd.concat([
    df_benign_train,
    df_botnet_all[df_botnet_all['family'] != TARGET_ZERO_DAY]
], ignore_index=True)

n_benign_test = min(100000, len(df_benign_test_pool))
df_benign_for_test = df_benign_test_pool.sample(n=n_benign_test, random_state=42, replace=False)
df_target_for_test = df_botnet_all[df_botnet_all['family'] == TARGET_ZERO_DAY]
df_test_zeroday = pd.concat([df_benign_for_test, df_target_for_test], ignore_index=True)

print(f"Train (famiglie note, senza {TARGET_ZERO_DAY}): {df_train.shape}")
print(f"Test zero-day ({TARGET_ZERO_DAY} + benigno mai visto): {df_test_zeroday.shape}")

X_train = df_train[feature_cols].values
y_train = df_train['label'].values
X_test_zeroday = df_test_zeroday[feature_cols].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_zeroday_scaled = scaler.transform(X_test_zeroday)

print(f"\nAddestramento di XGBoost (identico al LOBO, senza {TARGET_ZERO_DAY})...")
xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42)
xgb.fit(X_train_scaled, y_train)

SAMPLE_SIZE = 3000
rng = np.random.default_rng(42)

idx_train_sample = rng.choice(X_train_scaled.shape[0], size=min(SAMPLE_SIZE, X_train_scaled.shape[0]), replace=False)
X_train_sample = X_train_scaled[idx_train_sample]

idx_test_sample = rng.choice(X_test_zeroday_scaled.shape[0], size=min(SAMPLE_SIZE, X_test_zeroday_scaled.shape[0]), replace=False)
X_test_sample = X_test_zeroday_scaled[idx_test_sample]

explainer = shap.TreeExplainer(xgb)

print("\nCalcolo SHAP - scenario IN-DISTRIBUTION (famiglie note)...")
shap_values_indist = explainer.shap_values(X_train_sample)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_indist, X_train_sample, feature_names=feature_cols, show=False)
plt.title(f"SHAP - Scenario In-Distribution (famiglie note, senza {TARGET_ZERO_DAY})")
os.makedirs("results", exist_ok=True)
plt.savefig("results/shap_summary_in_distribution.png", bbox_inches='tight')
plt.close()
print("✅ Salvato: results/shap_summary_in_distribution.png")

print(f"\nCalcolo SHAP - scenario ZERO-DAY ({TARGET_ZERO_DAY})...")
shap_values_zeroday = explainer.shap_values(X_test_sample)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_zeroday, X_test_sample, feature_names=feature_cols, show=False)
plt.title(f"SHAP - Scenario Zero-Day ({TARGET_ZERO_DAY})")
plt.savefig(f"results/shap_summary_zeroday_{TARGET_ZERO_DAY}.png", bbox_inches='tight')
plt.close()
print(f"✅ Salvato: results/shap_summary_zeroday_{TARGET_ZERO_DAY}.png")

def top_features(shap_values, names, top_n=10):
    importance = np.abs(shap_values).mean(axis=0)
    order = np.argsort(importance)[::-1][:top_n]
    return [(names[i], round(importance[i], 4)) for i in order]

print("\n" + "="*60)
print("TOP 10 FEATURE - IN-DISTRIBUTION vs ZERO-DAY")
print("="*60)
top_indist = top_features(shap_values_indist, feature_cols)
top_zeroday = top_features(shap_values_zeroday, feature_cols)

print(f"\n{'In-Distribution':<30}{'Zero-Day (' + TARGET_ZERO_DAY + ')':<30}")
for (f1, v1), (f2, v2) in zip(top_indist, top_zeroday):
    print(f"{f1 + ' (' + str(v1) + ')':<30}{f2 + ' (' + str(v2) + ')':<30}")

print("\n--- FASE 4 COMPLETATA ---")