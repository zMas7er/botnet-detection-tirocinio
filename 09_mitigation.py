import os
import glob
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, average_precision_score
import time
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.preprocessing import clean_numeric
from src.train_baseline import evaluate, save_results

print("--- FASE 5: MITIGAZIONE 1 - STATISTICA (15 ITERAZIONI) ---")
base_dir = os.path.dirname(os.path.abspath(__file__))
ctu_path = os.path.join(base_dir, "data/raw/ctu13")
TARGET_ZERO_DAY = "Virut"
ITERATIONS = 15

FAMILY_MAP = {"42": "Neris", "44": "Rbot", "46": "Virut", "47": "Menti"}


def smart_load_ctu(directory, fraction_benign=0.03):
    files = glob.glob(os.path.join(directory, "**", "*.binetflow"), recursive=True)
    chunks_list = []

    for f in files:
        for chunk in pd.read_csv(f, chunksize=100000, low_memory=False):
            label_col = "Label" if "Label" in chunk.columns else chunk.columns[-1]
            is_botnet = chunk[label_col].astype(str).str.contains("Botnet", case=False, na=False)

            botnet_chunk = chunk[is_botnet].copy()
            botnet_chunk['label'] = 1
            scenario_match = botnet_chunk[label_col].astype(str).str.extract(r'V(\d+)')[0]
            botnet_chunk['scenario'] = scenario_match
            botnet_chunk['family'] = botnet_chunk['scenario'].map(FAMILY_MAP).fillna("Unknown_Botnet")

            benign_chunk = chunk[~is_botnet].sample(frac=fraction_benign, random_state=42).copy()
            benign_chunk['label'] = 0
            benign_chunk['family'] = "benign"
            benign_chunk['scenario'] = "0"

            chunks_list.append(botnet_chunk)
            chunks_list.append(benign_chunk)

    df = pd.concat(chunks_list, ignore_index=True)
    leaky = ["StartTime", "SrcAddr", "DstAddr", "Sport", "Dport", "Label"]
    return df.drop(columns=[c for c in leaky if c in df.columns])


print("Caricamento e pulizia globale...")
df = smart_load_ctu(ctu_path)
df_clean = clean_numeric(df)

leaky_features = ['Dur', 'TotBytes', 'SrcBytes', 'TotPkts']
print(f"Rimuovo le feature dataset-specific: {leaky_features}")
feature_cols = [c for c in df_clean.columns
                if c not in ['label', 'family', 'source_dataset', 'scenario']
                and c not in leaky_features]

df_benign_all = df_clean[df_clean['family'] == 'benign']
df_botnet_all = df_clean[df_clean['family'] != 'benign']

accumulated_metrics = {k: [] for k in ["f1_botnet", "precision_botnet", "recall_botnet", "auroc", "pr_auc", "false_positive_rate"]}

for i in range(ITERATIONS):
    t0 = time.time()


    df_benign_train, df_benign_test_pool = train_test_split(
        df_benign_all, test_size=0.2, random_state=42 + i
    )

    df_train = pd.concat([
        df_benign_train,
        df_botnet_all[df_botnet_all['family'] != TARGET_ZERO_DAY]
    ], ignore_index=True)

    n_benign_test = min(5000, len(df_benign_test_pool))
    df_benign_for_test = df_benign_test_pool.sample(n=n_benign_test, random_state=99 + i, replace=False)
    df_target_for_test = df_botnet_all[df_botnet_all['family'] == TARGET_ZERO_DAY]
    df_test = pd.concat([df_benign_for_test, df_target_for_test], ignore_index=True)


    overlap = set(df_benign_for_test.index) & set(df_benign_train.index)
    assert not overlap, f"LEAKAGE RESIDUO all'iterazione {i}: {len(overlap)} righe condivise tra train e test!"

    X_train = df_train[feature_cols].values
    y_train = df_train['label'].values
    X_test = df_test[feature_cols].values
    y_test = df_test['label'].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(sampling_strategy=0.3, random_state=42 + i)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42 + i)
    xgb.fit(X_train_res, y_train_res)

    y_pred = xgb.predict(X_test_scaled)
    y_prob = xgb.predict_proba(X_test_scaled)[:, 1]
    metrics = evaluate(y_test, y_pred, y_prob)

    for key in accumulated_metrics:
        accumulated_metrics[key].append(metrics.get(key, 0.0))

    print(f"  Run {i+1}/{ITERATIONS} completata in {time.time() - t0:.1f}s -> F1: {metrics.get('f1_botnet', 0):.4f}")

print(f"\n=== RISULTATI STATISTICI MITIGAZIONE 1 (Feature Pruning: No Volumetriche) ===")
final_results = {}
for key, values in accumulated_metrics.items():
    media = float(np.mean(values))
    std_dev = float(np.std(values))
    final_results[key] = f"{media:.4f} ± {std_dev:.4f}"
    print(f"  {key}: {final_results[key]}")

os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)
save_results({"XGBoost_Mitigated_Pruning": final_results},
              os.path.join(base_dir, "results", f"mitigation_pruning_{TARGET_ZERO_DAY}.json"))
print(f"\nRisultati salvati in results/mitigation_pruning_{TARGET_ZERO_DAY}.json")