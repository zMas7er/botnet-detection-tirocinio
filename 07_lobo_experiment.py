import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split as _tts
from imblearn.over_sampling import SMOTE
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.preprocessing import clean_numeric
from src.train_baseline import evaluate, save_results
from sklearn.metrics import average_precision_score
print("--- FASE 3: ESPERIMENTO LOBO STATISTICO (15 ITERAZIONI) ---")

base_dir = os.path.dirname(os.path.abspath(__file__))
ctu_path = os.path.join(base_dir, "data/raw/ctu13")

FAMILY_MAP = {
    "42": "Neris",
    "44": "Rbot",
    "46": "Virut",
    "47": "Menti"
}

ITERATIONS = 3
TARGETS_ZERO_DAY = ["Virut", "Rbot"]

def smart_load_ctu(directory, fraction_benign=0.05):
    """Legge i file a fette, tiene tutto il malware e campiona il traffico normale."""
    files = glob.glob(os.path.join(directory, "**", "*.binetflow"), recursive=True)
    if not files:
        raise FileNotFoundError("Nessun file .binetflow trovato!")
    
    chunks_list = []
    
    for f in files:
        # Lettura 100.000 righe alla volta
        for chunk in pd.read_csv(f, chunksize=100000, low_memory=False):
            label_col = "Label" if "Label" in chunk.columns else chunk.columns[-1]
            
            # 1. Isolare le botnet
            is_botnet = chunk[label_col].astype(str).str.contains("Botnet", case=False, na=False)
            botnet_chunk = chunk[is_botnet].copy()
            
            # Estrarre il numero dello scenario dalla label
            if not botnet_chunk.empty:
                scenario_match = botnet_chunk[label_col].astype(str).str.extract(r'V(\d+)')[0]
                botnet_chunk['scenario'] = scenario_match
                botnet_chunk['family'] = botnet_chunk['scenario'].map(FAMILY_MAP).fillna("Unknown_Botnet")
                botnet_chunk['label'] = 1
            
            # 2. Isolare il traffico benigno
            benign_chunk = chunk[~is_botnet].sample(frac=fraction_benign, random_state=42).copy()
            if not benign_chunk.empty:
                benign_chunk['family'] = "benign"
                benign_chunk['scenario'] = "0"
                benign_chunk['label'] = 0
                
            chunks_list.append(botnet_chunk)
            chunks_list.append(benign_chunk)

    df_final = pd.concat(chunks_list, ignore_index=True)
    
    leaky = ["StartTime", "SrcAddr", "DstAddr", "Sport", "Dport", "Label"]
    df_final = df_final.drop(columns=[c for c in leaky if c in df_final.columns])
    
    return df_final



print("1. Estrazione e Compressione dei dati in corso...")
df = smart_load_ctu(ctu_path, fraction_benign=0.03)

print("2. Pulizia dati numerici e One-Hot Encoding...")
df_clean = clean_numeric(df)

print(f"\nFamiglie trovate nel dataset:")
print(df_clean['family'].value_counts())

df_benign_all = df_clean[df_clean['family'] == 'benign']
df_botnet_all = df_clean[df_clean['family'] != 'benign']

keep_cols = ['label', 'family', 'source_dataset', 'scenario']
feature_cols = [c for c in df_clean.columns if c not in keep_cols]

all_results = {}

for target in TARGETS_ZERO_DAY:
    print("\n" + "="*60)
    print(f" 🚨 PROTOCOLLO LOBO: ZERO-DAY TARGET -> '{target}' 🚨")
    print("="*60)
    
    accumulated_metrics = {k: [] for k in ["accuracy", "f1_botnet", "precision_botnet", "recall_botnet", "pr_auc", "auroc", "false_positive_rate"]}
    
    for i in range(ITERATIONS):
        t0 = time.time()
        
        df_benign_train, df_benign_test_pool = _tts(df_benign_all, test_size=0.2, random_state=42 + i)
        
        df_train = pd.concat([
            df_benign_train,
            df_botnet_all[df_botnet_all['family'] != target]
        ], ignore_index=True)
        
        # Test Set: Benigno di test + SOLO il target
        n_benign_test = min(100000, len(df_benign_test_pool))
        df_benign_for_test = df_benign_test_pool.sample(n=n_benign_test, random_state=42 + i, replace=False)
        df_target_for_test = df_botnet_all[df_botnet_all['family'] == target]
        df_test = pd.concat([df_benign_for_test, df_target_for_test], ignore_index=True)
        
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
            
        print(f"  Run {i + 1}/{ITERATIONS} completata in {time.time() - t0:.1f}s -> Recall su {target}: {metrics.get('recall_botnet', 0):.4f}")
        
    print(f"\n=== RISULTATI STATISTICI ZERO-DAY ({target}) ===")
    target_results = {}
    for key, values in accumulated_metrics.items():
        media = float(np.mean(values))
        std_dev = float(np.std(values))
        target_results[key] = f"{media:.4f} ± {std_dev:.4f}"
        print(f"  {key}: {target_results[key]}")
        
    all_results[target] = target_results

os.makedirs("results", exist_ok=True)
save_results(all_results, "results/lobo_statistical_results.json")
print("\n--- ESPERIMENTO LOBO COMPLETATO CON SUCCESSO ---")