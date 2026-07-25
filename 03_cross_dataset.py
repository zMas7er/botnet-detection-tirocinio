import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import load_cic, clean_numeric, split_scale_balance, get_feature_columns, align_cic_features
from src.train_baseline import train_and_evaluate, evaluate, save_results

print("--- FASE 2: ESPERIMENTO CROSS-DATASET (2017 -> 2018) ---\n")

base_dir = os.path.dirname(os.path.abspath(__file__))

# 1. PREPARAZIONE DATASET DI TRAINING (2017)
print("1. Caricamento e preparazione Dataset Sorgente (CIC-IDS2017)...")
cic17_path = os.path.join(base_dir, "data/raw/cic_ids2017")
df_2017 = load_cic(cic17_path, "CIC-IDS2017")
if 'Fwd Header Length.1' in df_2017.columns:
    df_2017 = df_2017.drop(columns=['Fwd Header Length.1'])
df_2017_clean = clean_numeric(df_2017)

X_train_17, X_test_17, y_train_17, y_test_17, scaler_17, feats_17 = split_scale_balance(df_2017_clean, test_size=0.2, use_smote=True, smote_strategy=0.3)

print("\n2. Addestramento dei modelli su CIC-IDS2017...")
results_17, models = train_and_evaluate(X_train_17, X_test_17, y_train_17, y_test_17, verbose=False)

print("\n[Risultati IN-DISTRIBUTION (Test su 2017)]")
print(f"RandomForest -> F1: {results_17['RandomForest']['f1_botnet']} | AUROC: {results_17['RandomForest']['auroc']}")
if 'XGBoost' in results_17:
    print(f"XGBoost      -> F1: {results_17['XGBoost']['f1_botnet']} | AUROC: {results_17['XGBoost']['auroc']}")

save_results(results_17, "results/baseline_2017_in_distribution.json")


# 2. PREPARAZIONE DATASET DI TEST (2018)
print("\n3. Caricamento Dataset Destinazione (CSE-CIC-IDS2018) per il Cross-Dataset...")
cic18_path = os.path.join(base_dir, "data/raw/cse_cic_ids2018")
df_2018 = load_cic(cic18_path, "CSE-CIC-IDS2018")
df_2018_clean = clean_numeric(df_2018)

print("4. Allineamento delle feature (Traduzione 2018 -> 2017)...")
df_2018_aligned = align_cic_features(df_2018_clean)

missing_in_2018 = [c for c in feats_17 if c not in df_2018_aligned.columns]
if missing_in_2018:
    raise ValueError(
        f"ALLINEAMENTO FALLITO: {len(missing_in_2018)} colonne presenti nel "
        f"training 2017 non esistono nel 2018 dopo la rinomina: {missing_in_2018}\n"
        "Questo esperimento cross-dataset NON è valido finché queste colonne "
        "non vengono aggiunte alla mappa in align_cic_features() o rimosse "
        "consapevolmente da entrambi i lati."
    )
print(f"   ✅ Verificato: tutte le {len(feats_17)} feature del training 2017 "
      f"sono presenti nel 2018 allineato.")

X_2018 = df_2018_aligned[feats_17].values
y_2018 = df_2018_aligned['label'].values

print("5. Scalatura dei dati 2018 (usando lo scaler imparato sul 2017)...")
X_2018_scaled = scaler_17.transform(X_2018)


# 3. VALUTAZIONE CROSS-DATASET
print("\n" + "="*50)
print(" 🚨 RISULTATI CROSS-DATASET (Train: 2017 -> Test: 2018) 🚨")
print("="*50)

cross_results = {}
for name, model in models.items():
    print(f"\nValutazione {name}...")
    y_pred_18 = model.predict(X_2018_scaled)
    y_prob_18 = model.predict_proba(X_2018_scaled)[:, 1]
    
    metrics_18 = evaluate(y_2018, y_pred_18, y_prob_18)
    cross_results[name] = metrics_18
    
    print(f"  F1-Score Botnet:     {metrics_18['f1_botnet']}  (Crollo atteso!)")
    print(f"  AUROC:               {metrics_18['auroc']}")
    print(f"  False Positive Rate: {metrics_18['false_positive_rate']}")

print("\n6. Salvataggio risultati...")
save_results(cross_results, "results/cross_dataset_17_to_18.json")
print("--- FASE 2 COMPLETATA ---")