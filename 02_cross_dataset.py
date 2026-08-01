
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from src.preprocessing import load_cic, clean_numeric, get_feature_columns, align_cic_features
from src.experiment_utils import select_threshold, evaluate_full, log_experiment

SEED = 42
base_dir = os.path.dirname(os.path.abspath(__file__))

def run_cross_dataset():
    print("--- 02: ESPERIMENTO CROSS-DATASET (2017 -> 2018) ---")
    
    # ==========================================
    # 1. PREPARAZIONE DATI ORIGINE (2017)
    # ==========================================
    print("\n[1] Caricamento dataset di Origine (CIC-IDS2017)...")
    dir_2017 = os.path.join(base_dir, "data", "raw", "cic_ids2017")
    df_2017 = clean_numeric(load_cic(dir_2017, "CIC-IDS2017"))

    bugged_cols = [c for c in df_2017.columns if c.endswith('.1') or c == 'FwdHeader Length']
    if bugged_cols:
        print(f"Rimozione colonne duplicate/buggate dal 2017: {bugged_cols}")
        df_2017 = df_2017.drop(columns=bugged_cols)
    
    feat_cols_2017 = get_feature_columns(df_2017)
    X_2017 = df_2017[feat_cols_2017].values
    y_2017 = df_2017["label"].values

    # Split 2017 (Train 80% / Val 20% - Non serve il test locale qui)
    X_train_17, X_val_17, y_train_17, y_val_17 = train_test_split(
        X_2017, y_2017, test_size=0.2, stratify=y_2017, random_state=SEED
    )

    print("Scaling e SMOTE sul 2017...")
    scaler = StandardScaler()
    X_train_17_sc = scaler.fit_transform(X_train_17)
    X_val_17_sc = scaler.transform(X_val_17)
    
    smote = SMOTE(sampling_strategy=0.3, random_state=SEED)
    X_train_17_res, y_train_17_res = smote.fit_resample(X_train_17_sc, y_train_17)

    # ==========================================
    # 2. ADDESTRAMENTO E SCELTA SOGLIA (Su 2017)
    # ==========================================
    print("\n[2] Addestramento XGBoost sul 2017...")
    # Usiamo gli iperparametri ottimi trovati dalla baseline
    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, eval_metric="logloss", n_jobs=-1, random_state=SEED)
    xgb.fit(X_train_17_res, y_train_17_res)

    print("Selezione soglia operativa sul Validation 2017 (FPR max 0.1%)...")
    y_prob_val_17 = xgb.predict_proba(X_val_17_sc)[:, 1]
    threshold, thresh_report = select_threshold(y_val_17, y_prob_val_17, max_fpr=0.001)
    print(f"--> SOGLIA CONGELATA: {threshold:.4f}")

    # ==========================================
    # 3. PREPARAZIONE DATI DESTINAZIONE (2018)
    # ==========================================
    print("\n[3] Caricamento dataset di Destinazione (CSE-CIC-IDS2018)...")
    dir_2018 = os.path.join(base_dir, "data", "raw", "cse_cic_ids2018")
    df_2018_raw = load_cic(dir_2018, "CSE-CIC-IDS2018")
    
    df_2018_aligned = align_cic_features(df_2018_raw)
    df_2018_clean = clean_numeric(df_2018_aligned)
    
    # Manteniamo SOLO le colonne che esistevano nel 2017 (nello stesso identico ordine!)
    missing_cols = [c for c in feat_cols_2017 if c not in df_2018_clean.columns]
    if missing_cols:
        raise ValueError(f"Allineamento fallito! Colonne mancanti nel 2018: {missing_cols}")
    
    X_2018 = df_2018_clean[feat_cols_2017].values # <-- Forza l'ordine esatto del 2017
    y_2018 = df_2018_clean["label"].values

    print(f"Check Allineamento Superato: 2017 ha {len(feat_cols_2017)} feature, 2018 ha {X_2018.shape[1]} feature in ordine identico.")

    # ==========================================
    # 4. VALUTAZIONE CROSS-DATASET
    # ==========================================
    print("\n[4] Applicazione modello al 2018 con soglia congelata...")
    X_2018_sc = scaler.transform(X_2018) # Usiamo lo scaler fittato sul 2017!
    
    y_prob_2018 = xgb.predict_proba(X_2018_sc)[:, 1]
    
    # Valutiamo usando la soglia calcolata sul 2017
    metrics = evaluate_full(y_2018, y_prob_2018, threshold)
    
    print("\n--- RISULTATI CROSS-DATASET (Train 17 -> Test 18) ---")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"{k}: {v}")
    print(f"Confusion Matrix: {metrics['confusion_matrix']}")

    log_experiment(
        experiment_name="cross_dataset_17_to_18",
        dataset="CIC-IDS2017 -> CSE-CIC-IDS2018",
        target_family=None,
        seed=SEED,
        split_info={"train_17": len(X_train_17), "val_17": len(X_val_17), "test_18": len(X_2018)},
        hyperparams={"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1, "threshold_used": threshold},
        metrics=metrics,
        y_prob_sample=y_prob_2018,
        extra={"threshold_report": thresh_report}
    )

if __name__ == "__main__":
    run_cross_dataset()