
import os, sys, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.preprocessing import load_ctu13, clean_numeric, get_feature_columns
from src.experiment_utils import select_threshold, evaluate_full, log_experiment, assert_family_excluded, load_lobo_testset, save_experiment_plots

ITERATIONS = 5
TARGET_ZERO_DAY = "Rbot"
base_dir = os.path.dirname(os.path.abspath(__file__))

def run_mitigation_robust():
    print(f"--- 06: MITIGAZIONE 2 (SOLO ROBUSTE) - TARGET: {TARGET_ZERO_DAY} ---")
    
    raw_dir = os.path.join(base_dir, "data", "raw", "ctu13")
    df_clean = clean_numeric(load_ctu13(raw_dir, fraction_benign=0.1))
    
    # Manteniamo SOLO le feature volumetriche (come da Capitolo 6.3)
    robust_features = ['Dur', 'TotBytes', 'SrcBytes', 'TotPkts']
    feature_cols = [c for c in robust_features if c in df_clean.columns]
    print(f"Feature mantenute (Solo Robuste): {feature_cols}")

    df_train_all = df_clean[df_clean["family"] != TARGET_ZERO_DAY].copy()
    assert_family_excluded(df_train_all, TARGET_ZERO_DAY)

    df_test_zd = load_lobo_testset(TARGET_ZERO_DAY)
    X_test = df_test_zd[feature_cols].values
    y_test = df_test_zd["label"].values

    accumulated_metrics = {k: [] for k in ["f1_botnet", "precision_botnet", "recall_botnet", "auroc", "pr_auc", "false_positive_rate"]}

    for i in range(ITERATIONS):
        X_train_val = df_train_all[feature_cols].values
        y_train_val = df_train_all["label"].values

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, stratify=y_train_val, random_state=42 + i
        )

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_val_sc = scaler.transform(X_val)
        X_test_sc = scaler.transform(X_test)

        smote = SMOTE(sampling_strategy=0.3, random_state=42 + i)
        X_train_res, y_train_res = smote.fit_resample(X_train_sc, y_train)

        xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42 + i)
        xgb.fit(X_train_res, y_train_res)

        y_prob_val = xgb.predict_proba(X_val_sc)[:, 1]
        threshold, thresh_report = select_threshold(y_val, y_prob_val, max_fpr=0.001)

        y_prob_test = xgb.predict_proba(X_test_sc)[:, 1]
        metrics = evaluate_full(y_test, y_prob_test, threshold)

        for key in accumulated_metrics:
            accumulated_metrics[key].append(metrics[key])

        print(f"Run {i+1}/{ITERATIONS} completata -> F1: {metrics['f1_botnet']:.4f} | Recall: {metrics['recall_botnet']:.4f} | Soglia: {threshold:.4f}")

        if i == ITERATIONS - 1:
            exp_name = f"mitigation_robust_{TARGET_ZERO_DAY}"
            log_experiment(exp_name, "CTU-13", TARGET_ZERO_DAY, 42+i, {"train": len(X_train)}, {}, metrics, y_prob_test, {"threshold_report": thresh_report})
            save_experiment_plots(y_test, y_prob_test, threshold, exp_name)

    print("\n=== RISULTATI STATISTICI (MITIGAZIONE 2: SOLO ROBUSTE) ===")
    for key, values in accumulated_metrics.items():
        print(f"  {key}: {np.mean(values):.4f} ± {np.std(values):.4f}")

if __name__ == "__main__":
    run_mitigation_robust()