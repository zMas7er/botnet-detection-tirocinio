
import os, sys, numpy as np, pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.preprocessing import load_ctu13, clean_numeric, get_feature_columns
from src.experiment_utils import assert_family_excluded, load_lobo_testset, select_threshold

SEED = 42
TARGET_ZERO_DAY = "Rbot"
base_dir = os.path.dirname(os.path.abspath(__file__))

def run_shap():
    print(f"--- 05: AUTOPSIA DEL MODELLO (SHAP) - TARGET: {TARGET_ZERO_DAY} ---")
    
    # 1. Caricamento Dati
    raw_dir = os.path.join(base_dir, "data", "raw", "ctu13")
    df_clean = clean_numeric(load_ctu13(raw_dir, fraction_benign=0.1))
    feature_cols = get_feature_columns(df_clean)

    # 2. Preparazione Training (Tutto tranne target)
    df_train_val = df_clean[df_clean["family"] != TARGET_ZERO_DAY].copy()
    assert_family_excluded(df_train_val, TARGET_ZERO_DAY) # Guardia anti-leakage
    
    X_train_val = df_train_val[feature_cols].values
    y_train_val = df_train_val["label"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, stratify=y_train_val, random_state=SEED
    )

    print("\nScaling e SMOTE (solo su Train)...")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    
    smote = SMOTE(sampling_strategy=0.3, random_state=SEED)
    X_train_res, y_train_res = smote.fit_resample(X_train_sc, y_train)

    print("Addestramento XGBoost...")
    hyperparams = {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1}
    xgb = XGBClassifier(**hyperparams, eval_metric="logloss", n_jobs=-1, random_state=SEED)
    xgb.fit(X_train_res, y_train_res)

    print("\nSelezione soglia operativa sul Validation...")
    y_prob_val = xgb.predict_proba(X_val_sc)[:, 1]
    threshold, _ = select_threshold(y_val, y_prob_val, max_fpr=0.001)
    print(f"--> SOGLIA CALCOLATA: {threshold:.4f}")

    # 3. Caricamento Test Set Zero-Day congelato
    print(f"\nCaricamento test set Zero-Day congelato dal disco ({TARGET_ZERO_DAY})...")
    df_test_zd = load_lobo_testset(TARGET_ZERO_DAY)
    X_test_zd = df_test_zd[feature_cols].values
    X_test_zd_sc = scaler.transform(X_test_zd)

    # 4. Calcolo SHAP
    print("\nInizializzazione SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(xgb)
    
    # Sottocampionamento per accelerare SHAP (3000 campioni per tipo)
    SAMPLE_SIZE = 3000
    rng = np.random.default_rng(SEED)
    
    idx_in = rng.choice(X_train_sc.shape[0], size=min(SAMPLE_SIZE, X_train_sc.shape[0]), replace=False)
    X_sample_in = X_train_sc[idx_in]
    
    idx_out = rng.choice(X_test_zd_sc.shape[0], size=min(SAMPLE_SIZE, X_test_zd_sc.shape[0]), replace=False)
    X_sample_out = X_test_zd_sc[idx_out]

    plots_dir = os.path.join(base_dir, "results", "plots", "shap")
    os.makedirs(plots_dir, exist_ok=True)

    print("Calcolo Valori SHAP: In-Distribution...")
    shap_val_in = explainer.shap_values(X_sample_in)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_val_in, X_sample_in, feature_names=feature_cols, show=False)
    plt.title("SHAP - In-Distribution (Famiglie Note)")
    plt.savefig(os.path.join(plots_dir, "shap_in_distribution.png"), bbox_inches='tight')
    plt.close()

    print("Calcolo Valori SHAP: Zero-Day...")
    shap_val_out = explainer.shap_values(X_sample_out)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_val_out, X_sample_out, feature_names=feature_cols, show=False)
    plt.title(f"SHAP - Zero-Day ({TARGET_ZERO_DAY})")
    plt.savefig(os.path.join(plots_dir, f"shap_zeroday_{TARGET_ZERO_DAY}.png"), bbox_inches='tight')
    plt.close()

    # Stampa Top 10 comparativa a terminale
    def top_features(shap_values, names, top_n=10):
        importance = np.abs(shap_values).mean(axis=0)
        order = np.argsort(importance)[::-1][:top_n]
        return [(names[i], round(importance[i], 4)) for i in order]

    top_in = top_features(shap_val_in, feature_cols)
    top_out = top_features(shap_val_out, feature_cols)
    
    print("\n" + "="*60)
    print("TOP 10 FEATURE - IN-DISTRIBUTION vs ZERO-DAY")
    print("="*60)
    print(f"{'In-Distribution':<30} | {'Zero-Day (' + TARGET_ZERO_DAY + ')':<30}")
    for (f1, v1), (f2, v2) in zip(top_in, top_out):
        print(f"{f1 + ' (' + str(v1) + ')':<30} | {f2 + ' (' + str(v2) + ')':<30}")
    print("="*60)
    print(f"✅ Grafici SHAP salvati in {plots_dir}")

if __name__ == "__main__":
    run_shap()