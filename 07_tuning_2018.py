
import os, sys, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.preprocessing import load_cic, clean_numeric, get_feature_columns
from src.experiment_utils import select_threshold, evaluate_full, log_experiment, save_experiment_plots

DATASET_NAME = "CSE-CIC-IDS2018"
ITERATIONS = 5
base_dir = os.path.dirname(os.path.abspath(__file__))

def plot_tuning_results(cv_results, param_name, model_name):

    plots_dir = os.path.join(base_dir, "results", "plots", "tuning")
    os.makedirs(plots_dir, exist_ok=True)
    
    results_df = pd.DataFrame(cv_results)
    
    plt.figure(figsize=(8, 5))
    # Applicato il fix per Seaborn (aggiunta di "param_")
    sns.lineplot(data=results_df, x=f"param_{param_name}", y='mean_test_score', marker='o')
    plt.title(f"Tuning Parametrico: {model_name} ({param_name}) - {DATASET_NAME}")
    plt.ylabel("F1-Score (Cross-Validation)")
    plt.xlabel(param_name)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    
    clean_param = param_name.replace("model__", "")
    plt.savefig(os.path.join(plots_dir, f"tuning_{DATASET_NAME.lower()}_{model_name}_{clean_param}.png"), dpi=300)
    plt.close()
    print(f"✅ Grafico tuning salvato: {clean_param}")

def build_pipeline(model, use_smote: bool = True):
    if use_smote:
        return ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(sampling_strategy=0.3, random_state=42)),
            ("model", model)
        ])
    else:
        # Se SMOTE è disattivato, usiamo una pipeline standard di scikit-learn senza imblearn
        from sklearn.pipeline import Pipeline
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])

def run_statistical_tuning():
    print(f"--- 07: TUNING E VALIDAZIONE STATISTICA ({DATASET_NAME}) ---")
    
    raw_dir = os.path.join(base_dir, "data", "raw", "cse_cic_ids2018")
    df_raw = load_cic(raw_dir, source_name=DATASET_NAME)
        
    df = clean_numeric(df_raw)
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values
    y = df["label"].values

    print(f"\nDati caricati: {X.shape[0]} righe | Feature: {X.shape[1]} | Botnet: {y.mean():.2%}")

    print("\n[1] Estrazione subset (20%) per Tuning Veloce...")
    X_tune, _, y_tune, _ = train_test_split(X, y, train_size=0.2, stratify=y, random_state=42)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    unique, counts = np.unique(y_tune, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    current_ratio = min(counts_dict.values()) / max(counts_dict.values())
    use_smote = current_ratio < 0.3
    
    if not use_smote:
        print(f"[SMOTE] Rapporto minoranza/maggioranza ({current_ratio:.4f}) >= 0.3. SMOTE disattivato.")
    else:
        print(f"[SMOTE] Rapporto minoranza/maggioranza ({current_ratio:.4f}) < 0.3. SMOTE attivo.")
    
    models = {
        "RandomForest": (RandomForestClassifier(random_state=42, n_jobs=-1),
                         {"model__n_estimators": [100, 200]}),
        "XGBoost": (XGBClassifier(eval_metric="logloss", n_jobs=-1, random_state=42),
                    {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1]})
    }

    best_estimators = {}
    for name, (base_model, params) in models.items():
        print(f"\nGrid Search per {name}...")
        grid = GridSearchCV(build_pipeline(base_model, use_smote=use_smote), params, cv=cv, scoring="f1", n_jobs=-1)
        grid.fit(X_tune, y_tune)
        best_estimators[name] = grid.best_estimator_
        print(f"Migliori parametri {name}: {grid.best_params_}")
        
        param_to_plot = list(params.keys())[0]
        plot_tuning_results(grid.cv_results_, param_to_plot, name)

    print(f"\n[2] Inizio Validazione Statistica ({ITERATIONS} Iterazioni) sull'intero dataset...")
    for name, pipeline in best_estimators.items():
        print(f"\n--- Valutazione: {name} ---")
        accumulated = {k: [] for k in ["f1_botnet", "precision_botnet", "recall_botnet", "auroc", "pr_auc", "false_positive_rate"]}
        
        for i in range(ITERATIONS):
            X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42+i)
            X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, stratify=y_train_val, random_state=42+i)
            
            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_val_sc = scaler.transform(X_val)
            X_test_sc = scaler.transform(X_test)
            
            if use_smote:
                smote = SMOTE(sampling_strategy=0.3, random_state=42+i)
                X_train_res, y_train_res = smote.fit_resample(X_train_sc, y_train)
            else:
                X_train_res, y_train_res = X_train_sc, y_train
            
            model = pipeline.named_steps["model"]
            model.fit(X_train_res, y_train_res)
            
            y_prob_val = model.predict_proba(X_val_sc)[:, 1]
            threshold, thresh_report = select_threshold(y_val, y_prob_val, max_fpr=0.001)
            
            y_prob_test = model.predict_proba(X_test_sc)[:, 1]
            metrics = evaluate_full(y_test, y_prob_test, threshold)
            
            for key in accumulated:
                accumulated[key].append(metrics[key])
            print(f"  Run {i+1}/{ITERATIONS} | F1: {metrics['f1_botnet']:.4f} | Recall: {metrics['recall_botnet']:.4f} | Soglia: {threshold:.4f}")
            
            if i == ITERATIONS - 1:
                exp_name = f"tuning_stats_{DATASET_NAME.lower()}_{name}"
                log_experiment(exp_name, DATASET_NAME, None, 42+i, {"train": len(X_train)}, grid.best_params_, metrics, y_prob_test, {"threshold_report": thresh_report})
                save_experiment_plots(y_test, y_prob_test, threshold, exp_name)

        print(f"\nRisultati Finali {name}:")
        for k, v in accumulated.items():
            print(f"  {k}: {np.mean(v):.4f} ± {np.std(v):.4f}")

if __name__ == "__main__":
    run_statistical_tuning()