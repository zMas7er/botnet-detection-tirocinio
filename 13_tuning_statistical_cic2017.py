from __future__ import annotations
import json
import os
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, roc_auc_score, recall_score, precision_score, accuracy_score,
    confusion_matrix, average_precision_score
)


from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.preprocessing import load_ctu13, load_cic, clean_numeric, get_feature_columns

base_dir = os.path.dirname(os.path.abspath(__file__))


DATASET = "ctu13"

DATASET_PATHS = {
    "ctu13": os.path.join(base_dir, "data/raw/ctu13"),
    "cic2017": os.path.join(base_dir, "data/raw/cic_ids2017"),
    "cse2018": os.path.join(base_dir, "data/raw/cse_cic_ids2018"),
}

RUN_SVM = False


def _false_positive_rate(y_true, y_pred) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def evaluate(y_true, y_pred, y_score) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_botnet": f1_score(y_true, y_pred, pos_label=1),
        "precision_botnet": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_botnet": recall_score(y_true, y_pred, pos_label=1),
        "auroc": roc_auc_score(y_true, y_score),
        "pr_auc": round(average_precision_score(y_true, y_score), 4),
        "false_positive_rate": _false_positive_rate(y_true, y_pred),
    }




def build_pipeline(model, use_smote: bool = True, random_state: int = 42) -> ImbPipeline:
    steps = [("scaler", StandardScaler())]
    if use_smote:
        steps.append(("smote", SMOTE(sampling_strategy=0.3, random_state=random_state)))
    steps.append(("model", model))
    return ImbPipeline(steps)


def perform_hyperparameter_tuning(X_train, y_train, random_state: int = 42) -> dict:
    print("\n--- INIZIO TUNING DEGLI IPERPARAMETRI ---")
    
    unique, counts = np.unique(y_train, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    minority_count = min(counts_dict.values())
    majority_count = max(counts_dict.values())
    current_ratio = minority_count / majority_count
    
    use_smote = current_ratio < 0.3
    if not use_smote:
        print(f"[SMOTE] Rapporto minoranza/maggioranza attuale ({current_ratio:.4f}) già >= 0.3. SMOTE disattivato automaticamente.")
    else:
        print(f"[SMOTE] Rapporto minoranza/maggioranza attuale ({current_ratio:.4f}) < 0.3. SMOTE attivo (target 0.3).")

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    best_models = {}

    print("Ottimizzazione Random Forest in corso...")
    rf_pipeline = build_pipeline(RandomForestClassifier(random_state=random_state, n_jobs=-1), use_smote=use_smote)
    rf_params = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20],
    }
    rf_grid = GridSearchCV(rf_pipeline, rf_params, cv=cv, scoring="f1", n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    best_models["RandomForest"] = rf_grid.best_estimator_
    print(f"Migliori parametri RF: {rf_grid.best_params_}")

    if HAS_XGB:
        print("Ottimizzazione XGBoost in corso...")
        xgb_pipeline = build_pipeline(
            XGBClassifier(eval_metric="logloss", n_jobs=-1, random_state=random_state), use_smote=use_smote
        )
        xgb_params = {
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 6],
            "model__learning_rate": [0.05, 0.1],
        }
        xgb_grid = GridSearchCV(xgb_pipeline, xgb_params, cv=cv, scoring="f1", n_jobs=-1)
        xgb_grid.fit(X_train, y_train)
        best_models["XGBoost"] = xgb_grid.best_estimator_
        print(f"Migliori parametri XGB: {xgb_grid.best_params_}")

    if RUN_SVM:
        print("Ottimizzazione SVM in corso...")
        svm_pipeline = build_pipeline(SVC(probability=True, random_state=random_state), use_smote=use_smote)
        svm_params = {"model__C": [0.1, 1.0], "model__kernel": ["rbf"]}
        svm_grid = GridSearchCV(svm_pipeline, svm_params, cv=cv, scoring="f1", n_jobs=-1)
        svm_grid.fit(X_train, y_train)
        best_models["SVM"] = svm_grid.best_estimator_
        print(f"Migliori parametri SVM: {svm_grid.best_params_}")

    print("--- FINE TUNING ---")
    return best_models, use_smote



def run_statistical_evaluation(models: dict, X, y, use_smote: bool, iterations: int = 15, test_size: float = 0.3,
                               random_state: int = 42) -> dict:
    print(f"\n--- INIZIO VALIDAZIONE STATISTICA ({iterations} RUN) ---")
    final_results = {}

    for name, pipeline in models.items():
        print(f"\nValutazione modello: {name}")
        accumulated_metrics = {k: [] for k in
                                ["accuracy", "f1_botnet", "precision_botnet",
                                 "recall_botnet", "auroc", "pr_auc", "false_positive_rate"]}

        for i in range(iterations):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=random_state + i
            )

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_score = pipeline.predict_proba(X_test)[:, 1]

            metrics = evaluate(y_test, y_pred, y_score)
            for key in accumulated_metrics:
                accumulated_metrics[key].append(metrics[key])

            print(f"  Run {i + 1}/{iterations} completata (F1: {metrics['f1_botnet']:.4f})")

        final_results[name] = {}
        print(f"\n=== RISULTATI STATISTICI {name} (media su {iterations} run) ===")
        for key, values in accumulated_metrics.items():
            media = float(np.mean(values))
            std_dev = float(np.std(values))
            final_results[name][key] = {
                "mean": round(media, 4),
                "std": round(std_dev, 4),
                "display": f"{media:.4f} ± {std_dev:.4f}",
            }
            print(f"  {key}: {final_results[name][key]['display']}")

    return final_results


def save_results(results: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nRisultati finali salvati in {path}")


if __name__ == "__main__":
    print(f"Caricamento dataset '{DATASET}' tramite il modulo di preprocessing...")
    data_path = DATASET_PATHS[DATASET]

    if DATASET == "ctu13":
        df_raw = load_ctu13(data_path)
    elif DATASET == "cic2017":
        df_raw = load_cic(data_path, "CIC-IDS2017")
    elif DATASET == "cse2018":
        df_raw = load_cic(data_path, "CSE-CIC-IDS2018")
    else:
        raise ValueError(f"DATASET non riconosciuto: {DATASET}")

    df_clean = clean_numeric(df_raw)

    feature_cols = get_feature_columns(df_clean)
    X = df_clean[feature_cols].values
    y = df_clean["label"].values
    print(f"Dati caricati: {X.shape[0]} righe, {X.shape[1]} feature. "
          f"Percentuale botnet: {y.mean():.2%}")

    print("\nEstrazione sottoinsieme (20%) per il tuning...")
    X_tune, _, y_tune, _ = train_test_split(
        X, y, train_size=0.2, stratify=y, random_state=42
    )
    print(f"Sottoinsieme di tuning: {X_tune.shape[0]} righe.")
    best_models, use_smote = perform_hyperparameter_tuning(X_tune, y_tune)

    final_stats = run_statistical_evaluation(best_models, X, y, use_smote=use_smote, iterations=3)

    save_results(final_stats, os.path.join(base_dir, "results", f"{DATASET}_tuned_statistical.json"))