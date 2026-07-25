from __future__ import annotations
import json
import os
import sys
import time
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, roc_auc_score, recall_score, precision_score, accuracy_score,
    confusion_matrix, average_precision_score
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.preprocessing import load_ctu13, load_cic, clean_numeric, get_feature_columns

base_dir = os.path.dirname(os.path.abspath(__file__))

DATASET = "ctu13"


MAX_ROWS_FOR_SVM = 50000


ITERATIONS = 15

DATASET_PATHS = {
    "ctu13": os.path.join(base_dir, "data/raw/ctu13"),
    "cic2017": os.path.join(base_dir, "data/raw/cic_ids2017"),
    "cse2018": os.path.join(base_dir, "data/raw/cse_cic_ids2018"),
}



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
        "pr_auc": average_precision_score(y_true, y_score),
        "false_positive_rate": _false_positive_rate(y_true, y_pred),
    }


def load_dataset(name: str):
    path = DATASET_PATHS[name]
    if name == "ctu13":
        df_raw = load_ctu13(path)
    elif name == "cic2017":
        df_raw = load_cic(path, "CIC-IDS2017")
    elif name == "cse2018":
        df_raw = load_cic(path, "CSE-CIC-IDS2018")
    else:
        raise ValueError(f"DATASET non riconosciuto: {name}")
    return clean_numeric(df_raw)


if __name__ == "__main__":
    print(f"--- SVM STANDALONE su '{DATASET}' (max {MAX_ROWS_FOR_SVM} righe) ---")

    print("Caricamento dati...")
    df_clean = load_dataset(DATASET)
    feature_cols = get_feature_columns(df_clean)
    X_full = df_clean[feature_cols].values
    y_full = df_clean["label"].values
    print(f"Dati totali: {X_full.shape[0]} righe. Percentuale botnet: {y_full.mean():.2%}")

    if X_full.shape[0] > MAX_ROWS_FOR_SVM:
        print(f"Dataset piu' grande di {MAX_ROWS_FOR_SVM} righe: sottocampiono "
              f"in modo stratificato per contenere i tempi di SVM.")
        X, _, y, _ = train_test_split(
            X_full, y_full,
            train_size=MAX_ROWS_FOR_SVM,
            stratify=y_full,
            random_state=42,
        )
    else:
        X, y = X_full, y_full
    print(f"Righe effettivamente usate per SVM: {X.shape[0]}")

    unique, counts = np.unique(y, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    minority_count = min(counts_dict.values())
    majority_count = max(counts_dict.values())
    current_ratio = minority_count / majority_count

    use_smote = current_ratio < 0.3


    print("\nTuning SVM (griglia ridotta)...")
    t0 = time.time()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    if not use_smote:
        print(f"[SMOTE] Rapporto minoranza/maggioranza attuale ({current_ratio:.4f}) già >= 0.3. SMOTE disattivato.")
        svm_pipeline = ImbPipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(probability=True, random_state=42)),
        ])
    else:
        print(f"[SMOTE] Rapporto attuale ({current_ratio:.4f}) < 0.3. SMOTE attivo.")
        svm_pipeline = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(sampling_strategy=0.3, random_state=42)),
            ("model", SVC(probability=True, random_state=42)),
        ])

    svm_params = {"model__C": [1.0, 10.0], "model__kernel": ["rbf"]}
    svm_grid = GridSearchCV(svm_pipeline, svm_params, cv=cv, scoring="f1", n_jobs=-1)
    svm_grid.fit(X, y)
    print(f"Tuning completato in {time.time() - t0:.1f} secondi.")
    print(f"Migliori parametri SVM: {svm_grid.best_params_}")
    best_svm = svm_grid.best_estimator_

    print(f"\nValidazione statistica SVM ({ITERATIONS} iterazioni)...")
    accumulated_metrics = {k: [] for k in
                            ["accuracy", "f1_botnet", "precision_botnet",
                             "recall_botnet", "pr_auc", "auroc", "false_positive_rate"]}

    for i in range(ITERATIONS):
        t_iter = time.time()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42 + i
        )
        best_svm.fit(X_train, y_train)
        y_pred = best_svm.predict(X_test)
        y_score = best_svm.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, y_pred, y_score)
        for key in accumulated_metrics:
            accumulated_metrics[key].append(metrics[key])
        print(f"  Run {i + 1}/{ITERATIONS} completata in {time.time() - t_iter:.1f}s "
              f"(F1: {metrics['f1_botnet']:.4f})")

    print(f"\n=== RISULTATI STATISTICI SVM su {DATASET} (media su {ITERATIONS} run) ===")
    final_results = {"SVM": {}}
    for key, values in accumulated_metrics.items():
        media = float(np.mean(values))
        std_dev = float(np.std(values))
        final_results["SVM"][key] = {
            "mean": round(media, 4), "std": round(std_dev, 4),
            "display": f"{media:.4f} ± {std_dev:.4f}",
        }
        print(f"  {key}: {final_results['SVM'][key]['display']}")

    final_results["SVM"]["n_rows_used"] = int(X.shape[0])
    final_results["SVM"]["subsampled"] = bool(X_full.shape[0] > MAX_ROWS_FOR_SVM)

    out_path = os.path.join(base_dir, "results", f"svm_only_{DATASET}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(final_results, f, indent=4)
    print(f"\nRisultati salvati in {out_path}")