
from __future__ import annotations
import json
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, roc_auc_score, recall_score, precision_score,
    confusion_matrix, classification_report, average_precision_score
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def _false_positive_rate(y_true, y_pred) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def evaluate(y_true, y_pred, y_score) -> dict:
    return {
        "f1_botnet": round(f1_score(y_true, y_pred, pos_label=1), 4),
        "precision_botnet": round(precision_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "recall_botnet": round(recall_score(y_true, y_pred, pos_label=1), 4),
        "auroc": round(roc_auc_score(y_true, y_score), 4),
        "pr_auc": round(average_precision_score(y_true, y_score), 4),
        "false_positive_rate": round(_false_positive_rate(y_true, y_pred), 4),
    }


def train_and_evaluate(
    X_train, X_test, y_train, y_test,
    models: dict | None = None,
    random_state: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Allena i modelli richiesti (default: RF + XGBoost, come indicato nel
    PDF come stato dell'arte pratico per dati tabellari di rete) e
    restituisce un dizionario {nome_modello: {metriche...}}.
    """
    if models is None:
        models = {
            "RandomForest": RandomForestClassifier(
                n_estimators=200, max_depth=None,
                n_jobs=-1, random_state=random_state,
            ),
        }
        if HAS_XGB:
            models["XGBoost"] = XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                eval_metric="logloss", n_jobs=-1, random_state=random_state,
            )

    results = {}
    fitted_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)[:, 1]

        metrics = evaluate(y_test, y_pred, y_score)
        results[name] = metrics
        fitted_models[name] = model

        if verbose:
            print(f"\n=== {name} ===")
            for k, v in metrics.items():
                print(f"  {k}: {v}")
            print(classification_report(y_test, y_pred, target_names=["benign", "botnet"], zero_division=0))

    return results, fitted_models


def save_results(results: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Risultati salvati in {path}")