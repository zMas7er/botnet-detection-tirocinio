
from __future__ import annotations
import os
import json
import csv
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, roc_auc_score, recall_score, precision_score,
    confusion_matrix, average_precision_score, roc_curve,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
LOG_CSV = os.path.join(RESULTS_DIR, "experiment_log.csv")


# ==========================================================================
# 1. GUARDIA ANTI-LEAKAGE DI FAMIGLIA (CTU-13)
# ==========================================================================
def assert_family_excluded(df_train: pd.DataFrame, target_family: str):
    
    present = df_train["family"].unique().tolist()
    if target_family in present:
        raise RuntimeError(
            f"LEAKAGE DI FAMIGLIA: '{target_family}' presente nel training! "
            f"Famiglie in train: {present}"
        )
    # Se il loader marca famiglie non riconosciute come Unknown_Botnet,
    # e la funzione chiamante non ha filtrato per scenario/family map
    # aggiornata, qui non possiamo saperlo con certezza: e' compito dello
    # script chiamante usare SEMPRE load_ctu13() di preprocessing.py (che
    # deriva family dal nome della cartella, unica fonte di verita') e MAI
    # una regex separata sulla label.


# ==========================================================================
# 2. SELEZIONE E CONGELAMENTO DELLA SOGLIA
# ==========================================================================
def select_threshold(y_val, y_prob_val, max_fpr: float = 0.001, objective: str = "max_recall_under_fpr"):
    
    fpr, tpr, thresholds = roc_curve(y_val, y_prob_val)
    if objective == "max_recall_under_fpr":
        valid = fpr <= max_fpr
        if not valid.any():
            # nessuna soglia rispetta il vincolo: si prende la piu' severa disponibile
            best_idx = np.argmin(fpr)
        else:
            best_idx = np.argmax(tpr[valid])
            best_idx = np.where(valid)[0][best_idx]
    elif objective == "youden":
        j = tpr - fpr
        best_idx = np.argmax(j)
    else:
        raise ValueError(f"objective non riconosciuto: {objective}")

    threshold = float(thresholds[best_idx])
    report = {
        "objective": objective,
        "max_fpr_constraint": max_fpr,
        "threshold": threshold,
        "validation_fpr_at_threshold": float(fpr[best_idx]),
        "validation_tpr_at_threshold": float(tpr[best_idx]),
    }
    return threshold, report


def apply_threshold(y_prob, threshold: float):
    return (y_prob >= threshold).astype(int)


# ==========================================================================
# 3. VALUTAZIONE COMPLETA (con soglia esplicita, mai model.predict() implicito)
# ==========================================================================
def evaluate_full(y_true, y_prob, threshold: float) -> dict:
    y_pred = apply_threshold(y_prob, threshold)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "threshold_used": threshold,
        "n_positive_predictions": int(y_pred.sum()),
        "n_test_total": int(len(y_true)),
        "f1_botnet": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "precision_botnet": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall_botnet": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "auroc": round(roc_auc_score(y_true, y_prob), 4),
        "pr_auc": round(average_precision_score(y_true, y_prob), 4),
        "false_positive_rate": round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0.0,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


# ==========================================================================
# 4. LOGGING UNIFICATO (JSON completo + riga CSV)
# ==========================================================================
def _sanitize(obj):
    """Converte ricorsivamente tipi numpy (bool_, int64, float64, ndarray)
    in tipi Python nativi, altrimenti json.dump fallisce."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def log_experiment(
    experiment_name: str,
    dataset: str,
    target_family: str | None,
    seed: int,
    split_info: dict,
    hyperparams: dict,
    metrics: dict,
    y_prob_sample: np.ndarray | None = None,
    extra: dict | None = None,
):
    """
    Salva UN file JSON completo per l'esperimento (in results/logs/) e
    aggiunge una riga riassuntiva a results/experiment_log.csv.
    y_prob_sample: array di probabilita' (o un campione, se enorme) salvato
    per poter ricostruire in seguito curva PR / distribuzione senza
    dover rilanciare il training.
    """
    os.makedirs(os.path.join(RESULTS_DIR, "logs"), exist_ok=True)
    record = {
        "experiment_name": experiment_name,
        "dataset": dataset,
        "target_family": target_family,
        "seed": seed,
        "split_info": split_info,
        "hyperparams": hyperparams,
        "metrics": metrics,
        "extra": extra or {},
    }
    if y_prob_sample is not None:
        record["y_prob_sample"] = np.asarray(y_prob_sample)[:2000].round(5).tolist()
        record["y_prob_sample_note"] = "primi 2000 valori, per PR-curve/istogrammi senza rilanciare il training"

    json_path = os.path.join(RESULTS_DIR, "logs", f"{experiment_name}.json")
    with open(json_path, "w") as f:
        json.dump(_sanitize(record), f, indent=2)

    row = {
        "experiment_name": experiment_name, "dataset": dataset,
        "target_family": target_family, "seed": seed,
        **{f"hp_{k}": v for k, v in hyperparams.items()},
        **{k: v for k, v in metrics.items() if not isinstance(v, dict)},
    }
    write_header = not os.path.exists(LOG_CSV)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(LOG_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"[log_experiment] Salvato: {json_path}")
    print(f"[log_experiment] Riga aggiunta a: {LOG_CSV}")
    return json_path


# ==========================================================================
# 5. TEST-SET LOBO FISSI (costruiti una volta, riusati identici ovunque)
# ==========================================================================
def build_and_save_lobo_testset(df_clean: pd.DataFrame, target_family: str,
                                 n_benign_test: int, seed: int = 42) -> str:


    from sklearn.model_selection import train_test_split

    df_benign = df_clean[df_clean["family"] == "benign"]
    df_target = df_clean[df_clean["family"] == target_family]

    _, benign_test_pool = train_test_split(df_benign, test_size=0.2, random_state=seed)
    n = min(n_benign_test, len(benign_test_pool))
    benign_test = benign_test_pool.sample(n=n, random_state=seed)

    df_test = pd.concat([benign_test, df_target], ignore_index=True)

    os.makedirs(os.path.join(RESULTS_DIR, "fixed_testsets"), exist_ok=True)
    path = os.path.join(RESULTS_DIR, "fixed_testsets", f"lobo_test_{target_family}.parquet")
    df_test.to_parquet(path)
    print(f"[build_and_save_lobo_testset] {target_family}: {len(df_test)} righe "
          f"({n} benigne + {len(df_target)} target) salvate in {path}")
    return path


def load_lobo_testset(target_family: str) -> pd.DataFrame:
    path = os.path.join(RESULTS_DIR, "fixed_testsets", f"lobo_test_{target_family}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Test-set fisso per '{target_family}' non trovato. "
            f"Esegui prima 04_build_lobo_testsets.py."
        )
    return pd.read_parquet(path)


# ==========================================================================
# 6. GRAFICI AUTOMATICI PER OGNI ESPERIMENTO
# ==========================================================================
def save_experiment_plots(y_true, y_prob, threshold: float, experiment_name: str):
    
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay

    plots_dir = os.path.join(RESULTS_DIR, "plots", experiment_name)
    os.makedirs(plots_dir, exist_ok=True)

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # --- 1. Curva Precision-Recall ---
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_prob)
    idx = np.argmin(np.abs(thresholds_pr - threshold)) if len(thresholds_pr) > 0 else 0
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color="steelblue", lw=2, label="Curva PR")
    ax.scatter(recall[idx], precision[idx], color="red", zorder=5,
               label=f"Soglia={threshold:.3f}\nP={precision[idx]:.3f}, R={recall[idx]:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Curva Precision-Recall\n{experiment_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "precision_recall_curve.png"), dpi=150)
    plt.close()

    # --- 2. Distribuzione probabilita' ---
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(y_prob[y_true == 0], bins=50, alpha=0.6, color="steelblue", label="Benigno")
    ax.hist(y_prob[y_true == 1], bins=50, alpha=0.6, color="salmon", label="Botnet")
    ax.axvline(threshold, color="red", lw=2, linestyle="--", label=f"Soglia={threshold:.3f}")
    ax.set_xlabel("Probabilita' predetta (classe botnet)")
    ax.set_ylabel("Conteggio")
    ax.set_title(f"Distribuzione Probabilita'\n{experiment_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "prob_distribution.png"), dpi=150)
    plt.close()

    # --- 3. Matrice di confusione ---
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benigno", "Botnet"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Matrice di Confusione\n{experiment_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    print(f"[save_experiment_plots] 3 grafici salvati in: {plots_dir}")