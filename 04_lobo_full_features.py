"""
04_lobo_full_features.py
Esegue l'esperimento Zero-Day (LOBO) con tutte le feature.
Valutazione statistica su 5 iterazioni.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from src.preprocessing import (
    load_ctu13,
    clean_numeric,
    get_feature_columns,
)

from src.experiment_utils import (
    select_threshold,
    evaluate_full,
    log_experiment,
    assert_family_excluded,
    load_lobo_testset,
    save_experiment_plots,
)

SEED = 42
ITERATIONS = 5
base_dir = os.path.dirname(os.path.abspath(__file__))


def run_lobo_experiment():

    print("--- 04: ESPERIMENTO LOBO (ZERO-DAY) - FULL FEATURES ---")

    raw_dir = os.path.join(base_dir, "data", "raw", "ctu13")

    df_clean = clean_numeric(
        load_ctu13(raw_dir, fraction_benign=0.1)
    )

    feature_cols = get_feature_columns(df_clean)

    target_families = [
        "Neris",
        "Rbot",
        "Virut",
        "Menti",
    ]

    for target in target_families:

        print("\n" + "=" * 60)
        print(f"TARGET ZERO-DAY: {target}")
        print("=" * 60)

        df_train_val = df_clean[df_clean["family"] != target].copy()

        # Guardia anti leakage
        assert_family_excluded(df_train_val, target)

        # Test-set congelato
        df_test = load_lobo_testset(target)

        X_test = df_test[feature_cols].values
        y_test = df_test["label"].values

        accumulated_metrics = {
            "f1_botnet": [],
            "precision_botnet": [],
            "recall_botnet": [],
            "auroc": [],
            "pr_auc": [],
            "false_positive_rate": [],
        }

        for i in range(ITERATIONS):

            current_seed = SEED + i

            X_train_val = df_train_val[feature_cols].values
            y_train_val = df_train_val["label"].values

            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val,
                y_train_val,
                test_size=0.2,
                stratify=y_train_val,
                random_state=current_seed,
            )

            scaler = StandardScaler()

            X_train_sc = scaler.fit_transform(X_train)
            X_val_sc = scaler.transform(X_val)
            X_test_sc = scaler.transform(X_test)

            smote = SMOTE(
                sampling_strategy=0.3,
                random_state=current_seed,
            )

            X_train_res, y_train_res = smote.fit_resample(
                X_train_sc,
                y_train,
            )

            hyperparams = {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
            }

            xgb = XGBClassifier(
                **hyperparams,
                eval_metric="logloss",
                n_jobs=-1,
                random_state=current_seed,
            )

            xgb.fit(X_train_res, y_train_res)

            # Validation
            y_prob_val = xgb.predict_proba(X_val_sc)[:, 1]

            threshold, thresh_report = select_threshold(
                y_val,
                y_prob_val,
                max_fpr=0.001,
            )

            # Test
            y_prob_test = xgb.predict_proba(X_test_sc)[:, 1]

            metrics = evaluate_full(
                y_test,
                y_prob_test,
                threshold,
            )

            for key in accumulated_metrics:
                accumulated_metrics[key].append(metrics[key])

            print(
                f"Run {i+1}/{ITERATIONS} -> "
                f"F1: {metrics['f1_botnet']:.4f} | "
                f"Recall: {metrics['recall_botnet']:.4f} | "
                f"AUROC: {metrics['auroc']:.4f} | "
                f"Soglia: {threshold:.4f}"
            )

            # Salva solo ultima run
            if i == ITERATIONS - 1:

                log_experiment(
                    experiment_name=f"lobo_full_features_{target}",
                    dataset="CTU-13",
                    target_family=target,
                    seed=current_seed,
                    split_info={
                        "train_size": len(X_train),
                        "val_size": len(X_val),
                        "test_size": len(X_test),
                    },
                    hyperparams={
                        **hyperparams,
                        "threshold_used": threshold,
                    },
                    metrics=metrics,
                    y_prob_sample=y_prob_test,
                    extra={
                        "threshold_report": thresh_report,
                    },
                )

                save_experiment_plots(
                    y_true=y_test,
                    y_prob=y_prob_test,
                    threshold=threshold,
                    experiment_name=f"lobo_full_features_{target}",
                )

        print("\n=== RISULTATI STATISTICI ===")

        for key, values in accumulated_metrics.items():
            print(
                f"{key}: "
                f"{np.mean(values):.4f} ± {np.std(values):.4f}"
            )


if __name__ == "__main__":
    run_lobo_experiment()