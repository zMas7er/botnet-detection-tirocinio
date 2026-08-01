import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from src.preprocessing import load_ctu13, clean_numeric, get_feature_columns
from src.experiment_utils import select_threshold, evaluate_full, log_experiment, save_experiment_plots

SEED = 42
DATASET = "CTU-13"
base_dir = os.path.dirname(os.path.abspath(__file__))

print(f"--- BASELINE: {DATASET} ---")
df = clean_numeric(load_ctu13(os.path.join(base_dir, "data/raw/ctu13"), fraction_benign=0.1))
feature_cols = get_feature_columns(df)
X = df[feature_cols].values
y = df["label"].values
print(f"Righe totali dataset: {X.shape[0]} | Feature: {X.shape[1]} | Botnet: {y.mean():.2%}")

grids = {
    "RandomForest": (RandomForestClassifier(random_state=SEED, n_jobs=-1),
                     {"model__n_estimators": [100, 200], "model__max_depth": [None, 10]}),
    "XGBoost": (XGBClassifier(eval_metric="logloss", n_jobs=-1, random_state=SEED),
                {"model__n_estimators": [100, 200], "model__max_depth": [3, 6], "model__learning_rate": [0.1]}),
    "SVM": (SVC(probability=True, random_state=SEED),
            {"model__C": [1.0, 10.0], "model__kernel": ["rbf"]}),
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

for name, (base_model, params) in grids.items():
    print(f"\n==========================================")
    print(f"--- MODELLO: {name} ---")
    print(f"==========================================")
    
    # Hard-cap a 50.000 campioni per la SVM per evitare limiti di scalabilità O(n^2)
    if name == "SVM" and X.shape[0] > 50000:
        print("[SVM] Applicazione hard-cap a 50.000 campioni stratificati...")
        X_sub, _, y_sub, _ = train_test_split(X, y, train_size=50000, stratify=y, random_state=SEED)
    else:
        X_sub, y_sub = X, y

    X_train, X_temp, y_train, y_temp = train_test_split(X_sub, y_sub, test_size=0.4, stratify=y_sub, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED)
    
    pos_rate = y_train.mean()
    minority_ratio = pos_rate / (1 - pos_rate)
    use_smote = minority_ratio < 0.3
    print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]} | SMOTE: {use_smote}")

    def build_pipeline(model):
        steps = [("scaler", StandardScaler())]
        if use_smote:
            steps.append(("smote", SMOTE(sampling_strategy=0.3, random_state=SEED)))
        steps.append(("model", model))
        return ImbPipeline(steps)

    grid = GridSearchCV(build_pipeline(base_model), params, cv=cv, scoring="f1", n_jobs=-1)
    grid.fit(X_train, y_train)
    best_pipeline = grid.best_estimator_
    print(f"Migliori parametri: {grid.best_params_}")

    y_prob_val = best_pipeline.predict_proba(X_val)[:, 1]
    threshold, threshold_report = select_threshold(y_val, y_prob_val, max_fpr=0.001)
    print(f"Soglia scelta su validation: {threshold:.4f}")

    y_prob_test = best_pipeline.predict_proba(X_test)[:, 1]
    metrics = evaluate_full(y_test, y_prob_test, threshold)
    print(f"Test -> F1={metrics['f1_botnet']} Recall={metrics['recall_botnet']} "
          f"AUROC={metrics['auroc']} FPR={metrics['false_positive_rate']}")

    log_experiment(
        experiment_name=f"baseline_{DATASET}_{name}",
        dataset=DATASET,
        target_family=None,
        seed=SEED,
        split_info={"train": len(X_train), "val": len(X_val), "test": len(X_test), "method": "60/20/20"},
        hyperparams={**grid.best_params_, "threshold_objective": threshold_report["objective"]},
        metrics=metrics,
        y_prob_sample=y_prob_test,
        extra={"threshold_report": threshold_report, "use_smote": use_smote},
    )

    save_experiment_plots(
        y_true=y_test, 
        y_prob=y_prob_test, 
        threshold=threshold, 
        experiment_name=f"baseline_{DATASET}_{name}"
    )

print("\n--- BASELINE CTU-13 COMPLETATA ---")