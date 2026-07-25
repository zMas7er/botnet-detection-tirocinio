import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import load_cic, clean_numeric, split_scale_balance
from src.train_baseline import train_and_evaluate, save_results

print("--- FASE 1: ADDESTRAMENTO IN-DISTRIBUTION 2017 ---")

base_dir = os.path.dirname(os.path.abspath(__file__))

print("1. Caricamento dataset 2017...")
cic17_path = os.path.join(base_dir, "data/raw/cic_ids2017")
df = load_cic(cic17_path, "CIC-IDS2017")

print("2. Pulizia numerica...")
df_clean = clean_numeric(df)

print("3. Suddivisione, Standardizzazione e Bilanciamento SMOTE...")
X_train, X_test, y_train, y_test, scaler, feats = split_scale_balance(df_clean)
print(f"Dimensione Train: {X_train.shape} | Dimensione Test: {X_test.shape}")

print("4. Addestramento dei modelli...")
results, fitted_models = train_and_evaluate(X_train, X_test, y_train, y_test, verbose=True)

print("5. Salvataggio metriche per i grafici futuri...")
os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)
save_results(results, os.path.join(base_dir, "results", "baseline_2017.json"))

print("--- FASE 1 COMPLETATA CON SUCCESSO ---")