import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import load_cic, clean_numeric, split_scale_balance
from src.train_baseline import train_and_evaluate, save_results

print("--- FASE 1: ADDESTRAMENTO IN-DISTRIBUTION 2018 ---")

# 1. Caricamento dati
print("1. Caricamento dataset 2018...")
cic18_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/raw/cse_cic_ids2018")
df = load_cic(cic18_path, "CSE-CIC-IDS2018")

# 2. Pulizia (rimozione NaN/Inf, drop colonne costanti)
print("2. Pulizia numerica...")
df_clean = clean_numeric(df)

# 3. Preparazione (Split, Scalatura e Bilanciamento con SMOTE)
print("3. Suddivisione, Standardizzazione e Bilanciamento SMOTE...")
X_train, X_test, y_train, y_test, scaler, feats = split_scale_balance(df_clean, test_size=0.3, use_smote=False)
print(f"Dimensione Train: {X_train.shape} | Dimensione Test: {X_test.shape}")

# 4. Addestramento e Valutazione
print("4. Addestramento dei modelli...")
results, fitted_models = train_and_evaluate(X_train, X_test, y_train, y_test, verbose=True)

# 5. Salvataggio risultati
print("5. Salvataggio metriche per i grafici futuri...")
os.makedirs("results", exist_ok=True)
save_results(results, "results/baseline_2018.json")

print("\n--- FASE 1 COMPLETATA CON SUCCESSO ---")