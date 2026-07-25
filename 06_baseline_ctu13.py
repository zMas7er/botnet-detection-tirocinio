import os
import sys

# Aggiungiamo 'src' ai percorsi di Python
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import load_ctu13, clean_numeric, split_scale_balance
from src.train_baseline import train_and_evaluate, save_results

print("--- TEST PRELIMINARE: ADDESTRAMENTO IN-DISTRIBUTION CTU-13 ---")

# 1. Caricamento dati
print("1. Caricamento dell'unico scenario CTU-13 presente...")
ctu_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/raw/ctu13")
df = load_ctu13(ctu_path)

# 2. Pulizia (rimozione NaN/Inf, drop colonne costanti)
print("2. Pulizia numerica e codifica delle variabili categoriche del CTU...")
df_clean = clean_numeric(df)

# 3. Preparazione
print("3. Suddivisione, Standardizzazione e Bilanciamento con SMOTE...")
X_train, X_test, y_train, y_test, scaler, feats = split_scale_balance(
    df_clean, test_size=0.2, use_smote=True, smote_strategy=0.3
)

print(f"Dimensione Train (Bilanciato): {X_train.shape} | Dimensione Test (Reale): {X_test.shape}")

# 4. Addestramento e Valutazione
print("4. Addestramento dei modelli sul traffico CTU-13...")
results, fitted_models = train_and_evaluate(X_train, X_test, y_train, y_test, verbose=True)

# 5. Salvataggio risultati
print("5. Salvataggio metriche...")
os.makedirs("results", exist_ok=True)
save_results(results, "results/baseline_ctu13.json")

print("\n--- TEST PRELIMINARI SUI 3 DATASET COMPLETATI ---")