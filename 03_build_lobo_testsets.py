
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import load_ctu13, clean_numeric
from src.experiment_utils import build_and_save_lobo_testset

base_dir = os.path.dirname(os.path.abspath(__file__))

def main():
    print("--- 03: COSTRUZIONE TEST SET LOBO (CONGELATI) ---")
    
    # Carichiamo il dataset (usiamo fraction_benign=0.1 per coerenza con la baseline)
    raw_dir = os.path.join(base_dir, "data", "raw", "ctu13")
    df = clean_numeric(load_ctu13(raw_dir, fraction_benign=0.1))

    print("\n--- DEBUG FAMIGLIE TROVATE ---")
    print(df["family"].value_counts())
    
    # Fissiamo a 50.000 il numero di flussi benigni per avere un test set equilibrato ma sfidante
    N_BENIGN_TEST = 50000
    
    print("\n[1] Costruzione Test Set per NERIS...")
    build_and_save_lobo_testset(df, target_family="Neris", n_benign_test=N_BENIGN_TEST, seed=42)

    print("\n[2] Costruzione Test Set per RBOT...")
    build_and_save_lobo_testset(df, target_family="Rbot", n_benign_test=N_BENIGN_TEST, seed=42)
    
    print("\n[3] Costruzione Test Set per VIRUT...")
    build_and_save_lobo_testset(df, target_family="Virut", n_benign_test=N_BENIGN_TEST, seed=42)

    print("\n[4] Costruzione Test Set per MENTI...")
    build_and_save_lobo_testset(df, target_family="Menti", n_benign_test=N_BENIGN_TEST, seed=42)
    
    print("\n--- TEST SET CONGELATI CON SUCCESSO IN results/fixed_testsets/ ---")

if __name__ == "__main__":
    main()