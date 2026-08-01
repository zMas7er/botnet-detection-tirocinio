import os, sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.preprocessing import load_cic, clean_numeric, get_feature_columns, align_cic_features

base_dir = os.path.dirname(os.path.abspath(__file__))

def audit_datasets():
    print("--- AUDITING E ALLINEAMENTO FEATURE CROSS-DATASET ---")
    
    # 1. Caricamento e pulizia CIC-IDS2017
    print("\nCaricamento e pulizia CIC-IDS2017...")
    path_2017 = os.path.join(base_dir, "data", "raw", "cic_ids2017")
    df_17 = load_cic(path_2017, "CIC-IDS2017")
    bugged_cols = [c for c in df_17.columns if c.endswith('.1') or c == 'FwdHeader Length']
    if bugged_cols:
        df_17 = df_17.drop(columns=bugged_cols)
    df_17_clean = clean_numeric(df_17)
    feats_17 = set(get_feature_columns(df_17_clean))

    # 2. Caricamento e pulizia CSE-CIC-IDS2018
    print("Caricamento e pulizia CSE-CIC-IDS2018...")
    path_2018 = os.path.join(base_dir, "data", "raw", "cse_cic_ids2018")
    df_18 = load_cic(path_2018, "CSE-CIC-IDS2018")
    df_18_aligned = align_cic_features(df_18)
    df_18_clean = clean_numeric(df_18_aligned)
    feats_18 = set(get_feature_columns(df_18_clean))

    # 3. Intersezione e Differenze
    common_features = sorted(list(feats_17.intersection(feats_18)))
    only_17 = sorted(list(feats_17 - feats_18))
    only_18 = sorted(list(feats_18 - feats_17))

    print(f"\n[REPORT STRUTTURALE]")
    print(f"  - Feature totali in 2017: {len(feats_17)}")
    print(f"  - Feature totali in 2018: {len(feats_18)}")
    print(f"  - Feature comuni (intersezione): {len(common_features)}")
    print(f"  - Feature presenti solo in 2017: {len(only_17)}")
    print(f"  - Feature presenti solo in 2018: {len(only_18)}")

    if only_17:
        print(f"    Esclusive 2017: {only_17}")
    if only_18:
        print(f"    Esclusive 2018: {only_18}")

    # 4. Controllo Distribuzionale sulle feature comuni (campione descrittivo)
    print("\n[REPORT DISTRIBUZIONALE (Controllo Unità e Scale)]")
    print(f"{'Feature':<35} | {'Mean (2017)':<12} | {'Mean (2018)':<12} | {'Max (2017)':<12} | {'Max (2018)':<12}")
    print("-" * 95)
    
    # Prendiamo un sottoinsieme di feature chiave (es. volumetriche e temporali) da monitorare
    sample_feats = [f for f in common_features if any(k in f.lower() for k in ['duration', 'pkt', 'byt', 'flow'])]
    if not sample_feats:
        sample_feats = common_features[:10]

    audit_records = []
    for f in sample_feats:
        m17 = df_17_clean[f].mean()
        m18 = df_18_clean[f].mean()
        max17 = df_17_clean[f].max()
        max18 = df_18_clean[f].max()
        
        print(f"{f:<35} | {m17:<12.2f} | {m18:<12.2f} | {max17:<12.2f} | {max18:<12.2f}")
        
        audit_records.append({
            "feature": f,
            "mean_2017": float(m17), "mean_2018": float(m18),
            "max_2017": float(max17), "max_2018": float(max18)
        })

    # Salvataggio del report di audit in JSON per la tesi
    out_path = os.path.join(base_dir, "results", "logs", "feature_audit_report.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    import json
    with open(out_path, "w") as f_out:
        json.dump({
            "common_features_count": len(common_features),
            "common_features": common_features,
            "exclusive_to_2017": only_17,
            "exclusive_to_2018": only_18,
            "distribution_sample": audit_records
        }, f_out, indent=4)
    print(f"\n✅ Report di auditing salvato con successo in: {out_path}")

if __name__ == "__main__":
    audit_datasets()