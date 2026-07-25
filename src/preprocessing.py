from __future__ import annotations
import glob
import os
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

CTU13_LEAKY_COLS = [
    "StartTime", "SrcAddr", "DstAddr", "Sport", "Dport",
]

CIC_LEAKY_COLS = [
    "Flow ID", "Src IP", "Source IP", "Dst IP", "Destination IP",
    "Src Port", "Source Port", "Dst Port", "Destination Port",
    "Timestamp",
]


def _ctu13_label_row(label: str) -> tuple[int, str]:
    label = str(label)
    if "Botnet" in label or "botnet" in label:
        return 1, "botnet"
    return 0, "benign"


CTU13_SCENARIO_FAMILY = {
    "42": "Neris", "43": "Neris", "44": "Rbot", "45": "Rbot",
    "46": "Virut", "47": "Menti", "48": "Sogou", "49": "Murlo",
    "50": "Neris", "51": "Rbot", "52": "Rbot", "53": "NSIS.ay",
    "54": "Virut",
}


def load_ctu13(raw_dir: str) -> pd.DataFrame:
    patterns = ["*.binetflow", "*.biargus"]
    files = []
    for p in patterns:
        files += glob.glob(os.path.join(raw_dir, "**", p), recursive=True)
    files = sorted(set(files))

    if not files:
        raise FileNotFoundError(
            f"Nessun file di flusso (.binetflow/.biargus) trovato in {raw_dir}. "
            "Scarica CTU-13 da https://www.stratosphereips.org/datasets-ctu13 "
            "e posizionalo qui prima di procedere. Controlla il nome esatto "
            "del file scaricato: se non e' ne' .binetflow ne' .biargus, "
            "aggiungi il pattern giusto qui sopra."
        )

    frames = []
    for f in files:
        df = pd.read_csv(f)
        m = re.search(r"Botnet-(\d+)", f)
        scenario = m.group(1) if m else "unknown"
        family = CTU13_SCENARIO_FAMILY.get(scenario, f"scenario_{scenario}")

        label_col = "Label" if "Label" in df.columns else df.columns[-1]
        labels = df[label_col].apply(_ctu13_label_row)
        df["label"] = [l[0] for l in labels]
        df["family"] = np.where(df["label"] == 1, family, "benign")
        df["source_dataset"] = "CTU-13"
        df["scenario"] = scenario
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)
    full = full.drop(columns=[c for c in CTU13_LEAKY_COLS if c in full.columns])
    if "Label" in full.columns:
        full = full.drop(columns=["Label"])
    return full


def load_cic(raw_dir: str, source_name: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(raw_dir, "**", "*.csv"), recursive=True)
    if not files:
        raise FileNotFoundError(
            f"Nessun file .csv trovato in {raw_dir}. "
            "Scarica il dataset da https://www.unb.ca/cic/datasets/ e "
            "posizionalo qui prima di procedere."
        )

    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)

    label_col = "Label" if "Label" in full.columns else full.columns[-1]
    full["label"] = (~full[label_col].astype(str).str.upper().eq("BENIGN")).astype(int)
    full["family"] = np.where(full["label"] == 1, full[label_col].astype(str), "benign")
    full["source_dataset"] = source_name

    full = full.drop(columns=[c for c in CIC_LEAKY_COLS if c in full.columns])
    full = full.drop(columns=[label_col]) if label_col in full.columns else full
    return full

def clean_numeric(
    df: pd.DataFrame,
    keep_cols=("label", "family", "source_dataset", "scenario"),
    max_categorical_cardinality: int = 20,
) -> pd.DataFrame:
    keep = [c for c in keep_cols if c in df.columns]
    feature_cols = [c for c in df.columns if c not in keep]
    df = df.copy()

    numeric_cols, categorical_cols = [], []
    for c in feature_cols:
        converted = pd.to_numeric(df[c], errors="coerce")
        # se la conversione numerica fallisce per la maggior parte dei
        # valori non-null, trattiamo la colonna come categorica
        non_null = df[c].notna().sum()
        still_valid = converted.notna().sum()
        if non_null == 0 or still_valid / max(non_null, 1) < 0.5:
            categorical_cols.append(c)
        else:
            df[c] = converted
            numeric_cols.append(c)

    if categorical_cols:
        true_identifiers = [c for c in categorical_cols if df[c].nunique() > 0.5 * len(df)]
        informative_cat = [c for c in categorical_cols if c not in true_identifiers]

        if true_identifiers:
            print(f"[clean_numeric] Droppate {len(true_identifiers)} colonne "
                  f"probabili identificatori (quasi un valore per riga): {true_identifiers}")
            df = df.drop(columns=true_identifiers)

        if informative_cat:
            for c in informative_cat:
                if df[c].nunique() > max_categorical_cardinality:
                    top = df[c].value_counts().nlargest(max_categorical_cardinality - 1).index
                    df[c] = df[c].where(df[c].isin(top), other="OTHER")
                    print(f"[clean_numeric] '{c}': categorie rare raggruppate in 'OTHER' "
                          f"(mantenute le {max_categorical_cardinality - 1} piu' frequenti)")
            print(f"[clean_numeric] One-hot encoding per colonne categoriche: {informative_cat}")
            df = pd.get_dummies(df, columns=informative_cat, prefix=informative_cat)
            numeric_cols += [c for c in df.columns
                             if any(c.startswith(p + "_") for p in informative_cat)]

    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    n_before = len(df)
    df = df.dropna(subset=numeric_cols)
    n_after = len(df)
    dropped = n_before - n_after
    if dropped:
        print(f"[clean_numeric] Rimosse {dropped} righe con NaN/inf "
              f"({dropped / n_before:.2%} del totale).")

    nunique = df[numeric_cols].nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        print(f"[clean_numeric] Rimosse {len(constant_cols)} colonne costanti: {constant_cols}")
        df = df.drop(columns=constant_cols)

    return df


def get_feature_columns(df: pd.DataFrame, keep_cols=("label", "family", "source_dataset", "scenario")) -> list[str]:
    return [c for c in df.columns if c not in keep_cols]

def align_cic_features(df_2018: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        'Protocol': 'Protocol',
        'Tot Fwd Pkts': 'Total Fwd Packets',
        'Tot Bwd Pkts': 'Total Backward Packets',
        'TotLen Fwd Pkts': 'Total Length of Fwd Packets',
        'TotLen Bwd Pkts': 'Total Length of Bwd Packets',
        'Fwd Pkt Len Max': 'Fwd Packet Length Max',
        'Fwd Pkt Len Min': 'Fwd Packet Length Min',
        'Fwd Pkt Len Mean': 'Fwd Packet Length Mean',
        'Fwd Pkt Len Std': 'Fwd Packet Length Std',
        'Bwd Pkt Len Max': 'Bwd Packet Length Max',
        'Bwd Pkt Len Min': 'Bwd Packet Length Min',
        'Bwd Pkt Len Mean': 'Bwd Packet Length Mean',
        'Bwd Pkt Len Std': 'Bwd Packet Length Std',
        'Flow Byts/s': 'Flow Bytes/s',
        'Flow Pkts/s': 'Flow Packets/s',
        'Flow IAT Mean': 'Flow IAT Mean',
        'Flow IAT Std': 'Flow IAT Std',
        'Flow IAT Max': 'Flow IAT Max',
        'Flow IAT Min': 'Flow IAT Min',
        'Fwd IAT Tot': 'Fwd IAT Total',
        'Fwd IAT Mean': 'Fwd IAT Mean',
        'Fwd IAT Std': 'Fwd IAT Std',
        'Fwd IAT Max': 'Fwd IAT Max',
        'Fwd IAT Min': 'Fwd IAT Min',
        'Bwd IAT Tot': 'Bwd IAT Total',
        'Bwd IAT Mean': 'Bwd IAT Mean',
        'Bwd IAT Std': 'Bwd IAT Std',
        'Bwd IAT Max': 'Bwd IAT Max',
        'Bwd IAT Min': 'Bwd IAT Min',
        'Fwd PSH Flags': 'Fwd PSH Flags',
        'Fwd Header Len': 'Fwd Header Length',
        'Bwd Header Len': 'Bwd Header Length',
        'Fwd Pkts/s': 'Fwd Packets/s',
        'Bwd Pkts/s': 'Bwd Packets/s',
        'Pkt Len Min': 'Min Packet Length',
        'Pkt Len Max': 'Max Packet Length',
        'Pkt Len Mean': 'Packet Length Mean',
        'Pkt Len Std': 'Packet Length Std',
        'Pkt Len Var': 'Packet Length Variance',
        'FIN Flag Cnt': 'FIN Flag Count',
        'SYN Flag Cnt': 'SYN Flag Count',
        'RST Flag Cnt': 'RST Flag Count',
        'PSH Flag Cnt': 'PSH Flag Count',
        'ACK Flag Cnt': 'ACK Flag Count',
        'URG Flag Cnt': 'URG Flag Count',
        'ECE Flag Cnt': 'ECE Flag Count',
        'Down/Up Ratio': 'Down/Up Ratio',
        'Pkt Size Avg': 'Average Packet Size',
        'Fwd Seg Size Avg': 'Avg Fwd Segment Size',
        'Bwd Seg Size Avg': 'Avg Bwd Segment Size',
        'Subflow Fwd Pkts': 'Subflow Fwd Packets',
        'Subflow Fwd Byts': 'Subflow Fwd Bytes',
        'Subflow Bwd Pkts': 'Subflow Bwd Packets',
        'Subflow Bwd Byts': 'Subflow Bwd Bytes',
        'Init Fwd Win Byts': 'Init_Win_bytes_forward',
        'Init Bwd Win Byts': 'Init_Win_bytes_backward',
        'Fwd Act Data Pkts': 'act_data_pkt_fwd',
        'Fwd Seg Size Min': 'min_seg_size_forward',
        'Active Mean': 'Active Mean',
        'Active Std': 'Active Std',
        'Active Max': 'Active Max',
        'Active Min': 'Active Min',
        'Idle Mean': 'Idle Mean',
        'Idle Std': 'Idle Std',
        'Idle Max': 'Idle Max',
        'Idle Min': 'Idle Min'
    }
    
    df = df_2018.copy()
    df = df.rename(columns=mapping)
    return df

def split_scale_balance(
    df: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = 42,
    use_smote: bool = True,
    smote_strategy: float = 0.3,
):

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if use_smote:
        from imblearn.over_sampling import SMOTE
        pos_rate = y_train.mean()
        if 0 < pos_rate < smote_strategy:
            sm = SMOTE(sampling_strategy=smote_strategy, random_state=random_state)
            X_train, y_train = sm.fit_resample(X_train, y_train)
        else:
            print(f"[split_scale_balance] Skip SMOTE: pos_rate train = {pos_rate:.4f} "
                  f"gia' >= target {smote_strategy}")

    return X_train, X_test, y_train, y_test, scaler, feature_cols
