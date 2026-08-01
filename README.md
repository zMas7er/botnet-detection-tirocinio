# Rilevamento di Botnet con Machine Learning — Progetto di Tirocinio

Pipeline sperimentale per la valutazione della robustezza e generalizzazione di modelli di Machine Learning (Random Forest, XGBoost) nel rilevamento di botnet, integrando analisi di interpretabilità (SHAP) e strategie di mitigazione per scenari *Out-of-Distribution* e *Zero-Day*.

## Struttura del progetto

botnet_project/
├── data/raw/
│   ├── ctu13/                     # scenari CTU-13 (.binetflow)
│   ├── cic_ids2017/               # CIC-IDS2017 CSV dataset
│   └── cse_cic_ids2018/           # CSE-CIC-IDS2018 CSV dataset
├── src/
│   └── preprocessing.py           # caricamento, pulizia, gestione categoriche e NaN/inf
├── 01_baseline_2017.py            # Baseline in-distribution su CIC-IDS2017
├── 01_baseline_2018.py            # Baseline in-distribution su CSE-CIC-IDS2018
├── 01_baseline_ctu13.py           # Baseline in-distribution su CTU-13
├── 02_cross_dataset.py            # Test cross-dataset (CIC-IDS2017 → CSE-CIC-IDS2018)
├── 03_build_lobo_testsets.py      # Congelamento e generazione test set LOBO (.parquet)
├── 04_lobo_full_features.py       # Test Zero-Day LOBO (Leave-One-Botnet-Out)
├── 05_shap_analysis.py            # Analisi di interpretabilità SHAP (in-distribution vs zero-day)
├── 06_mitigation_pruning.py       # Mitigazione 1: Feature Pruning su Rbot
├── 06_mitigation_pruning_Virut.py # Mitigazione 1: Feature Pruning su Virut (famiglia indipendente)
├── 06_mitigation_robust.py        # Mitigazione 2: Solo Feature Robuste su Rbot
├── 06_mitigation_robust_Virut.py  # Mitigazione 2: Solo Feature Robuste su Virut
├── 07_tuning_2017.py / 2018 / ctu # Grid Search e ottimizzazione iperparametri (3-fold CV)
├── 08_feature_audit.py            # Audit distribuzionale feature cross-dataset
├── 12_svm.py                      # Valutazione comparativa Support Vector Machine (kernel RBF)
├── requirements.txt
└── results/                       # Log metriche (.json), log globali (.csv) e grafici (.png)

## Come riprodurre gli esperimenti

pip install -r requirements.txt

Popolare data/raw/ con i dataset di riferimento, poi eseguire gli script nell'ordine:

1. python 01_baseline_2017.py / 01_baseline_2018.py / 01_baseline_ctu13.py — baseline in-distribution
2. python 07_tuning_2017.py (e successivi) — ottimizzazione iperparametri con Grid Search
3. python 02_cross_dataset.py — test di generalizzazione cross-dataset (2017 → 2018)
4. python 03_build_lobo_testsets.py — preparazione dei test set di riferimento LOBO
5. python 04_lobo_full_features.py — esecuzione test Zero-Day LOBO (Neris, Rbot, Virut, Menti)
6. python 05_shap_analysis.py — estrazione dei summary plot SHAP globali e locali
7. python 06_mitigation_pruning.py & python 06_mitigation_robust.py (inclusi i test su Virut) — esecuzione delle mitigazioni

Ogni script genera automaticamente i log strutturati in results/logs/*.json, le righe di riepilogo in results/experiment_log.csv e i plot in results/plots/.

## Sintesi metodologica

- Modelli: Random Forest e XGBoost (ensemble tree-based), selezionati per efficacia su dati tabulari di rete, velocità e compatibilità nativa con explainers basati su alberi (TreeExplainer).
- Metriche di Valutazione: F1-Score, Precision, Recall, PR-AUC, False Positive Rate (FPR) e AUROC. L'accuracy è esclusa per evitare l'Accuracy Paradox su dataset fortemente sbilanciati.
- Prevenzione Data Leakage: Rimozione sistematica di ID, porte e timestamp assoluti. Fitting di StandardScaler e campionamento SMOTE rigorosamente confinati al solo training set di ciascuna iterazione. Test set LOBO congelati e disgiunti a livello di riga.
- Validazione Statistica Rigorosa: Tutti gli esperimenti finali e le mitigazioni sono stati condotti su 5 iterazioni indipendenti con seed casuali differenti, riportando media e deviazione standard (± std) per garantire assoluta stabilità e replicabilità.

## Sintesi dei risultati principali

| Esperimento | Risultato chiave (Media ± std) |
|---|---|
| Baseline In-Distribution | F1 ≥ 0.95 su CIC-IDS; F1 = 0.6299 su CTU-13 (spazio feature ridotto a log NetFlow grezzi). |
| Cross-Dataset (2017 → 2018) | Crollo dell'F1 da 0.9619 a 0.0002; AUROC stabile a 0.9466 (dimostrazione dello shift della soglia statica). |
| LOBO (Zero-Day - Rbot / Virut) | AUROC alto (0.9759 e 0.9300), ma Recall operativa azzerata (0.04% e 1.31%) a causa del covariate shift volumetrico. |
| Analisi SHAP | Le feature volumetriche (Dur, SrcBytes, TotBytes) dominano le decisioni in ogni scenario: il problema non è la loro importanza, ma lo slittamento dei valori assoluti tra famiglie diverse. |
| Mitigazione 1 (Feature Pruning) | Disastrosa: AUROC crolla sotto 0.5 (0.0903) e Recall a 0.12%; conferma che le feature volumetriche non sono rumore ma il segnale centrale. |
| Mitigazione 2 (Solo Robuste) | Ripristino perfetto dell'AUROC (0.9737 su Rbot, 0.9199 su Virut), confermando l'esattezza dell'analisi SHAP, pur evidenziando il limite intrinseco della soglia statica. Validato su famiglia indipendente (Virut). |

## Dataset e citazioni

- CTU-13: Garcia, S., Grill, M., Stiborek, J., Zunino, A. "An empirical comparison of botnet detection methods." Computers & Security, vol. 45, pp. 100-123, 2014.
- CIC-IDS2017 / CSE-CIC-IDS2018: Sharafaldin, I., Lashkari, A.H., Ghorbani, A.A. "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization." ICISSP, 2018.

Nota: I dataset non sono inclusi nel repository per ragioni di dimensioni e licenza.