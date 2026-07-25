# Rilevamento di Botnet con Machine Learning — Progetto di Tirocinio

Pipeline sperimentale per la valutazione della robustezza e generalizzazione
di modelli ML (Random Forest, XGBoost) nel rilevamento di botnet, con analisi
di interpretabilità (SHAP) e mitigazione.

## Struttura del progetto

```
botnet_project/
├── data/raw/
│   ├── ctu13/                     # scenari CTU-13 (.binetflow)
│   ├── cic_ids2017/               # Friday-WorkingHours-Morning.pcap_ISCX.csv
│   └── cse_cic_ids2018/           # Friday-02-03-2018_TrafficForML_CICFlowMeter.csv
├── src/
│   ├── preprocessing.py           # caricamento, pulizia, feature engineering
│   └── train_baseline.py          # training e metriche di valutazione
├── run_fase1_baseline.py          # Fase 1: baseline in-distribution (tutti i dataset)
├── 03_cross_dataset.py            # Fase 2a: cross-dataset (CIC-IDS2017 → CSE-CIC-IDS2018)
├── 06_baseline_ctu13.py           # Fase 1: baseline dedicata a CTU-13
├── 07_lobo_experiment.py          # Fase 2b: LOBO (Leave-One-Botnet-Out)
├── 08_shap_analysis.py            # Fase 3: interpretabilità SHAP (in-distribution vs zero-day)
├── 09_mitigation.py               # Fase 4: mitigazione tramite feature pruning (Mitigazione 1)
├── 10_mitigation.py               # Fase 4: mitigazione tramite feature robuste (Mitigazione 2)
├── 12_svm.py                      # Valutazione comparativa SVM standalone
├── 04_plot_results.py             # generazione grafico di confronto cross-dataset
├── requirements.txt
└── results/                       # metriche (.json) e grafici (.png) generati
```

## Come riprodurre gli esperimenti

```bash
pip install -r requirements.txt
```

Popolare `data/raw/` con i dataset (vedi sezione "Dataset e citazioni" sotto),
poi eseguire gli script nell'ordine seguente:

1. `python run_fase1_baseline.py` — baseline in-distribution su CIC-IDS2017 e CSE-CIC-IDS2018
2. `python 06_baseline_ctu13.py` — baseline in-distribution su CTU-13
3. `python 03_cross_dataset.py` — esperimento cross-dataset (2017 → 2018)
4. `python 04_plot_results.py` — grafico di confronto in-distribution vs cross-dataset
5. `python 07_lobo_experiment.py` — esperimento LOBO (impostare `TARGET_ZERO_DAY` nello script per ripetere su famiglie diverse)
6. `python 08_shap_analysis.py` — analisi SHAP (in-distribution vs zero-day)
7. `python 09_mitigation.py / 10_mitigation.py` — esperimenti di mitigazione (feature pruning e selezione feature robuste)

Ogni script salva le proprie metriche in `results/*.json` e, dove previsto,
i grafici in `results/*.png`.

## Sintesi metodologica

- **Modelli**: Random Forest e XGBoost (ensemble tree-based), scelti per il
  miglior compromesso accuratezza/interpretabilità su dati tabellari di rete.
- **Metriche**: F1, Precision, Recall e AUROC sulla classe botnet, più False
  Positive Rate — mai la sola accuracy, fuorviante su classi sbilanciate.
- **Prevenzione data leakage**: rimozione sistematica di IP, porte e
  timestamp assoluti; scaling e SMOTE fittati esclusivamente sul training
  set; nei test Zero-Day (LOBO/mitigazione) il traffico benigno di test è
  sempre disgiunto per riga da quello di training.
- **Sbilanciamento delle classi**: gestito con SMOTE (target 30% della
  classe minoritaria sul training set), evitando di riportare le classi a
  un rapporto 50/50 irrealistico.
- **Validazione Statistica**: condotta su 15 iterazioni indipendenti per i dataset principali, e su 3 iterazioni robuste per i pesanti calcoli su CTU-13 e LOBO, riportando media e deviazione standard (media +- std).

## Sintesi dei risultati principali

| Esperimento | Risultato chiave |
|---|---|
| Baseline in-distribution | F1 ≥ 0.95 su CIC-IDS2017/CSE-CIC-IDS2018; F1 0.27–0.36 su CTU-13 (feature set più limitato) |
| Cross-dataset (2017→2018) | Crollo da F1 0.976 (in-distribution) a F1 0.0004 (cross-dataset) |
| LOBO (Zero-Day) | Recall dal 90% (in-distribution) al 68% (Rbot) e 42% (Virut) |
| SHAP | Le stesse feature volumetriche (Dur, SrcBytes, TotBytes) dominano sia in-distribution che zero-day: il problema è uno spostamento nei valori, non nell'importanza delle feature |
| Mitigazione 1 (feature pruning) | Controproducente: Recall su Rbot scende dal 68% al 16% dopo la rimozione delle feature volumetriche |
| Mitigazione 2 (feature robuste) | Efficace: l'uso mirato di feature basate su SHAP recupera l'F1-Score fino a 0.81 (Rbot) e 0.58 (Virut) |

## Dataset e citazioni

- **CTU-13**: Garcia, S., Grill, M., Stiborek, J., Zunino, A. "An empirical
  comparison of botnet detection methods." Computers & Security, vol. 45,
  pp. 100-123, 2014. — https://www.stratosphereips.org/datasets-ctu13
  (licenza CC-BY)
- **CIC-IDS2017 / CSE-CIC-IDS2018**: Sharafaldin, I., Lashkari, A.H.,
  Ghorbani, A.A. "Toward Generating a New Intrusion Detection Dataset and
  Intrusion Traffic Characterization." ICISSP, 2018. —
  https://www.unb.ca/cic/datasets/

I dataset non sono inclusi nel repository per dimensioni e licenza; vanno
scaricati separatamente dalle fonti sopra indicate.
