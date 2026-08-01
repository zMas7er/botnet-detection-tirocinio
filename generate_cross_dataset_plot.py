import matplotlib.pyplot as plt
import numpy as np

# Etichette delle categorie sull'asse X
labels = [
    'Test In-Distribution\n(Train 2017 -> Test 2017)', 
    'Test Cross-Dataset\n(Train 2017 -> Test 2018)'
]

# Dati ufficiali dai log per F1-Score e AUROC [In-Distribution, Cross-Dataset]
f1_scores = [0.9619, 0.0002]    # F1-Score (Botnet)
auroc_scores = [1.0000, 0.9466]  # AUROC

x = np.arange(len(labels))
width = 0.35  # Spessore delle barre

fig, ax = plt.subplots(figsize=(9, 6))

# Creazione delle barre con i colori standard (Verde per F1, Blu per AUROC)
rects1 = ax.bar(x - width/2, f1_scores, width, label='F1-Score (Botnet)', color='#2ca02c')
rects2 = ax.bar(x + width/2, auroc_scores, width, label='AUROC', color='#1f77b4')

# Configurazione di assi, titolo e legenda
ax.set_ylabel('Punteggio (0 - 1.0)', fontsize=11)
ax.set_title('Confronto Prestazioni Modello XGBoost (Effetto della Generalizzazione)', fontsize=12, fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.legend(loc='upper right', fontsize=10)
ax.set_ylim(0, 1.1)

# Funzione per stampare i valori numerici esatti sopra ogni singola barra
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        # Se il valore è vicino allo zero, posizioniamo comunque l'etichetta sopra la barra in modo leggibile
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()

# Salvataggio dell'immagine in alta definizione pronta per la tesi
output_path = 'figura_cross_dataset_xgboost.png'
plt.savefig(output_path, dpi=300)
print(f"Grafico generato e salvato correttamente come: {output_path}")

plt.show()