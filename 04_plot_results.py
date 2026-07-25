import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("results", exist_ok=True)

# 1. VALORI STATISTICI AGGIORNATI (Da Tabella 4.2 e Capitolo 5)
f1_means = [0.9761, 0.0004]
auroc_means = [0.9999, 0.9395]

# 2. DEVIAZIONI STANDARD (Per le barre di errore)
f1_stds = [0.0048, 0.0000]
auroc_stds = [0.0002, 0.0000]

labels = ['Test In-Distribution\n(Train 2017 -> Test 2017)', 'Test Cross-Dataset\n(Train 2017 -> Test 2018)']
x = np.arange(len(labels))
width = 0.35  

fig, ax = plt.subplots(figsize=(10, 6))

rects1 = ax.bar(x - width/2, f1_means, width, yerr=f1_stds, capsize=5, label='F1-Score (Botnet)', color='#2ca02c')
rects2 = ax.bar(x + width/2, auroc_means, width, yerr=auroc_stds, capsize=5, label='AUROC', color='#1f77b4')

ax.set_ylabel('Punteggio (0 - 1.0)')
ax.set_title('Confronto Prestazioni Modello XGBoost (Effetto della Generalizzazione)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.set_ylim([0, 1.1])

def autolabel(rects, stds):
    for rect, std in zip(rects, stds):
        height = rect.get_height()
        # Offset dinamico: se c'è deviazione standard, alza il numero per non coprire il trattino
        offset = 5 if std > 0 else 3
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height + std),
                    xytext=(0, offset),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

autolabel(rects1, f1_stds)
autolabel(rects2, auroc_stds)

# Salvataggio e visualizzazione
plt.tight_layout()
plt.savefig('results/generalization_collapse_plot_updated.png', dpi=300)
print("✅ Grafico aggiornato generato e salvato in 'results/generalization_collapse_plot_updated.png'")
plt.show()