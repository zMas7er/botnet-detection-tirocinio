import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

os.makedirs("results", exist_ok=True)

cm = np.array([[227458, 43],
               [   15, 85842]])

plt.figure(figsize=(7, 5))
sns.set_theme(style="white")

# Heatmap
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Benigno (Predetto)", "Botnet (Predetto)"],
            yticklabels=["Benigno (Reale)", "Botnet (Reale)"],
            annot_kws={"size": 14, "weight": "bold"})

plt.title("Matrice di Confusione - XGBoost (In-Distribution 2018)", fontsize=14, pad=15)
plt.ylabel("Classe Reale", fontsize=12, fontweight="bold")
plt.xlabel("Classe Predetta dal Modello", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig("results/confusion_matrix_2018.png", dpi=300)
print("✅ Grafico generato con successo in 'results/confusion_matrix_2018.png'")
plt.show()