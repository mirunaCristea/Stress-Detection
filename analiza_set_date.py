from preprocesare._01_incarcare_taiere import load_data_wesad, cut_30s
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

DATASET_PATH = Path("data/WESAD")  # <- schimbă subiectul
SUBJECTS = [f"S{i}" for i in range(2,18) if i != 12]  # S2..S17, fără S12

labels_all = []
OUT_DIR = Path("figs_dataset")      # unde salvăm imaginile
OUT_DIR.mkdir(exist_ok=True, parents=True)

def savefig_nice(path, tight=True, dpi=180,close=True):
    if tight:
        plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"[SALVAT] {Path(path).resolve()}")
    if close:
        plt.close()

for subj in SUBJECTS:
    pkl_path = DATASET_PATH/f"{subj}.pkl"
    if not pkl_path.exists():
        print(f"[AVERT] Fișier lipsă: {pkl_path}, sărim.")
        continue

    signals, labels = load_data_wesad(pkl_path)
    signals, labels = cut_30s(signals, labels, fs_dict={'EDA':4,'TEMP':4,'ACC':32,'BVP':64}, fs_label=700)

    labels = labels[np.isin(labels, [1,2,3,4])]  # păstrăm doar etichetele cunoscute
    labels_all.append(labels)

# ---- CORECT: concatenează toate etichetele într-un singur vector ----
if len(labels_all) == 0:
    raise RuntimeError("Nu am găsit niciun fișier valid; verifică DATASET_PATH.")
labels_all = np.concatenate(labels_all)

# --- Mapare la 2 clase --- #
labels_binary = np.where(labels_all == 2, 'Stress', 'Non-stress')

# --- Calcule --- #
counts = pd.Series(labels_binary).value_counts()
percentages = counts / counts.sum() * 100

print("\n--- Distribuție totală ---")
print(counts)
print(percentages.round(2))

# asigură ordinea dorită în grafice
order = ['Stress', 'Non-stress']
counts = counts.reindex(order, fill_value=0)
percentages = percentages.reindex(order, fill_value=0)

# --- Ploturi --- #
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
ax = counts.plot(kind='bar', color=['tomato','skyblue'], rot=0)
plt.title("Distribuția claselor (Stress vs Non-Stress)")
plt.ylabel("Număr eșantioane")
plt.grid(axis='y', alpha=0.3)

# Adaugă etichete numerice deasupra fiecărei bare.
# Încercăm ax.bar_label (disponibil în matplotlib >= 3.4). Dacă nu e disponibil,
# revenim la fallback-ul cu ax.patches pentru compatibilitate.
try:
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', label_type='edge')
except Exception:
    max_count = counts.max() if len(counts) else 0
    offset = max(1, int(max_count * 0.01))  # offset mic deasupra barei
    for p in ax.patches:
        height = p.get_height()
        ax.text(
            p.get_x() + p.get_width() / 2,
            height + offset,
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontsize=9
        )

plt.subplot(1,2,2)
plt.pie(counts, labels=counts.index, autopct='%1.1f%%',
        colors=['tomato','skyblue'], startangle=90)
plt.title("Proporția totală a claselor")
plt.tight_layout()

savefig_nice(OUT_DIR/"class_distribution_stress_vs_nonstress.png", close=True)



