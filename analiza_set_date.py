from preprocesare._01_incarcare_taiere import load_data_wesad, cut_30s, map_labels_to_binary_vector
from preprocesare._02_filtrare_semnale import filter_eda, filter_bvp, filter_acc, filter_temp
from feature_extraction._01_ferestre_feature import sliding_windows

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

DATASET_PATH = Path("data/WESAD")
SUBJECTS = [f"S{i}" for i in range(2, 18) if i != 12]

FS = {'EDA': 4, 'TEMP': 4, 'ACC': 32, 'BVP': 64}
FS_LABEL = 700

WINDOW_S = 30
STEP_S = 5
MAX_TRANSITION_RATIO = 0.25

OUT_DIR = Path("figs_dataset")
OUT_DIR.mkdir(exist_ok=True, parents=True)

def savefig_nice(path, tight=True, dpi=180, close=True):
    if tight:
        plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"[SALVAT] {Path(path).resolve()}")
    if close:
        plt.close()

y_all = []

for subj in SUBJECTS:
    pkl_path = DATASET_PATH / f"{subj}.pkl"
    if not pkl_path.exists():
        print(f"[AVERT] Fișier lipsă: {pkl_path}, sărim.")
        continue

    signals, labels = load_data_wesad(pkl_path)
    signals, labels = cut_30s(signals, labels, fs_dict=FS, fs_label=FS_LABEL)

    # filtrare semnale (ca în pipeline)
    signals["EDA"] = filter_eda(signals["EDA"], fs=FS["EDA"])
    signals["BVP"] = filter_bvp(signals["BVP"], fs=FS["BVP"])
    signals["TEMP"] = filter_temp(signals["TEMP"], fs=FS["TEMP"])
    signals["ACC"] = filter_acc(signals["ACC"], fs=FS["ACC"])

    # labels binare {0,1,-1}
    labels_bin = map_labels_to_binary_vector(labels)

    # ferestre + y (labels_list)
    _, _, _, _, labels_list = sliding_windows(
        signals=signals,
        labels=labels_bin,
        fs_dict=FS,
        fs_label=FS_LABEL,
        window_s=WINDOW_S,
        step_s=STEP_S,
        max_transition_ratio=MAX_TRANSITION_RATIO
    )

    y_subj = np.asarray(labels_list, dtype=np.int8)
    if y_subj.size == 0:
        print(f"{subj}: 0 ferestre (posibil prea strict MAX_TRANSITION_RATIO)")
        continue

    u, c = np.unique(y_subj, return_counts=True)
    print(f"{subj}: ferestre={len(y_subj)}  class_counts={dict(zip(u.tolist(), c.tolist()))}")

    y_all.append(y_subj)

if len(y_all) == 0:
    raise RuntimeError("Nu s-a generat nicio fereastră. Relaxează MAX_TRANSITION_RATIO sau verifică datele.")

y_all = np.concatenate(y_all)

labels_binary = np.where(y_all == 1, 'Stress', 'Non-stress')
counts = pd.Series(labels_binary).value_counts()
percentages = counts / counts.sum() * 100

print("\n--- Distribuție totală PE FERESTRE ---")
print(counts)
print(percentages.round(2))

order = ['Stress', 'Non-stress']
counts = counts.reindex(order, fill_value=0)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
ax = counts.plot(kind='bar', rot=0)
plt.title("Distribuția pe ferestre (Stress vs Non-Stress)")
plt.ylabel("Număr ferestre")
plt.grid(axis='y', alpha=0.3)

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2, height, f"{int(height)}",
            ha='center', va='bottom', fontsize=9)

plt.subplot(1, 2, 2)
plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
plt.title("Proporția pe ferestre")
plt.tight_layout()

savefig_nice(OUT_DIR / "class_distribution_WINDOWS_stress_vs_nonstress.png", close=True)
