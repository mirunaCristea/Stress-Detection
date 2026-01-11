# analiza_set_date.py
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preprocesare._01_incarcare_taiere import load_data_wesad, cut_30s, map_labels_to_binary_vector
from preprocesare._02_filtrare_semnale import filter_eda, filter_bvp, filter_acc, filter_temp
from feature_extraction._01_ferestre_feature import sliding_windows


# ---------------- CONFIG ----------------
DATASET_PATH = Path("data/WESAD")
SUBJECTS = [f"S{i}" for i in range(2, 18) if i != 12]

FS = {"EDA": 4, "TEMP": 4, "ACC": 32, "BVP": 64}
FS_LABEL = 700

WINDOW_S = 30
STEP_S = 5
MAX_TRANSITION_RATIO = 0.25

OUT_DIR = Path("figs_dataset")
OUT_DIR.mkdir(exist_ok=True, parents=True)


def savefig_nice(path: Path, dpi: int = 180):
    """Salvează figura curentă într-un mod consistent (tight + dpi)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[SALVAT] {path.resolve()}")


def collect_window_labels_for_subject(subj: str) -> np.ndarray:
    """
    Rulează același pipeline ca în build_dataset și întoarce etichetele pe ferestre
    (0=non-stres, 1=stres) pentru un subiect.
    """
    pkl_path = DATASET_PATH / f"{subj}.pkl"
    if not pkl_path.exists():
        print(f"[AVERT] Fișier lipsă: {pkl_path}, sărim.")
        return np.array([], dtype=np.int8)

    signals, labels = load_data_wesad(pkl_path)
    signals, labels = cut_30s(signals, labels, fs_dict=FS, fs_label=FS_LABEL, cut_s=30)

    # Filtrare semnale (aceeași ca în pipeline)
    signals["EDA"] = filter_eda(signals["EDA"], fs=FS["EDA"])
    signals["BVP"] = filter_bvp(signals["BVP"], fs=FS["BVP"])
    signals["TEMP"] = filter_temp(signals["TEMP"], fs=FS["TEMP"])
    signals["ACC"] = filter_acc(signals["ACC"], fs=FS["ACC"])

    labels_bin = map_labels_to_binary_vector(labels)

    # Ferestre + labels_list (majoritate pe fereastră)
    _, _, _, _, labels_list = sliding_windows(
        signals=signals,
        labels=labels_bin,
        fs_dict=FS,
        fs_label=FS_LABEL,
        window_s=WINDOW_S,
        step_s=STEP_S,
        max_transition_ratio=MAX_TRANSITION_RATIO,
    )

    return np.asarray(labels_list, dtype=np.int8)


def plot_window_class_distribution(y_all: np.ndarray, out_path: Path):
    """
    Plotează distribuția claselor pe ferestre:
      - bar chart (număr ferestre)
      - pie chart (proporție)
    """
    if y_all.size == 0:
        raise RuntimeError("Nu există ferestre în y_all.")

    labels_str = np.where(y_all == 1, "Stress", "Non-stress")
    counts = pd.Series(labels_str).value_counts()

    print("\n--- Distribuție totală PE FERESTRE ---")
    print(counts)
    print((counts / counts.sum() * 100).round(2))

    # Ordine fixă în grafic
    counts = counts.reindex(["Stress", "Non-stress"], fill_value=0)

    plt.figure(figsize=(12, 4))

    # Bar
    plt.subplot(1, 2, 1)
    ax = counts.plot(kind="bar", rot=0)
    plt.title("Distribuția pe ferestre (Stress vs Non-Stress)")
    plt.ylabel("Număr ferestre")
    plt.grid(axis="y", alpha=0.3)

    for p in ax.patches:
        h = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2, h, f"{int(h)}",
                ha="center", va="bottom", fontsize=9)

    # Pie
    plt.subplot(1, 2, 2)
    plt.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
    plt.title("Proporția pe ferestre")

    savefig_nice(out_path)


def main():
    y_list = []

    for subj in SUBJECTS:
        y_subj = collect_window_labels_for_subject(subj)
        if y_subj.size == 0:
            print(f"{subj}: 0 ferestre (posibil MAX_TRANSITION_RATIO prea strict).")
            continue

        u, c = np.unique(y_subj, return_counts=True)
        print(f"{subj}: ferestre={len(y_subj)} class_counts={dict(zip(u.tolist(), c.tolist()))}")
        y_list.append(y_subj)

    if not y_list:
        raise RuntimeError("Nu s-a generat nicio fereastră. Verifică setările/fișierele.")

    y_all = np.concatenate(y_list)
    plot_window_class_distribution(
        y_all=y_all,
        out_path=OUT_DIR / "class_distribution_WINDOWS_stress_vs_nonstress.png",
    )


if __name__ == "__main__":
    main()
