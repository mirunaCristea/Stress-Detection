# vizualizari/plot_compare_models.py
"""
Compară performanța LOSO între două modele (ex: RF vs LogReg)
pe baza CSV-urilor salvate în data/results/.

Generează o figură cu:
- bar chart pentru mediile metricilor (Accuracy, Balanced Acc, Precision/Recall/F1 Stress, F1 Macro)
- și salvează figura în figs_dataset/
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG (ajustează dacă ai alte nume)
# =========================
RF_CSV = Path("data/results/loso_rf.csv")
LOGREG_CSV = Path("data/results/loso_logreg_best_thr40.csv")  # pragul final

OUT_DIR = Path("figs_dataset")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FIG = OUT_DIR / "compare_models_loso_rf_vs_logreg.png"

# ce metrici vrem să comparăm (trebuie să existe în ambele CSV-uri)
METRICS = [
    ("acc", "Accuracy"),
    ("bal_acc", "Balanced Accuracy"),
    ("precision_stress", "Precision (Stress)"),
    ("recall_stress", "Recall (Stress)"),
    ("f1_stress", "F1 (Stress)"),
    ("f1_macro", "F1 Macro"),
]


def read_and_summarize(path: Path, model_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Nu găsesc {model_name}: {path.resolve()}")

    df = pd.read_csv(path)

    # verificăm dacă toate coloanele există
    missing = [col for col, _ in METRICS if col not in df.columns]
    if missing:
        raise ValueError(
            f"{model_name} ({path.name}) nu are coloanele: {missing}\n"
            f"Coloane găsite: {list(df.columns)}"
        )

    # medie + deviație standard (pe subiecți / folduri)
    summary = []
    for col, label in METRICS:
        summary.append({
            "metric": col,
            "label": label,
            "mean": float(df[col].mean()),
            "std": float(df[col].std(ddof=1)) if len(df[col]) > 1 else 0.0,
            "model": model_name
        })

    return pd.DataFrame(summary)


def main():
    rf_sum = read_and_summarize(RF_CSV, "Random Forest")
    lr_sum = read_and_summarize(LOGREG_CSV, "LogReg (best thr=0.40)")

    # combinăm pentru plot
    all_sum = pd.concat([rf_sum, lr_sum], ignore_index=True)

    # pivot ca să avem două coloane (RF vs LR) pe fiecare metrică
    pivot_mean = all_sum.pivot(index="label", columns="model", values="mean")
    pivot_std = all_sum.pivot(index="label", columns="model", values="std")

    # =========================
    # PLOT (bar chart comparativ)
    # =========================
    plt.figure(figsize=(12, 6))

    labels = pivot_mean.index.tolist()
    models = pivot_mean.columns.tolist()

    x = range(len(labels))
    bar_width = 0.38

    # poziții bare
    x1 = [i - bar_width/2 for i in x]
    x2 = [i + bar_width/2 for i in x]

    # valori
    m1 = pivot_mean[models[0]].values
    m2 = pivot_mean[models[1]].values

    s1 = pivot_std[models[0]].values
    s2 = pivot_std[models[1]].values

    plt.bar(x1, m1, width=bar_width, yerr=s1, capsize=4, label=models[0])
    plt.bar(x2, m2, width=bar_width, yerr=s2, capsize=4, label=models[1])

    plt.xticks(list(x), labels, rotation=20, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Scor")
    plt.title("Comparație performanță LOSO: Random Forest vs Logistic Regression")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(loc="best")

    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=180, bbox_inches="tight")
    plt.close()

    print(f"[SALVAT] {OUT_FIG.resolve()}")
    print("\n=== Medii (pe subiecți) ===")
    print(pivot_mean.round(4))


if __name__ == "__main__":
    main()
