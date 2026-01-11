# vizualizari/plot_praguri.py
"""
Comparație performanță pentru diferite praguri de decizie
(Accuracy, Recall, Precision, F1 pentru clasa Stress)
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONFIGURARE
# =========================

# CSV-ul generat anterior din analiza pragurilor
CSV_PATH = "data/results/threshold_sweep_summary.csv"

# Folderul unde salvăm figura (îl ai deja în proiect)
OUT_DIR = Path("figs_dataset")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FIG = OUT_DIR / "comparatie_praguri.png"


# =========================
# ÎNCĂRCARE DATE
# =========================

df = pd.read_csv(CSV_PATH)

# sortăm pragurile descrescător: 0.50 → 0.35
df = df.sort_values("threshold", ascending=False)

thresholds = df["threshold"]


# =========================
# PLOT
# =========================

plt.figure(figsize=(10, 6))

plt.plot(thresholds, df["acc"], marker="o", label="Accuracy")
plt.plot(thresholds, df["recall_stress"], marker="o", label="Recall (Stress)")
plt.plot(thresholds, df["precision_stress"], marker="o", label="Precision (Stress)")
plt.plot(thresholds, df["f1_stress"], marker="o", label="F1 (Stress)")

plt.xlabel("Prag de decizie")
plt.ylabel("Scor")
plt.title("Influența pragului de decizie asupra performanței")
plt.grid(True, alpha=0.3)
plt.legend()

# inversăm axa X pentru lizibilitate
plt.gca().invert_xaxis()

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=200)
plt.close()

print(f"[SALVAT] {OUT_FIG.resolve()}")
