# main_train_all.py
"""
WESAD – Stress Detection (LOSO)

Acest fișier conține două moduri de rulare:

A) RUN FINAL (best threshold)
   - rulează LogReg o singură dată cu pragul ales (ex: 0.40)
   - salvează un singur CSV

B) THRESHOLD SWEEP (analiză praguri)
   - rulează LogReg pe mai multe praguri: 0.50, 0.45, 0.40, 0.35
   - salvează CSV pentru fiecare prag
   - salvează un tabel summary + un grafic cu legendă

⚠️ Ca să nu te încurci:
- la final, pentru licență, lași activ doar (A) și ții RUN_SWEEP=False.
"""

from pathlib import Path

from build_dataset import (
    build_full_dataset,
    save_feature_dataset,
    load_feature_dataset
)

from modele.run_loso import run_loso, choose_threshold_from_train

from modele.cnn_dataset import build_cnn_dataset
from modele.run_loso_cnn import run_loso_cnn

import os

# limitează BLAS / numpy / sklearn
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"


# =========================
# CONFIGURARE GENERALĂ
# =========================

LOAD_FROM_PARQUET =True
FEATURES_PATH1 = "./data/features/wesad_features_all.parquet"
FEATURES_PATH2 = "./data/features/wesad_features_all.csv"
FEATURES_PATH_WITHOUT_NOISE1 = "./data/features/wesad_features_no_noise.parquet"
FEATURES_PATH_WITHOUT_NOISE2 = "./data/features/wesad_features_no_noise.csv"
FEATURES_PATH_WITHOUT_ACC1 = "./data/features/wesad_features_no_acc.parquet"
FEATURES_PATH_WITHOUT_ACC2 = "./data/features/wesad_features_no_acc.csv"
WESAD_PATH = "data/WESAD"

# Foldere output
RESULTS_DIR = Path("data/results")
FIGS_DIR = Path("figs_dataset")   # aici salvăm figura cu pragurile


# =========================
# CONFIGURARE RULARE
# =========================

RUN_RF = True
RUN_LOGREG = True
RUN_SVM = True
RUN_CNN=False

# ✅ switch simplu (nu mai comentezi cod)
RUN_SWEEP = False   # True doar când vrei analiza pragurilor

# Pragul final (ales după analiză)
BEST_THRESHOLD = 0.40

# Praguri pentru sweep (analiză)
SWEEP_THRESHOLDS = [0.50, 0.45, 0.40, 0.35]


def main():
    # =========================
    # 0) Asigură folderele de output
    # =========================
    Path("data/features").mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    # =========================
    # 1) LOAD / BUILD DATASET
    # =========================
    if LOAD_FROM_PARQUET:
        print("[INFO] Încarc datasetul din parquet...")
        X, y, groups = load_feature_dataset(FEATURES_PATH1)
    else:
        print("[INFO] Construiesc datasetul din WESAD...")
        X, y, groups = build_full_dataset(WESAD_PATH)

        print("[INFO] Salvez datasetul pentru rulări viitoare...")
        save_feature_dataset(X, y, groups, FEATURES_PATH_WITHOUT_NOISE1,FEATURES_PATH_WITHOUT_NOISE2)

    print("\n[INFO] Dataset încărcat:")
    print("  X shape:", X.shape)
    print("  y shape:", y.shape)
    print("  subjects:", len(set(groups)))

    # =========================
    # 2) ANTRENARE + EVALUARE (LOSO)
    # =========================

    # ---------------------------------------------------------
    # A) RUN FINAL (BEST THRESHOLD)  -> pentru rezultatul final
    # ---------------------------------------------------------
    if RUN_LOGREG:
        print("\n[A] RUN FINAL: Logistic Regression (LOSO) cu BEST_THRESHOLD")
        print(f"    BEST_THRESHOLD = {BEST_THRESHOLD:.2f}")

        res_best = run_loso(X, y, groups, model_name="logreg", threshold=BEST_THRESHOLD)

        out_best = RESULTS_DIR / f"loso_logreg_best_thr{int(BEST_THRESHOLD * 100)}.csv"
        res_best.to_csv(out_best, index=False)
        print(f"[SALVAT] {out_best}")

    # ---------------------------------------------------------
    # B) THRESHOLD SWEEP (analiză praguri)  -> doar când vrei graficul
    # ---------------------------------------------------------
    if RUN_SWEEP:
        run_threshold_sweep(X, y, groups)

    # =========================
    # (opțional) RF
    # =========================
    if RUN_RF:
        print("\n[INFO] Rulez Random Forest (LOSO)...")
        res_rf = run_loso(X, y, groups, model_name="rf")
        out_rf = RESULTS_DIR / "loso_rf.csv"
        res_rf.to_csv(out_rf, index=False)
        print(f"[SALVAT] {out_rf}")
    # =========================
    # (opțional) SVM
    # =========================
    if RUN_SVM:
        print("\n[INFO] Rulez SVM (LOSO)...")
        res_svm = run_loso(X, y, groups, model_name="svm")
        out_svm = RESULTS_DIR / "loso_svm.csv"
        res_svm.to_csv(out_svm, index=False)
        print(f"[SALVAT] {out_svm}")

#=========================================
# MODEL CNN
#=========================================
    if RUN_CNN:
        X_cnn, y_cnn, groups_cnn = build_cnn_dataset(
            wesad_dir=WESAD_PATH,
            window_s=30,
            step_s=5,
            target_fs=32
        )

        res_cnn = run_loso_cnn(X_cnn, y_cnn, groups_cnn, epochs=15, batch_size=64,
                            use_dynamic_threshold=True, objective="f1_stress")


# ============================================================
# FUNCȚIE: THRESHOLD SWEEP + CSV + FIGURĂ (cu legendă)
# ============================================================
def run_threshold_sweep(X, y, groups):
    """
    Rulează LogReg LOSO pe mai multe praguri și salvează:
    - CSV per prag
    - un summary CSV
    - o figură (Accuracy / Balanced Acc / Precision / Recall / F1 vs threshold)
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    print("\n[B] THRESHOLD SWEEP: Logistic Regression (LOSO) pe praguri")
    print("    Praguri:", SWEEP_THRESHOLDS)

    summary_rows = []

    for thr in SWEEP_THRESHOLDS:
        print(f"\n--- THRESHOLD = {thr:.2f} ---")
        df = run_loso(X, y, groups, model_name="logreg", threshold=thr)

        # Salvează rezultatele pe fold-uri
        out_csv = RESULTS_DIR / f"loso_logreg_thr{int(thr * 100)}.csv"
        df.to_csv(out_csv, index=False)
        print(f"[SALVAT] {out_csv}")

        # Rezumat pe prag (medii pe subiecți)
        row = {
            "threshold": thr,
            "acc": float(df["acc"].mean()),
            "bal_acc": float(df["bal_acc"].mean()),
            "precision_stress": float(df["precision_stress"].mean()),
            "recall_stress": float(df["recall_stress"].mean()),
            "f1_stress": float(df["f1_stress"].mean()),
            "f1_macro": float(df["f1_macro"].mean()),
        }
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values("threshold", ascending=False)

    print("\n=== REZUMAT praguri (medii pe subiecți) ===")
    print(summary.round(4))

    out_summary = RESULTS_DIR / "threshold_sweep_summary.csv"
    summary.to_csv(out_summary, index=False)
    print(f"[SALVAT] {out_summary}")

    # ---------------------------
    # FIGURĂ cu legendă clară
    # ---------------------------
    plt.figure(figsize=(10, 6))

    plt.plot(summary["threshold"], summary["acc"], marker="o", label="Accuracy")
    plt.plot(summary["threshold"], summary["bal_acc"], marker="o", label="Balanced Accuracy")
    plt.plot(summary["threshold"], summary["precision_stress"], marker="o", label="Precision (Stress)")
    plt.plot(summary["threshold"], summary["recall_stress"], marker="o", label="Recall (Stress)")
    plt.plot(summary["threshold"], summary["f1_stress"], marker="o", label="F1 (Stress)")

    plt.gca().invert_xaxis()
    plt.xlabel("Threshold (P(stress) ≥ threshold)")
    plt.ylabel("Valoare metrică")
    plt.title("Efectul pragului asupra performanței (LogReg, LOSO)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")

    out_fig = FIGS_DIR / "threshold_sweep_metrics.png"
    plt.savefig(out_fig, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[SALVAT] {out_fig}")


if __name__ == "__main__":
    main()
