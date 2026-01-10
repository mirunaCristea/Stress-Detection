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

from modele.run_loso import run_loso

from modele.cnn_dataset import build_cnn_dataset
from modele.run_loso_cnn import run_loso_cnn
from modele.rf_cnn import save_rf_model, save_cnn_model, train_final_cnn,train_final_rf,train_subject_holdout_rf, train_subject_holdout_cnn

import os
from modele.plots import (
    plot_confusion_matrix, plot_roc_curve, plot_pr_curve,
    plot_score_distributions, plot_calibration_curve,
    threshold_sweep, plot_threshold_sweep,
    plot_rf_feature_importance, plot_permutation_importance,plot_loss_and_accuracy,
)
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
RUN_LOGREG = False
RUN_SVM = False
RUN_CNN=True
SAVE_MODEL=True

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
        #res_rf = run_loso(X, y, groups, model_name="rf")
        #out_rf = RESULTS_DIR / "loso_rf.csv"
        #res_rf.to_csv(out_rf, index=False)
        #print(f"[SALVAT] {out_rf}")
        if SAVE_MODEL:
            out = train_subject_holdout_rf(
            X=X, y=y, groups=groups,
            seed=123,
            save_path="modele/saved_models/rf_train14_holdout.pkl"
)
            y_true_rf = out["y_true"]
            y_pred_rf = out["y_pred"]
            model_rf = out["model"]
            feat_cols = out["feature_cols"]
            X_te = out["X_test"]

            # probas:

            y_prob_rf = model_rf.predict_proba(X_te)[:, 1]
            plot_confusion_matrix(out["confusion_matrix"], title=f"RF CM (test={out['test_subject']})", normalize="true", savepath="modele/saved_graphs/rf_cm.png")
            plot_roc_curve(y_true_rf, y_prob_rf, title="RF ROC", savepath="modele/saved_graphs/rf_roc.png")
            plot_pr_curve(y_true_rf, y_prob_rf, title="RF PR", savepath="modele/saved_graphs/rf_pr.png")
            plot_score_distributions(y_true_rf, y_prob_rf, title="RF score distributions", savepath="modele/saved_graphs/rf_score_distributions.png")
            plot_calibration_curve(y_true_rf, y_prob_rf, title="RF calibration", savepath="modele/saved_graphs/rf_calibration.png")

            plot_rf_feature_importance(model_rf, feature_names=feat_cols, top_k=20, title="RF feature importance", savepath="modele/saved_graphs/rf_feature_importance.png")
            plot_permutation_importance(model_rf, X_te, y_true_rf, feature_names=feat_cols, scoring="f1", top_k=20, title="RF permutation importance", savepath="modele/saved_graphs/rf_permutation_importance.png")
            # final_rf, feature_cols = train_final_rf(X, y, use_scaler=True)
            # save_rf_model(final_rf, feature_cols=feature_cols, path="modele/saved_models/rf_stress_final.pkl")
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
            window_s=60,
            step_s=5,
            target_fs=32
        )

        #res_cnn = run_loso_cnn(X_cnn, y_cnn, groups_cnn, epochs=20, batch_size=64,use_dynamic_threshold=True, objective="f1_stress")
        # out_cnn = RESULTS_DIR / "loso_cnn.csv"
        # res_cnn.to_csv(out_cnn, index=False)
        # print(f"[SALVAT] {out_cnn}")
        
        if SAVE_MODEL:
            out = train_subject_holdout_cnn(
            X=X_cnn, y=y_cnn, groups=groups_cnn,
            seed=123,
            save_path="modele/saved_models/cnn_train14_holdout.pth"
)
            # final_cnn = train_final_cnn(X_cnn, y_cnn, epochs=20, batch_size=64, lr=1e-3)
            # save_cnn_model(final_cnn, in_channels=X_cnn.shape[1], path="modele/saved_models/cnn_stress_final.pth")
            y_true = out["y_true"]
            y_prob = out["y_prob"]
            cm = out["confusion_matrix"]  # dict cu "matrix"

            plot_confusion_matrix(cm, title=f"CNN CM (test={out['test_subject']})", normalize="true", savepath="modele/saved_graphs/cnn_cm.png")
            plot_roc_curve(y_true, y_prob, title="CNN ROC", savepath="modele/saved_graphs/cnn_roc.png")
            plot_pr_curve(y_true, y_prob, title="CNN PR", savepath="modele/saved_graphs/cnn_pr.png")
            plot_score_distributions(y_true, y_prob, title="CNN score distributions", savepath="modele/saved_graphs/cnn_score_distributions.png")
            plot_calibration_curve(y_true, y_prob, title="CNN calibration", savepath="modele/saved_graphs/cnn_calibration.png")
            sw = threshold_sweep(y_true, y_prob)
            plot_threshold_sweep(sw, title="CNN metrics vs threshold", savepath="modele/saved_graphs/cnn_threshold_sweep.png")
            plot_loss_and_accuracy(out["history"], title="CNN: Loss & Accuracy per epoch", savepath="modele/saved_graphs/cnn_loss_accuracy.png")
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


#=================FIGURI MODELE==================


if __name__ == "__main__":
    main()
