import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# IMPORTANT: rulezi scriptul din ROOT cu PYTHONPATH=.
from build_dataset import load_feature_dataset
from modele.run_loso import run_loso


# ================= CONFIG =================
FEATURES_PATH = "data/features/wesad_features_all.parquet"
FIGS_DIR = "figs_dataset"


# ================= LOAD DATA =================
X, y, groups = load_feature_dataset(FEATURES_PATH)


# ================= LOSO – COLECTARE SCORE-URI =================
# return_scores=True → primim y_true și y_score pentru ROC / PR
_, y_true, y_score = run_loso(
    X, y, groups,
    model_name="logreg",
    threshold=0.40,
    return_scores=True
)


# ================= ROC CURVE =================
fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"LogReg LOSO (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Logistic Regression (LOSO)")
plt.legend()
plt.grid(alpha=0.3)

plt.savefig(f"{FIGS_DIR}/roc_logreg_loso.png", dpi=180, bbox_inches="tight")
plt.close()


# ================= PRECISION–RECALL CURVE =================
precision, recall, _ = precision_recall_curve(y_true, y_score)

plt.figure(figsize=(6, 5))
plt.plot(recall, precision, label="LogReg LOSO")
plt.xlabel("Recall (Stress)")
plt.ylabel("Precision (Stress)")
plt.title("Precision–Recall Curve – Logistic Regression (LOSO)")
plt.grid(alpha=0.3)

plt.savefig(f"{FIGS_DIR}/pr_logreg_loso.png", dpi=180, bbox_inches="tight")
plt.close()

print("[SALVAT] ROC și Precision–Recall curves")
