import numpy as np
import pandas as pd   # ✅ LIPSEA — FOARTE IMPORTANT

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from modele.metrics import compute_classification_metrics, compute_confusion_matrix


def run_loso(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    model_name="logreg",
    threshold=0.35,
    return_scores=False   # ✅ NOU
):
    """
    LOSO (Leave-One-Subject-Out)

    Dacă return_scores=True:
      → returnează (df_res, y_true_all, y_score_all)
      → necesar pentru ROC / Precision–Recall
    """

    logo = LeaveOneGroupOut()
    results = []

    y_true_all = []
    y_score_all = []

    # =========================
    # 1) Alegere model
    # =========================
    if model_name == "logreg":
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42
        )
        use_proba = True
        ### prin class_weight="balanced" gestionez dezechilibrul claselor, creste penalizarea eroriilor pe clasa minoritara
    elif model_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
        use_proba = True
    elif model_name == "svm":
        clf = SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=42
        )
        use_proba = True
    else:
        raise ValueError("model_name must be 'logreg', 'rf', or 'svm'")


    # =========================
    # 2) Pipeline
    # =========================
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),    
        ("clf", clf)
    ])
    ## prin imputer cu mediană evit NaN-urile (dacă există), prin StandardScaler normalizez datele Z-score      
    cm_total = np.zeros((2, 2), dtype=int)

    # =========================
    # 3) LOSO
    # =========================
    for fold, (train_idx, test_idx) in enumerate(
        logo.split(X, y, groups=groups), start=1
    ):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipe.fit(X_train, y_train)

        # probabilități
        y_prob = pipe.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        # colectare scoruri pt ROC / PR
        if return_scores:
            y_true_all.extend(y_test.tolist())
            y_score_all.extend(y_prob.tolist())

        m = compute_classification_metrics(y_test, y_pred)
        cm = compute_confusion_matrix(y_test, y_pred)
        cm_total += cm["matrix"]

        results.append({
            "fold": fold,
            "acc": m["accuracy"],
            "bal_acc": m["balanced_accuracy"],
            "precision_stress": m["precision_stress"],
            "recall_stress": m["recall_stress"],
            "f1_stress": m["f1_stress"],
            "f1_macro": m["f1_macro"],
        })

        print(
            f"[Fold {fold:02d}] "
            f"Recall_stress={m['recall_stress']:.3f} | "
            f"F1_stress={m['f1_stress']:.3f} |"
            f"Balanced_Acc={m['balanced_accuracy']:.3f} |"
            f"Acc={m['accuracy']:.3f}"
        )

    df_res = pd.DataFrame(results)

    print("\n=== Rezumat LOSO (medii) ===")
    print(df_res.mean())

    print("\n=== Confusion Matrix TOTAL ===")
    print(cm_total)

    if return_scores:
        return df_res, np.array(y_true_all), np.array(y_score_all)

    return df_res
