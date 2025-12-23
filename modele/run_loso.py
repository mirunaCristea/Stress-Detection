# modele/train_loso.py
import numpy as np
import pandas as pd

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def run_loso(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, model_name="logreg"):
    logo = LeaveOneGroupOut()
    results = []

    # alege model
    if model_name == "logreg":
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear"
        )
    elif model_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
    else:
        raise ValueError("model_name must be 'logreg' or 'rf'")

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),   # tratează NaN din EDA/BVP etc.
        ("scaler", StandardScaler()),                    # scaling după feature extraction
        ("clf", clf)
    ])

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        test_subject = groups[test_idx][0]

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        results.append({
            "fold": fold,
            "subject": test_subject,
            "n_test": len(test_idx),
            "acc": accuracy_score(y_test, y_pred),
            "prec": precision_score(y_test, y_pred, zero_division=0),
            "rec": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        })

        print(f"[Fold {fold:02d}] subj={test_subject} | F1={results[-1]['f1']:.3f} | Acc={results[-1]['acc']:.3f}")

    df_res = pd.DataFrame(results)
    print("\n=== Rezumat LOSO ===")
    print(df_res[["acc","prec","rec","f1"]].mean())

    return df_res
