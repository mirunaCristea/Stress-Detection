# modele/run_loso_cnn.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.model_selection import LeaveOneGroupOut

from modele.metrics import compute_classification_metrics, compute_confusion_matrix
from modele.cnn_dataset import CNPTensorDataset

from modele.cnn_model import CNN1D
import time
import os

torch.set_num_threads(12)
torch.set_num_interop_threads(2)

def choose_threshold_from_train(y_true, y_prob, objective="f1_stress"):
    """
    Alege pragul (threshold) doar pe TRAIN, pentru a evita “umflarea” scorurilor pe TEST.
    """
    best_th = 0.5
    best_val = -1.0
    for th in np.linspace(0.05, 0.95, 91):
        y_pred = (y_prob >= th).astype(int)
        m = compute_classification_metrics(y_true, y_pred)
        val = m[objective]
        if val > best_val:
            best_val = val
            best_th = float(th)
    return best_th, best_val


@torch.no_grad()
def _predict_proba(model, loader, device):
    """
    Rulează modelul și întoarce:
      - y_true (np.ndarray int)
      - y_prob (np.ndarray float) = P(stress=1)
    """

    model.eval()
    probs = []
    ys = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        probs.append(p)
        ys.append(yb.numpy())
    return np.concatenate(ys), np.concatenate(probs)


def run_loso_cnn(
    X: np.ndarray,          # (N,C,T)
    y: np.ndarray,          # (N,)
    groups: np.ndarray,     # (N,)
    batch_size: int = 64,
    epochs: int = 15,
    lr: float = 1e-3,
    threshold: float = 0.35,
    use_dynamic_threshold: bool = True,
    objective: str = "f1_stress",
    device: str | None = None,
):
    """
    LOSO pentru CNN (PyTorch).
    Pentru fiecare fold:
      - Train CNN pe subiecții din train
      - Calculează probabilități pe train și test
      - Alege threshold pe train (opțional)
      - Evaluează pe test
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logo = LeaveOneGroupOut()
    results = []
    cm_total = np.zeros((2, 2), dtype=int)
    t0_total = time.perf_counter()

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups), start=1):
        t0_fold = time.perf_counter()

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        train_ds = CNPTensorDataset(X_train, y_train)
        test_ds  = CNPTensorDataset(X_test, y_test)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

        model = CNN1D(in_channels=X.shape[1]).to(device)

        # pos_weight pentru dezechilibru (doar pe TRAIN)
        n_pos = float((y_train == 1).sum())
        n_neg = float((y_train == 0).sum())
        pos_weight = torch.tensor([n_neg / (n_pos + 1e-8)], dtype=torch.float32, device=device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optim = torch.optim.Adam(model.parameters(), lr=lr)

        # ---- TRAIN ----
        model.train()

        for ep in range(epochs):
            losses = []
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                optim.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optim.step()
                losses.append(float(loss.detach().cpu().item()))

        # ---- PREDICT PROBA ----
        y_train_true, p_train = _predict_proba(model, train_loader, device)
        y_test_true,  p_test  = _predict_proba(model, test_loader, device)

        # threshold
        if use_dynamic_threshold and len(np.unique(y_train_true)) > 1:
            best_th, best_val = choose_threshold_from_train(y_train_true.astype(int), p_train, objective=objective)
        else:
            best_th, best_val = threshold, np.nan

        y_pred = (p_test >= best_th).astype(int)

        m = compute_classification_metrics(y_test_true.astype(int), y_pred)
        cm = compute_confusion_matrix(y_test_true.astype(int), y_pred)
        cm_total += cm["matrix"]

        results.append({
            "fold": fold,
            "th": best_th,
            "acc": m["accuracy"],
            "bal_acc": m["balanced_accuracy"],
            "precision_stress": m["precision_stress"],
            "recall_stress": m["recall_stress"],
            "f1_stress": m["f1_stress"],
            "f1_macro": m["f1_macro"],
        })
        fold_sec = time.perf_counter() - t0_fold

        print(
            f"[Fold {fold:02d}]  time={fold_sec:6.1f}s | th={best_th:.2f} | "
            f"Recall_stress={m['recall_stress']:.3f} | "
            f"F1_stress={m['f1_stress']:.3f} | "
            f"BalAcc={m['balanced_accuracy']:.3f} | "
            f"Acc={m['accuracy']:.3f}"
        )

    df_res = pd.DataFrame(results)

    print("\n=== Rezumat LOSO CNN (medii) ===")
    print(df_res[["acc","bal_acc","precision_stress","recall_stress","f1_stress","f1_macro"]].mean())

    print("\n=== Confusion Matrix TOTAL (CNN) ===")
    print(cm_total)
    total_sec = time.perf_counter() - t0_total
    print(f"\n[TOTAL] LOSO CNN runtime: {total_sec:.1f}s ({total_sec/60:.1f} min)")

    return df_res


