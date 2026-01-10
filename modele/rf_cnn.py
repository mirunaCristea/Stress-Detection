# modele/model_io.py
import os
import time
from typing import Any

import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from modele.cnn_dataset import CNPTensorDataset
from modele.cnn_model import CNN1D
from modele.metrics import compute_classification_metrics, compute_confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------
# CNN (PyTorch) - TRAIN FINAL
# -------------------------

def split_by_subject(
    groups: np.ndarray,
    seed: int = 123,
    test_subject: str | None = None,
):
    """
    Returnează (test_subj, train_mask, test_mask)
    """
    groups = np.asarray(groups)
    uniq = np.unique(groups)

    if test_subject is None:
        rng = np.random.default_rng(seed)
        test_subj = str(rng.choice(uniq))
    else:
        test_subj = str(test_subject)
        if test_subj not in uniq:
            raise ValueError(f"Subiect {test_subj} nu există. Avem: {uniq}")

    test_mask = (groups == test_subj)
    train_mask = ~test_mask
    return test_subj, train_mask, test_mask



def train_one_cnn(
    train_loader: DataLoader,
    in_channels: int,
    epochs: int = 15,
    lr: float = 1e-3,
    device: str = "cpu",
    pos_weight: torch.Tensor | None = None,
    print_every: int = 0,
    tag: str = "",
    threshold: float = 0.5,   # 👈 pentru accuracy
):
    """
    Antrenează 1 CNN model pe train_loader.
    Returnează (model, last_epoch_loss_mean, history)
      history = {"loss": [...], "acc": [...]}
    """
    model = CNN1D(in_channels=in_channels).to(device)

    if pos_weight is None:
        pos_weight = torch.tensor([1.0], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"loss": [], "acc": []}
    last_mean_loss = None

    for ep in range(1, epochs + 1):
        model.train()
        losses = []

        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optim.zero_grad()
            logits = model(xb).squeeze(-1)   # (B,)
            loss = criterion(logits, yb)
            loss.backward()
            optim.step()

            losses.append(float(loss.detach().cpu().item()))

            # accuracy pe train (pe batch)
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs >= threshold).float()
                correct += int((preds == yb).sum().item())
                total += int(yb.numel())

        last_mean_loss = float(np.mean(losses)) if losses else None
        acc = float(correct / (total + 1e-12)) if total > 0 else np.nan

        history["loss"].append(last_mean_loss if last_mean_loss is not None else np.nan)
        history["acc"].append(acc)

        if print_every and (ep % print_every == 0):
            prefix = f"[{tag}] " if tag else ""
            print(f"{prefix}epoch {ep:02d}/{epochs} | loss={last_mean_loss:.4f} | acc={acc:.4f}")

    return model, last_mean_loss, history


def train_final_cnn(
    X: np.ndarray,                 # (N, C, T)
    y: np.ndarray,                 # (N,)
    batch_size: int = 64,
    epochs: int = 20,
    lr: float = 1e-3,
    device: str | None = None,
    print_every: int = 1,
):
    """
    Train FINAL CNN pe TOATE datele (fără test).
    Folosește fix aceeași logică de train ca în LOSO (pos_weight + Adam + BCEWithLogits).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # sanity check (ca să nu mai crape în DataLoader)
    X = np.asarray(X)
    y = np.asarray(y)
    if X.shape[0] != len(y):
        raise ValueError(f"[train_final_cnn] Mismatch: X={X.shape[0]} vs y={len(y)}")

    ds = CNPTensorDataset(X, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    # pos_weight calculat pe TOT setul (exact ca în run_loso_cnn pe train)
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    pos_weight = torch.tensor([n_neg / (n_pos + 1e-8)], dtype=torch.float32, device=device)

    t0 = time.perf_counter()

    model, last_loss, history = train_one_cnn(
        train_loader=loader,
        in_channels=X.shape[1],
        epochs=epochs,
        lr=lr,
        device=device,
        pos_weight=pos_weight,
        print_every=print_every,
        tag="FINAL CNN",
        threshold=0.5,
    )

    sec = time.perf_counter() - t0
    print(f"[FINAL CNN] done in {sec:.1f}s ({sec/60:.1f} min) | last_loss={last_loss:.4f} | ")

    return model, history

@torch.no_grad()
def eval_cnn_simple(
    model: torch.nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 256,
    device: str | None = None,
    threshold: float = 0.5,
 

):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval().to(device)

    ds = CNPTensorDataset(X_test, y_test)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    probs = []
    ys = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb).squeeze(-1)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        probs.append(p)
        ys.append(yb.detach().cpu().numpy())

    y_prob = np.concatenate(probs)
    y_true = np.concatenate(ys).astype(int)
    y_pred = (y_prob >= threshold).astype(int)
    return y_true, y_pred, y_prob


def train_subject_holdout_cnn(
    X: np.ndarray,          # (N, C, T)
    y: np.ndarray,          # (N,)
    groups: np.ndarray,     # (N,) ex: "S2","S3",...
    seed: int = 123,
    test_subject: str | None = None,   # dacă vrei fix Sxx
    threshold: float = 0.5,
    batch_size: int = 64,
    epochs: int = 20,
    lr: float = 1e-3,
    device: str | None = None,
    print_every: int = 1,
    save_path: str | None = None,
):
    """
    1) Alege 1 subiect pt test (random sau fix).
    2) Train CNN pe restul subiecților.
    3) Testează pe subiectul ales.
    4) (Opțional) Salvează modelul.
    Returnează dict cu model + rezultate.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)

    if X.shape[0] != len(y) or len(y) != len(groups):
        raise ValueError(f"Mismatch: X={X.shape[0]} y={len(y)} groups={len(groups)}")

    test_subj, tr_mask, te_mask = split_by_subject(groups, seed=seed, test_subject=test_subject)

    X_tr, y_tr = X[tr_mask], y[tr_mask]
    X_te, y_te = X[te_mask], y[te_mask]

    print(f"[CNN] test_subject={test_subj} | train={len(y_tr)} | test={len(y_te)}")

    # train pe TRAIN subset (14 subiecți)
    model, history = train_final_cnn(
        X_tr, y_tr,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        device=device,
        print_every=print_every,
    )

    # eval pe TEST subject
    y_true, y_pred, y_prob = eval_cnn_simple(
        model,
        X_te, y_te,
        batch_size=256,
        device=device,
        threshold=threshold,
       
    )

    metrics = compute_classification_metrics(y_true, y_pred)
    cm = compute_confusion_matrix(y_true, y_pred)

    print(
            f'[CNN] Metrics: '
            f"Recall_stress={metrics['recall_stress']:.3f} | "
            f"F1_stress={metrics['f1_stress']:.3f} | "
            f"BalAcc={metrics['balanced_accuracy']:.3f} | "
            f"Acc={metrics['accuracy']:.3f}"
    )
    print(f"[CNN] confusion_matrix:\n{cm['matrix']}")

    # save opțional
    if save_path:
        save_cnn_model(
            model=model,
            in_channels=int(X.shape[1]),
            path=save_path,
            extra={
                "seed": seed,
                "test_subject": test_subj,
                "threshold": float(threshold),
                "train_samples": int(len(y_tr)),
                "test_samples": int(len(y_te)),
            },
            
        )
        print(f"[ CNN] saved -> {save_path}")

    return {
        "model": model,
        "test_subject": test_subj,
        "metrics": metrics,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "history": history,
        
    }

# -------------------------
# CNN (PyTorch) - SAVE / LOAD
# -------------------------
def save_cnn_model(
    model: torch.nn.Module,
    in_channels: int,
    path: str = "modele/saved_models/cnn_stress_final.pth",
    extra: dict[str, Any] | None = None,
):
    """
    Salvează doar weights (state_dict) + config minim.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "model_cfg": {"in_channels": int(in_channels)},
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_cnn_model(path: str, device: str | None = None):
    """
    Reface arhitectura CNN și încarcă weights.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(path, map_location=device)
    in_channels = ckpt["model_cfg"]["in_channels"]

    model = CNN1D(in_channels=in_channels).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt.get("extra", {})


# -------------------------
# RF (scikit-learn) - TRAIN FINAL
# -------------------------
def train_final_rf(
    X,                             # pd.DataFrame sau np.ndarray (N, F)
    y: np.ndarray,                 # (N,)
    n_estimators: int = 500,
    max_depth=None,
    min_samples_leaf: int = 1,
    class_weight: str | dict | None = "balanced",
    random_state: int = 42,
    use_scaler: bool = True,
):
    """
    Train RF pe TOATE datele. Returnează:
      (model_or_pipeline, feature_cols_or_None)
    Dacă use_scaler=True -> model este Pipeline(scaler + rf).
    """
    # păstrează ordinea feature-urilor
    feature_cols = getattr(X, "columns", None)
    X_arr = X.values if feature_cols is not None else np.asarray(X)
    y_arr = np.asarray(y).astype(int)

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )

    if use_scaler:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", rf),
        ])
        model.fit(X_arr, y_arr)
        return model, (list(feature_cols) if feature_cols is not None else None)
    else:
        rf.fit(X_arr, y_arr)
        return rf, (list(feature_cols) if feature_cols is not None else None)


# -------------------------
# RF (scikit-learn) - SAVE / LOAD
# -------------------------
def save_rf_model(
    model,
    feature_cols=None,
    path: str = "modele/saved_models/rf_stress_final.pkl",
    extra: dict[str, Any] | None = None,
):
    """
    Salvează obiectul sklearn (model sau Pipeline) cu joblib/pickle.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({
        "model": model,
        "feature_cols": feature_cols,
        "extra": extra or {},
    }, path)


def load_rf_model(path: str):
    bundle = joblib.load(path)
    return bundle["model"], bundle.get("feature_cols", None), bundle.get("extra", {})


def train_subject_holdout_rf(
    X,                      # pd.DataFrame sau np.ndarray (N, F)
    y: np.ndarray,          # (N,)
    groups: np.ndarray,     # (N,) ex: "S2","S3",...
    seed: int = 123,
    test_subject: str | None = None,
    # RF params
    n_estimators: int = 500,
    max_depth=None,
    min_samples_leaf: int = 1,
    class_weight: str | dict | None = "balanced",
    random_state: int = 42,
    use_scaler: bool = True,
    # save
    save_path: str | None = None,
):
    """
    1) Alege 1 subiect pt test (random sau fix).
    2) Train RF pe restul subiecților.
    3) Testează pe subiectul ales.
    4) (Opțional) Salvează modelul.
    Returnează dict cu model + rezultate.
    """
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)

    if len(y) != len(groups):
        raise ValueError(f"Mismatch: y={len(y)} groups={len(groups)}")

    # suport atât DataFrame cât și np.ndarray
    n = len(y)
    if hasattr(X, "iloc"):
        if len(X) != n:
            raise ValueError(f"Mismatch: X={len(X)} vs y={n}")
    else:
        X = np.asarray(X)
        if X.shape[0] != n:
            raise ValueError(f"Mismatch: X={X.shape[0]} vs y={n}")

    test_subj, tr_mask, te_mask = split_by_subject(groups, seed=seed, test_subject=test_subject)

    X_tr = X.iloc[tr_mask] if hasattr(X, "iloc") else X[tr_mask]
    y_tr = y[tr_mask]
    X_te = X.iloc[te_mask] if hasattr(X, "iloc") else X[te_mask]
    y_te = y[te_mask]

    print(f"[ Random Forest] test_subject={test_subj} | train={len(y_tr)} | test={len(y_te)}")

    model, feat_cols = train_final_rf(
        X_tr, y_tr,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        use_scaler=use_scaler,
    )

    y_pred = model.predict(X_te)

    cm =compute_confusion_matrix (y_te, y_pred)
    metrics = compute_classification_metrics(y_te, y_pred)

    print(f"[Random Forest] confusion_matrix:\n{cm['matrix']}")
    print(
            f'[Random Forest] Metrics: '
            f"Recall_stress={metrics['recall_stress']:.3f} | "
            f"F1_stress={metrics['f1_stress']:.3f} | "
            f"BalAcc={metrics['balanced_accuracy']:.3f} | "
            f"Acc={metrics['accuracy']:.3f}"
    )

    if save_path:
        save_rf_model(
            model=model,
            feature_cols=feat_cols,
            path=save_path,
            extra={
                "seed": seed,
                "test_subject": test_subj,
                "train_samples": int(len(y_tr)),
                "test_samples": int(len(y_te)),
                "rf_params": {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf,
                    "class_weight": class_weight,
                    "random_state": random_state,
                    "use_scaler": use_scaler,
                },
            },
        )
        print(f"[Random Forest] saved -> {save_path}")

    return {
        "model": model,
        "feature_cols": feat_cols,
        "test_subject": test_subj,
        "confusion_matrix": cm,
        "metrics": metrics,
        "y_true": y_te,
        "y_pred": y_pred,
        "X_test": X_te,
    }