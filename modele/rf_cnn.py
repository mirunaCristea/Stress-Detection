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


# -------------------------
# CNN (PyTorch) - TRAIN FINAL
# -------------------------
def train_one_cnn(
    train_loader: DataLoader,
    in_channels: int,
    epochs: int = 15,
    lr: float = 1e-3,
    device: str = "cpu",
    pos_weight: torch.Tensor | None = None,
    print_every: int = 0,
    tag: str = "",
):
    """
    Antrenează 1 CNN model pe train_loader.
    Returnează (model, last_epoch_loss_mean).
    """
    model = CNN1D(in_channels=in_channels).to(device)

    # default pos_weight = 1 dacă nu e dat
    if pos_weight is None:
        pos_weight = torch.tensor([1.0], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    last_mean_loss = None

    for ep in range(1, epochs + 1):
        losses = []
        f1_stress=[]
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optim.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optim.step()
         
            losses.append(float(loss.detach().cpu().item()))

        last_mean_loss = float(np.mean(losses)) if losses else None

        if print_every and (ep % print_every == 0):
            prefix = f"[{tag}] " if tag else ""
            print(f"{prefix}epoch {ep:02d}/{epochs} | loss={last_mean_loss:.4f}")

    return model, last_mean_loss

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

    model, last_loss = train_one_cnn(
        train_loader=loader,
        in_channels=X.shape[1],
        epochs=epochs,
        lr=lr,
        device=device,
        pos_weight=pos_weight,
        print_every=print_every,
        tag="FINAL CNN",
    )

    sec = time.perf_counter() - t0
    print(f"[FINAL CNN] done in {sec:.1f}s ({sec/60:.1f} min) | last_loss={last_loss:.4f} | ")

    return model

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
