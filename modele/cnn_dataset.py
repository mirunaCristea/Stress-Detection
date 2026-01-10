# modele/cnn_dataset.py
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

from preprocesare._01_incarcare_taiere import load_data_wesad, cut_30s, map_labels_to_binary_vector
from preprocesare._02_filtrare_semnale import filter_eda, filter_bvp, filter_acc, filter_temp
from feature_extraction._01_ferestre_feature import sliding_windows


FS = {"ACC": 32, "BVP": 64, "EDA": 4, "TEMP": 4}
FS_LABEL = 700


def _acc_to_mag(acc_win: np.ndarray) -> np.ndarray:
    acc = np.asarray(acc_win, dtype=float)
    if acc.ndim == 2 and acc.shape[1] == 3:
        mag = np.sqrt((acc * acc).sum(axis=1))
    else:
        mag = acc.reshape(-1)
    return mag


def _resample_1d(x: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    """
    Resampling simplu fără scipy.
    - downsample dacă factor întreg (ex 64->32)
    - altfel interpolare liniară
    """
    x = np.asarray(x, dtype=float).reshape(-1)

    if fs_in == fs_out:
        return x

    # downsample cu factor întreg
    if fs_in % fs_out == 0:
        factor = fs_in // fs_out
        return x[::factor]

    # upsample / caz general: interpolare
    n_in = len(x)
    if n_in < 2:
        return np.zeros(int(np.round(n_in * fs_out / fs_in)), dtype=float)

    t_in = np.linspace(0, 1, n_in, endpoint=True)
    n_out = int(np.round(n_in * fs_out / fs_in))
    t_out = np.linspace(0, 1, n_out, endpoint=True)
    return np.interp(t_out, t_in, x)


def _zscore_per_channel(X_ct: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    X_ct shape: (C, T) -> normalizează pe fiecare canal
    """
    mu = X_ct.mean(axis=1, keepdims=True)
    sd = X_ct.std(axis=1, keepdims=True) + eps
    return (X_ct - mu) / sd


def build_cnn_dataset(
    wesad_dir: str,
    window_s: int = 30,
    step_s: int = 5,
    max_transition_ratio: float = 0.25,
    exclude_subjects=("S12",),
    target_fs: int = 32,
    channels=("EDA", "BVP", "TEMP", "ACC_MAG"),  # 4 canale by default
):
    """
    Construiește dataset pentru CNN:
      X: np.ndarray shape (N, C, T)
      y: np.ndarray shape (N,)
      groups: np.ndarray shape (N,) cu 'Sx'
    """

    pkl_files = sorted(glob.glob(os.path.join(wesad_dir, "S*.pkl")))
    if not pkl_files:
        raise RuntimeError(f"Nu găsesc fișiere .pkl în {wesad_dir}")

    X_all = []
    y_all = []
    groups_all = []

    for pkl_path in pkl_files:
        subj = os.path.basename(pkl_path).replace(".pkl", "")
        if subj in exclude_subjects:
            continue

        # 1) load + cut
        signals, labels = load_data_wesad(pkl_path)
        signals, labels = cut_30s(signals, labels, FS, fs_label=FS_LABEL, cut_s=30)

        # 2) filtrare (rămâne ca la tine)
        signals["EDA"] = filter_eda(signals["EDA"], fs=FS["EDA"])
        signals["BVP"] = filter_bvp(signals["BVP"], fs=FS["BVP"])
        signals["TEMP"] = filter_temp(signals["TEMP"], fs=FS["TEMP"])
        signals["ACC"] = filter_acc(signals["ACC"], fs=FS["ACC"])

        # 3) labels binare {0,1,-1}
        labels_bin = map_labels_to_binary_vector(labels)

        # 4) ferestre
        w_eda, w_bvp, w_temp, w_acc, y_win = sliding_windows(
            signals=signals,
            labels=labels_bin,
            fs_dict=FS,
            fs_label=FS_LABEL,
            window_s=window_s,
            step_s=step_s,
            max_transition_ratio=max_transition_ratio
        )

        if len(y_win) == 0:
            print(f"[INFO] {subj}: 0 ferestre.")
            continue

        # 5) construim tensor (C,T) per fereastră
        for i in range(len(y_win)):
            # semnale pe fereastră
            eda = np.asarray(w_eda[i]).reshape(-1)
            bvp = np.asarray(w_bvp[i]).reshape(-1)
            tmp = np.asarray(w_temp[i]).reshape(-1)
            acc_mag = _acc_to_mag(w_acc[i])

            # resample toate la target_fs
            eda_r = _resample_1d(eda, FS["EDA"], target_fs)
            bvp_r = _resample_1d(bvp, FS["BVP"], target_fs)
            tmp_r = _resample_1d(tmp, FS["TEMP"], target_fs)
            acc_r = _resample_1d(acc_mag, FS["ACC"], target_fs)

            # T ar trebui să fie ~ window_s * target_fs
            # aliniază la aceeași lungime (min) ca să fie stack corect
            T = min(len(eda_r), len(bvp_r), len(tmp_r), len(acc_r))
            if T < 10:
                continue  # fereastră prea scurtă/invalidă

            channel_map = {
                "EDA": eda_r[:T],
                "BVP": bvp_r[:T],
                "TEMP": tmp_r[:T],
                "ACC_MAG": acc_r[:T]
            }

            X_ct = np.stack([channel_map[ch] for ch in channels], axis=0)  # (C,T)
            X_ct = _zscore_per_channel(X_ct)  # normalizează safe, fără leakage

            X_all.append(X_ct.astype(np.float32))
            y_all.append(int(y_win[i]))
            groups_all.append(subj)

        print(f"[OK] {subj}: ferestre CNN={len(y_win)}")

    if not X_all:
        raise RuntimeError("Nu s-a generat niciun exemplu CNN.")

    X = np.stack(X_all, axis=0)  # (N,C,T)
    y = np.asarray(y_all, dtype=np.int64)
    groups = np.asarray(groups_all)

    # check
    u, c = np.unique(y, return_counts=True)
    print("\n=== CHECK CNN DATASET ===")
    print("X:", X.shape, "y:", y.shape, "groups:", len(set(groups)))
    print("Class counts:", dict(zip(u.tolist(), c.tolist())))

    return X, y, groups


class CNPTensorDataset(Dataset):
    """
    Dataset PyTorch pentru X: (N,C,T), y:(N,)
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()  # BCE loss cere float

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
