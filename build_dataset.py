# dataset/build_dataset_all.py
import os
import glob
import numpy as np
import pandas as pd

from preprocesare._01_incarcare_taiere import load_data_wesad, cut_30s, map_labels_to_binary
from preprocesare._02_filtrare_semnale import filter_eda, filter_bvp, filter_acc, filter_temp
from feature_extraction._01_ferestre_feature import sliding_windows, filter_windows_by_acc
from feature_extraction._05_concateneaza_features import concatenate_features


FS = {"ACC": 32, "BVP": 64, "EDA": 4, "TEMP": 4}
FS_LABEL = 700

WINDOW_SIZE = 60
STEP_SIZE = 5

ACC_STD_PERCENTILE = 98   # ca în main-ul tău (poți pune 90/85)
USE_ACC_FILTER = True


def _acc_std_per_window(acc_window):
    """std pe magnitudinea ACC (din fereastră), robust la (N,3) / (3,N)."""
    a = np.asarray(acc_window, dtype=float)
    if a.ndim == 2 and a.shape[0] == 3 and a.shape[1] != 3:
        a = a.T
    if a.ndim == 2 and a.shape[1] == 3:
        mag = np.sqrt(np.sum(a**2, axis=1))
    else:
        mag = np.abs(a).reshape(-1)
    return float(np.std(mag))


def build_subject_features(subject_path: str):
    """
    Returnează (X_df, y, groups) pentru un singur subiect.
    """
    sid = os.path.splitext(os.path.basename(subject_path))[0]  # ex "S2"

    # 1) load + labels binary
    signals, labels = load_data_wesad(subject_path)
    labels_binary = np.array([map_labels_to_binary(L) for L in labels])

    # 2) cut 30s
    cut_signals, cut_labels = cut_30s(signals, labels_binary, FS, fs_label=FS_LABEL)

    # 3) filtrare semnale
    eda_filt  = cut_signals["EDA"]
    bvp_filt  = filter_bvp(cut_signals["BVP"], fs=FS["BVP"])
    temp_filt = filter_temp(cut_signals["TEMP"], fs=FS["TEMP"])
    acc_filt  = filter_acc(cut_signals["ACC"], fs=FS["ACC"])  # trebuie să rămână (N,3)

    # 4) ferestre
    windows_eda, windows_bvp, windows_temp, windows_acc, labels_list = sliding_windows(
        signals={"EDA": eda_filt, "BVP": bvp_filt, "TEMP": temp_filt, "ACC": acc_filt},
        labels=np.asarray(cut_labels),
        fs_dict=FS,
        fs_label=FS_LABEL,
        window_s=WINDOW_SIZE,
        step_s=STEP_SIZE,
    )

    # 5) extragere feature-uri
    feats = concatenate_features(
        window_eda=windows_eda,
        window_bvp=windows_bvp,
        window_temp=windows_temp,
        window_acc=windows_acc,
        fs_bvp=FS["BVP"],
    )

    # 6) filtrare pe ACC (per subiect) + scoatere label -1
    if USE_ACC_FILTER and len(windows_acc) > 0:
        acc_std_list = np.array([_acc_std_per_window(w) for w in windows_acc])
        thr = float(np.percentile(acc_std_list, ACC_STD_PERCENTILE))

        feats_clean, labels_clean, _ = filter_windows_by_acc(
            windows_acc, feats, labels_list, std_threshold=thr
        )
    else:
        feats_clean, labels_clean = feats, labels_list

    X_df = pd.DataFrame(feats_clean)
    y = np.asarray(labels_clean)

    # scoți unknown
    mask = (y != -1)
    X_df = X_df.loc[mask].reset_index(drop=True)
    y = y[mask]

    groups = np.array([sid] * len(y))
    return X_df, y, groups


def build_full_dataset(wesad_dir="data/WESAD"):
    """
    Rulează pentru toți subiecții (*.pkl) și întoarce X, y, groups.
    """
    paths = sorted(glob.glob(os.path.join(wesad_dir, "*.pkl")))
    if len(paths) == 0:
        raise FileNotFoundError(f"Nu am găsit .pkl în {wesad_dir}")

    X_all = []
    y_all = []
    g_all = []

    for p in paths:
        sid = os.path.splitext(os.path.basename(p))[0]
        print(f"[SUBJECT] {sid} ...")

        Xs, ys, gs = build_subject_features(p)

        print(f"  windows kept: {len(ys)} | features: {Xs.shape[1] if len(ys)>0 else 0}")
        if len(ys) == 0:
            print("  [skip] zero ferestre după filtrare/label.")
            continue

        X_all.append(Xs)
        y_all.append(ys)
        g_all.append(gs)

    X = pd.concat(X_all, axis=0, ignore_index=True)
    y = np.concatenate(y_all)
    groups = np.concatenate(g_all)

    return X, y, groups



def save_feature_dataset(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, out_path="data/features/all_features.csv"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    X.to_csv(out_path, index=False)
    np.save(out_path.replace(".csv", "_y.npy"), y)
    np.save(out_path.replace(".csv", "_groups.npy"), groups)
    print("[OK] Salvat CSV+NPY:", out_path)

def load_feature_dataset(path="data/features/all_features.csv"):
    X = pd.read_csv(path)
    y = np.load(path.replace(".csv", "_y.npy"), allow_pickle=True)
    groups = np.load(path.replace(".csv", "_groups.npy"), allow_pickle=True)
    return X, y, groups