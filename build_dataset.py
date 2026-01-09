# build_dataset.py
import os
import glob
import numpy as np
import pandas as pd

from preprocesare._01_incarcare_taiere import load_data_wesad, cut_30s, map_labels_to_binary_vector
from preprocesare._02_filtrare_semnale import filter_eda, filter_bvp, filter_acc, filter_temp
from feature_extraction._01_ferestre_feature import sliding_windows
from feature_extraction._05_concateneaza_features import concatenate_features
# (sau cum se numește fișierul în care ai concatenate_features)



FS = {"ACC": 32, "BVP": 64, "EDA": 4, "TEMP": 4}
FS_LABEL = 700


# def _extract_features_from_window(win_eda, win_bvp, win_temp, win_acc):
#     """
#     Feature-uri simple (baseline) per fereastră.
#     Returnează un dict -> devine un rând în tabel.
#     """
#     win_eda = np.asarray(win_eda, dtype=float).reshape(-1)
#     win_bvp = np.asarray(win_bvp, dtype=float).reshape(-1)
#     win_temp = np.asarray(win_temp, dtype=float).reshape(-1)
#     win_acc = np.asarray(win_acc, dtype=float)  # (N,3)

#     # magnitudinea ACC
#     if win_acc.ndim == 2 and win_acc.shape[1] == 3:
#         acc_mag = np.sqrt(np.sum(win_acc * win_acc, axis=1))
#     else:
#         acc_mag = win_acc.reshape(-1)

#     def safe_stats(x):
#         x = x[np.isfinite(x)]
#         if x.size == 0:
#             return 0.0, 0.0, 0.0, 0.0
#         return float(np.mean(x)), float(np.std(x)), float(np.min(x)), float(np.max(x))

#     eda_mean, eda_std, eda_min, eda_max = safe_stats(win_eda)
#     bvp_mean, bvp_std, bvp_min, bvp_max = safe_stats(win_bvp)
#     temp_mean, temp_std, temp_min, temp_max = safe_stats(win_temp)
#     acc_mean, acc_std, acc_min, acc_max = safe_stats(acc_mag)

#     return {
#         "eda_mean": eda_mean, "eda_std": eda_std, "eda_min": eda_min, "eda_max": eda_max,
#         "bvp_mean": bvp_mean, "bvp_std": bvp_std, "bvp_min": bvp_min, "bvp_max": bvp_max,
#         "temp_mean": temp_mean, "temp_std": temp_std, "temp_min": temp_min, "temp_max": temp_max,
#         "acc_mean": acc_mean, "acc_std": acc_std, "acc_min": acc_min, "acc_max": acc_max,
#     }


def build_full_dataset(
    wesad_dir,
    window_s=60,
    step_s=5,
    max_transition_ratio=0.25,
    exclude_subjects=("S12",),
):
    """
    Construiește datasetul (feature-uri) pentru toți subiecții din folderul WESAD.
    Returnează:
      X: DataFrame (features)
      y: array (0/1)
      groups: array (ID subiect, pentru LOSO)
    """
    pkl_files = sorted(glob.glob(os.path.join(wesad_dir, "S*.pkl")))
    if not pkl_files:
        raise RuntimeError(f"Nu găsesc fișiere .pkl în {wesad_dir}")

    rows = []
    y_all = []
    groups = []

    for pkl_path in pkl_files:
        subj = os.path.basename(pkl_path).replace(".pkl", "")
        if subj in exclude_subjects:
            continue

        # 1) load + cut
        signals, labels = load_data_wesad(pkl_path)
        signals, labels = cut_30s(signals, labels, FS, fs_label=FS_LABEL, cut_s=30)

        # 2) filtrare semnale
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
            print(f"[INFO] {subj}: 0 ferestre (poate filtre prea stricte).")
            continue

        # 5) features per fereastră
        feat_rows = concatenate_features(
            window_eda=w_eda,
            window_bvp=w_bvp,
            window_temp=w_temp,
            window_acc=w_acc,
            fs_bvp=FS["BVP"]
        )
        if len(feat_rows) != len(y_win):
            raise RuntimeError(
                f"{subj}: mismatch ferestre: feat_rows={len(feat_rows)} vs y_win={len(y_win)}"
            )

# feat_rows are aceeași lungime ca y_win
        rows.extend(feat_rows)
        y_all.extend([int(v) for v in y_win])
        groups.extend([subj] * len(y_win))

        print(f"[OK] {subj}: ferestre={len(y_win)}")

    if not rows:
        raise RuntimeError("Nu s-a generat nicio fereastră / niciun feature. Verifică setările.")

    X = pd.DataFrame(rows)
    y = np.asarray(y_all, dtype=np.int8)
    groups = np.asarray(groups)

    # print de verificare (ca să vezi ce-mi ceri tu)
    u, c = np.unique(y, return_counts=True)
    print("\n=== CHECK DATASET ===")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Class counts:", dict(zip(u.tolist(), c.tolist())))

    return X, y, groups


def save_feature_dataset(X, y, groups, out_path1,out_path2):
    os.makedirs(os.path.dirname(out_path1), exist_ok=True)
    df = X.copy()
    df["y"] = y.astype(int)
    df["group"] = groups
    df.to_parquet(out_path1, index=False)
    print(f"[SALVAT] {out_path1}")
    df.to_csv(out_path2, index=False)
    print(f"[SALVAT] {out_path2}")


def load_feature_dataset(path):
    df = pd.read_parquet(path)
    y = df["y"].to_numpy(dtype=np.int8)
    groups = df["group"].to_numpy()
    X = df.drop(columns=["y", "group"])
    return X, y, groups
