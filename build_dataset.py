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



# def motion_score(acc_win):
#     """
#     Returnează un scor de mișcare pt o fereastră ACC.
#     Folosim std-ul magnitudinii (robust).
#     """
#     acc = np.asarray(acc_win, dtype=float)
#     if acc.ndim == 2 and acc.shape[1] == 3:
#         mag = np.sqrt((acc * acc).sum(axis=1))
#     else:
#         mag = acc.reshape(-1)

#     mag = mag[np.isfinite(mag)]
#     if mag.size == 0:
#         return 0.0
#     return float(np.std(mag))




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
        signals["EDA"] =signals["EDA"]
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

        # 4.5) Filtrare ferestre cu mișcare exagerată (artefacte)
        # w_eda, w_bvp, w_temp, w_acc, y_win = filter_noisy_windows_by_motion(
        #     w_eda, w_bvp, w_temp, w_acc, y_win,
        #     drop_top_pct=10,   # scoți top 10% cele mai agitate
        #     min_keep=50        # păstrezi măcar 50 ferestre/subiect
        # )
        
        # 5) features per fereastră
        feat_rows = concatenate_features(
            window_eda=w_eda,
            window_bvp=w_bvp,
            window_temp=w_temp,
            window_acc=w_acc,
            fs_bvp=FS["BVP"]
        )
  


        # OPTIONAL :
        # 5.1) scos ferestrele noisy cy ajutorul acceleratoeo
#########################################################       
        # keep_idx = []
        # for i in range(len(y_win)):
        #     if motion_score(w_acc[i]) < 0.30:   # prag de calibrat!
        #         keep_idx.append(i)

        # feat_rows = [feat_rows[i] for i in keep_idx]
        # y_win_kept = [y_win[i] for i in keep_idx]
        # y_win = y_win_kept
     ########################################################
     # 5.2) optional: scos feature-urile ACC din dataset
########################################################`
        # feat_rows_no_acc = []
        # for feats in feat_rows:
        #     feats_no_acc = {k: v for k, v in feats.items() if not k.startswith("ACC_")}
        #     feat_rows_no_acc.append(feats_no_acc)

        # feat_rows = feat_rows_no_acc
##########################################################        
# feat_rows are aceeași lungime ca y_win
        if len(feat_rows) != len(y_win):
            raise RuntimeError(
                f"{subj}: mismatch ferestre: feat_rows={len(feat_rows)} vs y_win={len(y_win)}"
            )
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
