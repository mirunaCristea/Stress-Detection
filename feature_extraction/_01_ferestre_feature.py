import numpy as np


def sliding_windows(
    signals,
    labels,
    fs_dict,
    fs_label,
    window_s=30,
    step_s=5,
    max_transition_ratio=0.25
):
    """
    signals: dict cu chei 'EDA', 'BVP', 'TEMP', 'ACC' după cut + filtrare
    labels:  vector 1D binarizat {0,1,-1} după map_labels_to_binary_vector
    """

    fs_eda = fs_dict["EDA"]
    fs_bvp = fs_dict["BVP"]
    fs_temp = fs_dict["TEMP"]
    fs_acc = fs_dict["ACC"]

    eda = np.asarray(signals["EDA"], dtype=float).reshape(-1)
    bvp = np.asarray(signals["BVP"], dtype=float).reshape(-1)
    temp = np.asarray(signals["TEMP"], dtype=float).reshape(-1)
    acc = np.asarray(signals["ACC"], dtype=float)  # (N,3)
    labels = np.asarray(labels, dtype=int).reshape(-1)

    # durata minimă comună (secunde)
    total_sec = min(
        len(eda) / fs_eda,
        len(bvp) / fs_bvp,
        len(temp) / fs_temp,
        acc.shape[0] / fs_acc,
        len(labels) / fs_label
    )

    windows_eda, windows_bvp, windows_temp, windows_acc, labels_list = [], [], [], [], []

    t = 0.0
    while t + window_s <= total_sec:
        t_start, t_end = t, t + window_s

        i_eda_start, i_eda_end = int(t_start * fs_eda), int(t_end * fs_eda)
        i_bvp_start, i_bvp_end = int(t_start * fs_bvp), int(t_end * fs_bvp)
        i_temp_start, i_temp_end = int(t_start * fs_temp), int(t_end * fs_temp)
        i_acc_start, i_acc_end = int(t_start * fs_acc), int(t_end * fs_acc)
        i_lab_start, i_lab_end = int(t_start * fs_label), int(t_end * fs_label)

        win_eda = eda[i_eda_start:i_eda_end]
        win_bvp = bvp[i_bvp_start:i_bvp_end]
        win_temp = temp[i_temp_start:i_temp_end]
        win_acc = acc[i_acc_start:i_acc_end]
        win_lab = labels[i_lab_start:i_lab_end]

        # 1) dacă prea mult -1 în fereastră -> aruncăm
        trans_ratio = np.mean(win_lab == -1)
        if trans_ratio > max_transition_ratio:
            t += step_s
            continue

        # 2) label majoritar doar din {0,1} (ignorăm -1)
        valid = win_lab[win_lab != -1]
        if valid.size == 0:
            t += step_s
            continue

        zeros = np.sum(valid == 0)
        ones = np.sum(valid == 1)
        maj = 1 if ones > zeros else 0

        windows_eda.append(win_eda)
        windows_bvp.append(win_bvp)
        windows_temp.append(win_temp)
        windows_acc.append(win_acc)
        labels_list.append(int(maj))

        t += step_s

    return windows_eda, windows_bvp, windows_temp, windows_acc, labels_list
