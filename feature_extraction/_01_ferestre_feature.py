import numpy as np

def sliding_windows(signals, labels, fs_dict, fs_label,window_s=30, step_s=5):


    """
    signals: dict cu chei 'EDA', 'BVP', 'TEMP', 'ACC'
             după cut_30s + filtrare
    labels:  vector 1D (după cut_30s + map_labels_to_binary)
    fs_dict: frecvențele pentru fiecare semnal
    fs_label: frecvența etichetelor (700 în WESAD)
    """

    fs_eda  = fs_dict["EDA"]
    fs_bvp  = fs_dict["BVP"]
    fs_temp = fs_dict["TEMP"]
    fs_acc  = fs_dict["ACC"]

    eda  = np.asarray(signals["EDA"],  dtype=float)
    bvp  = np.asarray(signals["BVP"],  dtype=float)
    temp = np.asarray(signals["TEMP"], dtype=float)
    acc3 = np.asarray(signals["ACC"],  dtype=float)  # (N,3)
    # 👇 Aici facem ACC robust:
    if acc3.ndim == 2:
        # forma (N, 3) -> calculăm magnitudinea pe axe
        acc_mag = np.linalg.norm(acc3, axis=1)
    elif acc3.ndim == 1:
        # deja 1D (de ex. dacă filter_acc a aplatizat axele)
        acc_mag = acc3
    else:
        raise ValueError(f"ACC are formă neașteptată: {acc3.shape}")
  

    # durata (în secunde) pentru fiecare
    dur_eda  = len(eda)   / fs_eda
    dur_bvp  = len(bvp)   / fs_bvp
    dur_temp = len(temp)  / fs_temp
    dur_acc  = len(acc_mag) / fs_acc
    dur_lab  = len(labels) / fs_label

    # luăm durata comună minimă
    total_sec = min(dur_eda, dur_bvp, dur_temp, dur_acc, dur_lab)

    windows_eda  = []
    windows_bvp  = []
    windows_temp = []
    windows_acc  = []
    labels_list  = []

    t = 0.0
    while t + window_s <= total_sec:
        t_start = t
        t_end   = t + window_s

        # indici pentru fiecare semnal
        i_eda_start  = int(t_start * fs_eda)
        i_eda_end    = int(t_end   * fs_eda)

        i_bvp_start  = int(t_start * fs_bvp)
        i_bvp_end    = int(t_end   * fs_bvp)

        i_temp_start = int(t_start * fs_temp)
        i_temp_end   = int(t_end   * fs_temp)

        i_acc_start  = int(t_start * fs_acc)
        i_acc_end    = int(t_end   * fs_acc)

        i_lab_start  = int(t_start * fs_label)
        i_lab_end    = int(t_end   * fs_label)

        # tăiem ferestrele
        win_eda  = eda[i_eda_start:i_eda_end]
        win_bvp  = bvp[i_bvp_start:i_bvp_end]
        win_temp = temp[i_temp_start:i_temp_end]
        win_acc  = acc3[i_acc_start:i_acc_end]
        win_lab  = labels[i_lab_start:i_lab_end]

        # majority vote pentru label
        vals, counts = np.unique(win_lab, return_counts=True)
        maj = int(vals[np.argmax(counts)])  # 0 sau 1

        windows_eda.append(win_eda)
        windows_bvp.append(win_bvp)
        windows_temp.append(win_temp)
        windows_acc.append(win_acc)
        labels_list.append(maj)

        t += step_s

    return windows_eda, windows_bvp, windows_temp, windows_acc, labels_list



def compute_acc_magnitude(acc_xyz):
    """
    acc_xyz: poate fi:
      - 1D (N,)  -> fie deja magnitudine, fie axele concatenate
      - 2D (N,3) -> x,y,z pe coloane
      - 2D (3,N) -> x,y,z pe linii

    Întoarcem un vector 1D cu magnitudinea pe fiecare eșantion.
    """
    acc_xyz = np.asarray(acc_xyz)

    # Caz 1: deja 1D (probabil magnitudine sau o singură axă)
    if acc_xyz.ndim == 1:
        # dacă pare să conțină axele concatenat (N*3), încercăm reshape
        if acc_xyz.size % 3 == 0:
            acc_xyz = acc_xyz.reshape(-1, 3)
            return np.sqrt(np.sum(acc_xyz**2, axis=1))
        # altfel îl tratăm ca magnitudine deja
        return np.abs(acc_xyz)

    # Caz 2: 2D, axele pe coloane (N,3)
    if acc_xyz.ndim == 2:
        if acc_xyz.shape[1] == 3:
            return np.sqrt(np.sum(acc_xyz**2, axis=1))
        # Caz 3: 2D, axele pe linii (3,N)
        if acc_xyz.shape[0] == 3:
            return np.sqrt(np.sum(acc_xyz**2, axis=0))

        # fallback: normă pe ultimul axis
        return np.linalg.norm(acc_xyz, axis=-1)

    # Dacă ajungem aici e ceva foarte dubios
    raise ValueError(f"Forma neașteptată pentru ACC: shape={acc_xyz.shape}")


def is_window_valid_acc(acc_window, std_threshold=0.6):
    """
    Returnează True dacă fereastra are mișcare acceptabilă.
    Considerăm fereastra invalidă dacă std(acc_mag) depășește pragul.
    """
    acc_mag = compute_acc_magnitude(acc_window)
    acc_std = np.std(acc_mag)

    return acc_std < std_threshold


def filter_windows_by_acc(windows_acc, windows_features, windows_labels,
                          std_threshold=0.6):
    """
    windows_acc: list of arrays (N_window_samples, 3)
    windows_features: list of feature vectors
    windows_labels: list of labels

    Returnează doar ferestrele valide.
    """
    valid_features = []
    valid_labels = []
    valid_acc = []

    for acc_w, feat_w, lab_w in zip(windows_acc, windows_features, windows_labels):
        if is_window_valid_acc(acc_w, std_threshold):
            valid_features.append(feat_w)
            valid_labels.append(lab_w)
            valid_acc.append(acc_w)

    return valid_features, valid_labels, valid_acc
