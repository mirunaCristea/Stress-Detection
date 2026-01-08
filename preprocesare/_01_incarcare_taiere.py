import pickle
import numpy as np


def load_data_wesad(file_path):
    """
    Deschide fișierul .pkl și returnează semnalele de la brățară + etichetele.
    Normalizează formele: EDA/BVP/TEMP -> 1D, ACC -> (N,3)
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    signals = data["signal"]["wrist"]
    labels = np.asarray(data["label"], dtype=np.int32).reshape(-1)

    fixed = {}
    for name, sig in signals.items():
        sig = np.asarray(sig)
        if name == "ACC":
            # ACC: (N,3)
            if sig.ndim != 2 or sig.shape[1] != 3:
                raise ValueError(f"ACC are formă neașteptată: {sig.shape}")
            fixed["ACC"] = sig.astype(np.float32)
        else:
            # EDA/BVP/TEMP: (N,) sau (N,1) -> (N,)
            fixed[name] = sig.reshape(-1).astype(np.float32)

    return fixed, labels


def cut_30s(signals, labels, fs_dict, fs_label=700, cut_s=30):
    """
    Eliminăm primele cut_s secunde din fiecare semnal și din etichete.
    """
    cut_signals = {}

    for name, signal in signals.items():
        if name not in fs_dict:
            raise KeyError(f"Lipsește fs pentru semnalul {name}. fs_dict={fs_dict}")

        fs = fs_dict[name]
        cut_index = int(fs * cut_s)

        if name == "ACC":
            cut_signals["ACC"] = signal[cut_index:, :]
        else:
            cut_signals[name] = signal[cut_index:]

    cut_labels = labels[int(fs_label * cut_s):]
    return cut_signals, cut_labels


def map_labels_to_binary_vector(labels):
    """
    Mapare vector etichete WESAD:
      2 -> 1 (stress)
      1,3,4 -> 0 (non-stress)
      altceva -> -1 (tranziție)
    """
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    out = np.full(labels.shape, -1, dtype=np.int8)

    out[labels == 2] = 1
    out[(labels == 1) | (labels == 3) | (labels == 4)] = 0
    return out
