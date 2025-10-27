import importlib.util

from preprocesare._01_incarcare_taiere import load_data_wesad, cut_30s
from vizualizari.plot_semnale import plot_raw_vs_filtered
from preprocesare._02_filtrare_semnale import butter_filter
import numpy as np

def ensure_1d(sig):
    """ACC Nx3 -> normă; altfel vector 1D curat."""
    sig = np.asarray(sig)
    if sig.ndim == 2 and sig.shape[1] == 3:
        return np.linalg.norm(sig, axis=1)
    return sig.reshape(-1)

FS_WRIST = {
    'ACC': 32,
    'BVP': 64,
    'EDA': 4,
    'TEMP': 4
}

signals, labels = load_data_wesad('data/WESAD/S2.pkl')
raw, labels = cut_30s(signals, labels, FS_WRIST)

# 2) extrage semnalele
eda_raw  = np.asarray(raw.get('EDA', []))
bvp_raw  = np.asarray(raw.get('BVP', []))
temp_raw = np.asarray(raw.get('TEMP', []))
acc_raw  = ensure_1d(raw.get('ACC', []))

# 3) filtre
eda_f = butter_filter(eda_raw,  FS_WRIST['EDA'],  high=0.25, btype='low',  order=4)
bvp_f = butter_filter(bvp_raw,  FS_WRIST['BVP'],  low=0.5, high=5, btype='band', order=4)
temp_f= butter_filter(temp_raw, FS_WRIST['TEMP'], high=0.2,  btype='low',  order=4)
acc_f = butter_filter(acc_raw,  FS_WRIST['ACC'],  high=5,    btype='low',  order=4)

plot_raw_vs_filtered(eda_raw, eda_f, FS_WRIST['EDA'], title='Comparatie semnale brute si filtrate', unit='g', labels=labels, fs_labels=700, show_labels=True)