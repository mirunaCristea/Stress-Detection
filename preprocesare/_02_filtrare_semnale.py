from scipy.signal import butter, sosfiltfilt, sosfilt
import numpy as np


def butter_filter(data, fs, low=None, high=None, order=4, btype='low'):
    x = np.asarray(data, dtype=float)
    if x.ndim != 1:
        x = x.reshape(-1)

    if btype == 'band':
        wn = [low / (0.5 * fs), high / (0.5 * fs)]
    elif btype == 'low':
        wn = high / (0.5 * fs)
    elif btype == 'high':
        wn = low / (0.5 * fs)
    else:
        raise ValueError("btype trebuie 'low'/'high'/'band'")

    sos = butter(order, wn, btype=btype, output='sos')

    if not np.isfinite(x).all():
        x = np.nan_to_num(x)

    n_sections = sos.shape[0]
    default_padlen = 3 * (n_sections * 2)

    if len(x) <= 1:
        return x.copy()

    padlen = default_padlen
    if len(x) <= default_padlen:
        padlen = max(1, len(x) - 1)

    if len(x) <= padlen:
        return sosfilt(sos, x)

    return sosfiltfilt(sos, x, padlen=padlen)


def filter_eda(eda, fs=4):
    return butter_filter(eda, fs, high=1, btype='low')


def filter_bvp(bvp, fs=64):
    return butter_filter(bvp, fs, low=0.5, high=6.0, btype='band')


def filter_temp(temp, fs=4):
    return butter_filter(temp, fs, high=0.1, btype='low')


def filter_acc(acc, fs=32):
    """
    ACC în WESAD este pe 3 axe (N,3).
    Filtrăm fiecare axă separat și păstrăm forma (N,3).
    """
    x = np.asarray(acc, dtype=float)

    if x.ndim == 2 and x.shape[0] == 3 and x.shape[1] != 3:
        x = x.T

    if x.ndim == 1:
        return butter_filter(x, fs, high=5.0, btype='low')

    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"ACC must have shape (N,3) or (3,N). Got {x.shape}")

    acc_f = np.zeros_like(x)
    for k in range(3):
        acc_f[:, k] = butter_filter(x[:, k], fs, high=5.0, btype='low')

    return acc_f
