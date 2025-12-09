from scipy.signal import butter, sosfiltfilt, sosfilt
import numpy as np

def butter_filter(data, fs, low=None, high=None, order=4, btype='low'):
    """
    Aplică un filtru Butterworth pe datele de intrare.

    Parametri:
    - data: array-like, semnalul de intrare care trebuie filtrat.
    - fs: float, frecvența de eșantionare a semnalului.
    - low: float sau None, frecvența de tăiere inferioară pentru un filtru trece-sus. Dacă este None, nu se aplică filtrare trece-sus.
    - high: float sau None, frecvența de tăiere superioară pentru un filtru trece-jos. Dacă este None, nu se aplică filtrare trece-jos.
    - order: int, ordinul filtrului Butterworth.

    Returnează:
    - filtered_data: array-like, semnalul filtrat.
    """
    x = np.asarray(data, dtype=float)
    if x.ndim != 1:
        x = x.reshape(-1)

    # filtru Butter -> SOS
    if btype == 'band':
        wn = [low/(0.5*fs), high/(0.5*fs)]
    elif btype == 'low':
        wn = high/(0.5*fs)
    elif btype == 'high':
        wn = low/(0.5*fs)
    else:
        raise ValueError("btype trebuie 'low'/'high'/'band'")

    sos = butter(order, wn, btype=btype, output='sos')

    # curățare NaN/Inf
    if not np.isfinite(x).all():
        x = np.nan_to_num(x)

    # calculează padlen „safe”
    n_sections = sos.shape[0]               # nr. de biquads
    default_padlen = 3 * (n_sections * 2)   # regula SciPy
    # dacă semnalul e prea scurt, micșorăm padlen
    if len(x) <= default_padlen:
        if len(x) <= 1:
            return x.copy()                 # nimic de filtrat
        padlen = max(1, len(x) - 1)         # <= len(x)-1, obligatoriu
    else:
        padlen = default_padlen

    # dacă după toate astea semnalul e tot mic, fallback pe sosfilt (cu lag)
    if len(x) <= padlen:
        return sosfilt(sos, x)

    return sosfiltfilt(sos, x, padlen=padlen)


def filter_eda(eda, fs=4):
    return butter_filter(eda, fs, high=1.0, btype='low')

def filter_bvp(bvp, fs=64):
    return butter_filter(bvp, fs, low=0.5, high=5.0, btype='band')

def filter_temp(temp, fs=4):
    return butter_filter(temp, fs, high=0.1, btype='low')

def filter_acc(acc, fs=32):
    return butter_filter(acc, fs, high=5.0, btype='low')