import numpy as np
import neurokit2 as nk

def _safe_stats_1d(x):
    """Întoarce mean, std, min, max pentru un vector 1D."""
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0 or np.all(~np.isfinite(x)):
        return 0.0, 0.0, 0.0, 0.0
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return float(np.mean(x)), float(np.std(x)), float(np.min(x)), float(np.max(x))

def _peak_frequency(x, fs=64):
    """
    Frecvența cu putere maximă în spectru (ignoră DC).
    Returnează 0.0 dacă nu se poate calcula robust.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size < 8 or not np.isfinite(fs) or fs <= 0:
        return 0.0
    if np.all(~np.isfinite(x)):
        return 0.0

    x = np.nan_to_num(x, nan=float(np.nanmean(x)))
    x = x - np.mean(x)  # scoți componenta DC

    # fereastră Hann pentru scăderea scurgerilor spectrale
    win = np.hanning(len(x))
    X = np.fft.rfft(x * win)
    P = np.abs(X) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)

    if P.size <= 1:
        return 0.0

    # ignoră 0 Hz (DC)
    idx = int(np.argmax(P[1:]) + 1)
    return float(freqs[idx])

def extract_bvp_features(bvp_segment, fs=64):
    """
    Extrage STRICT feature-urile cerute:
      - BVP_mean
      - BVP_std
      - BVP_min
      - BVP_max
      - BVP_peak_freq
    """
    bvp_segment = np.asarray(bvp_segment, dtype=float).reshape(-1)

    empty = {
        "BVP_mean": 0.0,
        "BVP_std": 0.0,
        "BVP_min": 0.0,
        "BVP_max": 0.0,
        "BVP_peak_freq": 0.0,
    }

    # dacă fereastra e prea scurtă sau aproape constantă
    if bvp_segment.size < int(2 * fs) or float(np.nanstd(bvp_segment)) < 1e-6:
        return empty

    try:
        # Curățare PPG (BVP) — păstrezi metoda ta
        cleaned = nk.ppg_clean(bvp_segment, sampling_rate=fs, method="elgendi")

        mean_bvp, std_bvp, min_bvp, max_bvp = _safe_stats_1d(cleaned)
        peak_freq = _peak_frequency(cleaned, fs)

        return {
            "BVP_mean": mean_bvp,
            "BVP_std": std_bvp,
            "BVP_min": min_bvp,
            "BVP_max": max_bvp,
            "BVP_peak_freq": peak_freq,
        }

    except Exception:
        return empty
