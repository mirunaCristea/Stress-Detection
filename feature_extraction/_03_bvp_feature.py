import neurokit2 as nk
import numpy as np
# bvp_filt = semnalul tău BVP filtrat (ex: bandpass 0.5–5 Hz)
# fs = 64 pentru Empatica E4


def _safe_stats(x):
    """Întoarce mean, median, min, max, std pentru un vector 1D."""
    x = np.asarray(x, dtype=float)
    if x.size == 0 or np.all(~np.isfinite(x)):
        return 0.0, 0.0, 0.0, 0.0, 0.0

    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    mean = float(np.mean(x))
    median = float(np.median(x))
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    std = float(np.std(x))
    return mean, median, xmin, xmax, std

def _approx_mode(x, bins=50):
    """Mod aproximat cu histogramă (ca în unele articole PPG)."""
    x = np.asarray(x, dtype=float)
    if x.size == 0 or np.all(~np.isfinite(x)):
        return 0.0
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0

    hist, edges = np.histogram(x, bins=bins)
    idx = int(np.argmax(hist))
    # centrul bin-ului
    mode_val = 0.5 * (edges[idx] + edges[idx + 1])
    return float(mode_val)

def extract_bvp_features(bvp_segment, fs=64):
    """
    Extrage feature-uri HR și HRV dintr-un segment BVP.
    Folosește doar time/frequency HRV (fără complexitate),
    ca să evităm warning-urile din NeuroKit.
    """

    bvp_segment = np.asarray(bvp_segment, dtype=float).reshape(-1)  

    # fallback default
    empty = {
        "bvp_hr_mean": 0.0, # pulsul mediu, crestere posibila stres UTIL
        "bvp_hr_std": 0.0, # variabilitatea pulsului; stres duce la scadere
        "bvp_mean_pp": 0.0,# durata medie a intervaluui dintre batai; stres dupa la PPI micșor
        "bvp_std_pp": 0.0, # variabilitatea intervalului dintre bătăi; stres duce la scadere UTIL
        "bvp_mean": 0.0, # amplitudinea medie a semnalului BVP; vasoconstricția reduce amplitudinea; stres scade amplitudinea UTIL
        "bvp_median": 0.0, # amplitudinea mediană a semnalului BVP; buna pt ferestre cu artefacte
        "bvp_mode": 0.0, #valoarea cea mai frecventa  a amplitudinii;
        "bvp_min": 0.0, #minimul amplitudinii
        "bvp_max": 0.0, #varfuri cele mai inalte
        "bvp_std": 0.0, # variatia amplitudinii UTIL
        "bvp_d1_mean": 0.0, # media primei derivate (viteza schimbarii semnalului); stres mareste viteza
        "bvp_d1_std": 0.0, # deviația standard a primei derivate UTIL
        "bvp_d2_mean": 0.0, # media celei de-a doua derivate (accelerația schimbarii semnalului); stres mareste acceleratia
        "bvp_d2_std": 0.0, # deviația standard a celei de-a doua derivate UTIL
    }

    # dacă fereastra e foarte scurtă sau constantă, ieșim direct
    if bvp_segment.size < int(2 * fs) or np.std(bvp_segment) < 1e-6:
        return empty

    try:
        # 1) Procesare PPG cu NeuroKit
        cleaned = nk.ppg_clean(bvp_segment, sampling_rate=fs, method="elgendi")
        peaks_info = nk.ppg_findpeaks(cleaned, sampling_rate=fs)
        peaks =  peaks_info["PPG_Peaks"]
        # HR din PPG_Rate (le folosim mereu, chiar dacă HRV pică)

        if peaks is None:
            return empty
        peaks = np.array(peaks, dtype=int).reshape(-1)

        # dacă avem foarte puține bătăi -> nu facem HRV
        if peaks.size < 3:
            return empty
        
        rr = np.diff(peaks) / float(fs)
 
        rr = rr[(rr >= 0.3) & (rr <= 2.5)]

        if rr.size <2:
            return empty

        # filtrăm RR ne-fiziologice (tachy/brady extreme sau artefact)

        mean_pp, meadian_pp, min_pp, max_pp, std_pp = _safe_stats(rr)


        hr_inst = 60.0 / rr
        hr_mean, hr_med, hr_min, hr_max, hr_std = _safe_stats(hr_inst)

        mean_bvp, median_bvp, min_bvp, max_bvp, std_bvp = _safe_stats(cleaned)
        mode_bvp = _approx_mode(cleaned, bins=50)

        # derivate
        d1 = np.diff(cleaned)*fs
        d2 = np.diff(d1)*fs

        d1_mean, d1_med, d1_min, d1_max, d1_std = _safe_stats(d1)
        d2_mean, d2_med, d2_min, d2_max, d2_std = _safe_stats(d2)

        features = {
            "bvp_hr_mean": hr_mean,
            "bvp_hr_std": hr_std,
            "bvp_mean_pp": mean_pp,
            "bvp_std_pp": std_pp,
            "bvp_mean": mean_bvp,
            "bvp_median": median_bvp,
            "bvp_mode": mode_bvp,   
            "bvp_min": min_bvp,
            "bvp_max": max_bvp,
            "bvp_std": std_bvp,
            "bvp_d1_mean": d1_mean,
            "bvp_d1_std": d1_std,
            "bvp_d2_mean": d2_mean,
            "bvp_d2_std": d2_std,
        }
        return features
    except Exception as e:
        # în caz de eroare neașteptată, returnăm goale
        return empty