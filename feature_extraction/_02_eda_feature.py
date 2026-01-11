import numpy as np
import neurokit2 as nk
import warnings
from neurokit2.misc import NeuroKitWarning

# NeuroKit poate emite warning-uri la ferestre scurte sau semnale “dificile”.
warnings.filterwarnings("ignore", category=NeuroKitWarning)


def extract_eda_features(eda_window: np.ndarray, fs: int = 4) -> dict:
    """
    Extrage STRICT feature-urile EDA folosite în proiect:
      - EDA_mean         : media semnalului EDA în fereastră
      - EDA_tonic_mean   : media componentei tonice (EDA_Tonic)
      - EDA_phasic_mean  : media componentei fazice (EDA_Phasic)
      - EDA_smna_mean    : media EDA netezit (moving average ~1 sec) - proxy tonic robust

    Returnează NaN dacă fereastra e prea scurtă / invalidă.
    """
    eda = np.asarray(eda_window, dtype=float).reshape(-1)

    empty = {
        "EDA_mean": np.nan,
        "EDA_tonic_mean": np.nan,
        "EDA_phasic_mean": np.nan,
        "EDA_smna_mean": np.nan,
    }

    if eda.size == 0 or np.all(~np.isfinite(eda)):
        return empty

    # curățare NaN
    eda = eda[np.isfinite(eda)]
    
     # Dacă e prea scurtă, decompoziția tonic/phasic nu e stabilă
    if eda.size < fs * 4:  # <4s → prea scurt pt decompoziție stabilă
        return empty

    try:
        # 1) Media globală EDA
        eda_mean = float(np.mean(eda))

        # 2) Decompoziție tonic / phasic
        signals, _ = nk.eda_process(eda, sampling_rate=fs)
        tonic_mean = float(np.mean(signals["EDA_Tonic"]))
        phasic_mean = float(np.mean(signals["EDA_Phasic"]))

        # 3) SMNA – moving average pe EDA (proxy tonic robust)
        # fereastră ~1 sec
        k = max(3, int(fs))
        kernel = np.ones(k) / k
        eda_smoothed = np.convolve(eda, kernel, mode="same")
        smna_mean = float(np.mean(eda_smoothed))

        return {
            "EDA_mean": eda_mean,
            "EDA_tonic_mean": tonic_mean,
            "EDA_phasic_mean": phasic_mean,
            "EDA_smna_mean": smna_mean,
        }

    except Exception:
        return empty
