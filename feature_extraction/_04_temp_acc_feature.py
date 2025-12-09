import numpy as np


def extract_temp_features(window):

    """
    Extrage featurile relevante din TEMP:
    - media temperaturii (tendinta generala)
    - deviația standard (variabilitatea)
    - valoarea maximă, minima
    -panta trendului (scadere posibil stres)
    """
    # in cazul in care segmentele sunt goale
    if len(window) == 0:
        return {
            'temp_mean': np.nan,
            'temp_std': np.nan,
            'temp_max': np.nan,
            'temp_min': np.nan,
            'temp_range': np.nan,
        }
    temp = np.asarray(window, dtype=float)
    #calcul panta
    x = np.arange(len(temp))
    panta = np.polyfit(x, temp, 1)[0]

    features = {
        'temp_mean': np.mean(window),
        'temp_std': np.std(window),
        'temp_max': np.max(window),
        'temp_min': np.min(window),
        'temp_slope': panta,
    }
    return features

def extract_acc_features(window):
    """
    Extrage feature-uri din magnitudinea ACC:
    - mean: nivel general de activitate
    - std: mișcare/agitație (crescută = posibil stres)
    - range_robust: P95 - P5, evită outlieri
    - energy: suma pătratelor (activitate totală)
    """  

    if len(window) == 0:
        return {
            "acc_mean": np.nan, "acc_std": np.nan, 
            "acc_range_robust": np.nan, "acc_energy": np.nan,
        }
    acc = np.asarray(window, dtype=float)
    robust_range = np.percentile(acc, 95) - np.percentile(acc, 5)
    energy = np.sum(acc ** 2)
    features = {
        'acc_mean': np.mean(window),
        'acc_std': np.std(window),
        'acc_range_robust': robust_range,
        'acc_energy': energy,
    }
    return features

