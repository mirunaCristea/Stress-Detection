import numpy as np

def extract_eda_features(window):
    features = {
        'eda_mean': np.mean(window),
        'eda_std': np.std(window),
        'eda_max': np.max(window),
        'eda_min': np.min(window),
        'eda_range': np.ptp(window),
    }
    return features
