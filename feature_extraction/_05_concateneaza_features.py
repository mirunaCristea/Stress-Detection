from feature_extraction._02_eda_feature import extract_eda_features
from feature_extraction._03_bvp_feature import extract_bvp_features
from feature_extraction._04_temp_acc_feature import extract_temp_features, extract_acc_features
import numpy as np

def concatenate_features(
    window_eda,
    window_bvp,
    window_temp,
    window_acc,
    fs_bvp=64
):
    """
    Concatenează feature-urile EDA, BVP, TEMP, ACC pentru fiecare fereastră.

    Intrări:
      - window_eda/window_bvp/window_temp/window_acc: liste de ferestre (output din sliding_windows)
      - fs_bvp: frecvența BVP (Hz), necesară pentru extract_bvp_features

    Ieșire:
      - features_list: listă de dict-uri (câte un dict pentru fiecare fereastră)
    """
    n_windows = len(window_eda)
    features_list = []

    for i in range(n_windows):
        feats = {}

        # extragere feature-uri
        feats.update(extract_eda_features(np.asarray(window_eda[i])))
        feats.update(extract_bvp_features(np.asarray(window_bvp[i]), fs=fs_bvp))
        feats.update(extract_temp_features(np.asarray(window_temp[i])))
        feats.update(extract_acc_features(np.asarray(window_acc[i])))

        features_list.append(feats)

    return features_list
