from feature_extraction._02_eda_feature import extract_eda_features
from feature_extraction._03_bvp_feature import extract_bvp_features
from feature_extraction._04_temp_acc_feature import extract_temp_features, extract_acc_features
import numpy as np
def concatenate_features(window_eda, window_bvp, window_temp, window_acc, fs_temp=4, fs_acc=32, fs_bvp=64):
    """
    Concatenează toate feature-urile extrase din semnalele EDA, BVP, TEMP și ACC pentru o fereastră dată.

    Parametri:
    - window_eda: array-like, fereastra de semnal EDA.
    - window_bvp: array-like, fereastra de semnal BVP.
    - window_temp: array-like, fereastra de semnal TEMP.
    - window_acc: array-like, fereastra de semnal ACC.
    - fs_temp: float, frecvența de eșantionare pentru semnalul TEMP.
    - fs_acc: float, frecvența de eșantionare pentru semnalul ACC.

    Returnează:
    - features_concat: dict, dicționar cu toate caracteristicile concatenate.
    """
    n_windows = len(window_eda)
    features_list = []

    for i in range(n_windows):
        win_eda  = np.asarray(window_eda[i])
        win_bvp  = np.asarray(window_bvp[i])
        win_temp = np.asarray(window_temp[i])
        win_acc  = np.asarray(window_acc[i])

        feats = {}
        # în concatenarea de features, dacă acc_energy e foarte mare:


# Extrage toate feature-urile
        eda_feats = extract_eda_features(win_eda)
        

# Extragem HRV & HR
        bvp_feats = extract_bvp_features(win_bvp, fs=fs_bvp)
        
    

# Extragem TEMP
        temp_feats = extract_temp_features(win_temp)
        

# Extragem ACC
        acc_feats = extract_acc_features(win_acc)
        
        
        feats.update(eda_feats)
        feats.update(bvp_feats)
        feats.update(temp_feats)
        feats.update(acc_feats)


# Adăugăm bvp_feat după aplicarea regulii

        
        features_list.append(feats)

    return features_list