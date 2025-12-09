# # import importlib.util

# # from preprocesare._01_incarcare_taiere import load_data_wesad, cut_30s
# # from vizualizari.plot_semnale import plot_raw_vs_filtered
# # from preprocesare._02_filtrare_semnale import butter_filter
# # import numpy as np

# # def ensure_1d(sig):
# #     """ACC Nx3 -> normă; altfel vector 1D curat."""
# #     sig = np.asarray(sig)
# #     if sig.ndim == 2 and sig.shape[1] == 3:
# #         return np.linalg.norm(sig, axis=1)
# #     return sig.reshape(-1)

# # FS_WRIST = {
# #     'ACC': 32,
# #     'BVP': 64,
# #     'EDA': 4,
# #     'TEMP': 4
# # }

# # signals, labels = load_data_wesad('data/WESAD/S2.pkl')
# # raw, labels = cut_30s(signals, labels, FS_WRIST)

# # # 2) extrage semnalele
# # eda_raw  = np.asarray(raw.get('EDA', []))
# # bvp_raw  = np.asarray(raw.get('BVP', []))
# # temp_raw = np.asarray(raw.get('TEMP', []))
# # acc_raw  = ensure_1d(raw.get('ACC', []))

# # # 3) filtre
# # eda_scl = butter_filter(eda_raw,  FS_WRIST['EDA'],  high=0.05, btype='low',  order=4)
# # eda_scr = butter_filter(eda_raw,  FS_WRIST['EDA'],  low=0.05, high=0.25,btype='band', order=4)
# # # bvp_f = butter_filter(bvp_raw,  FS_WRIST['BVP'],  low=0.5, high=5, btype='band', order=4)
# # # temp_f= butter_filter(temp_raw, FS_WRIST['TEMP'], high=0.2,  btype='low',  order=4)
# # # acc_f = butter_filter(acc_raw,  FS_WRIST['ACC'],  high=5,    btype='low',  order=4)

# # plot_raw_vs_filtered(eda_raw, eda_scl, FS_WRIST['EDA'], title='Componenta tonică (SCL) vs semnalul brut', unit='uS', labels=labels, fs_labels=700, show_labels=True)
# # plot_raw_vs_filtered(eda_raw, eda_scr, FS_WRIST['EDA'], title='Componenta fazică (SCR) vs semnalul brut', unit='uS', labels=labels, fs_labels=700, show_labels=True)
# # # plot_raw_vs_filtered(bvp_raw, bvp_f, FS_WRIST['BVP'], title='Comparatie semnale brute si filtrate', unit='', labels=labels, fs_labels=700, show_labels=True)
# # # plot_raw_vs_filtered(temp_raw, temp_f, FS_WRIST['TEMP'], title='Comparatie semnale brute si filtrate', unit='°C', labels=labels, fs_labels=700, show_labels=True)
# # # plot_raw_vs_filtered(acc_raw, acc_f, FS_WRIST['ACC'], title='Comparatie semnale brute si filtrate', unit='1/64g', labels=labels, fs_labels=700, show_labels=True)


# import pickle
# import numpy as np
# import matplotlib.pyplot as plt

# from preprocesare._02_filtrare_semnale import butter_filter

# # === 1) Încarci hr_ibi salvat ===
# with open("data/HR_IBI/S2_preproc.pkl", "rb") as f:
#     hr_ibi = pickle.load(f)

# peaks_idx = hr_ibi["peaks_idx"]

# # === 2) Încarci BVP-ul original ===
# with open("data/WESAD/S2.pkl", "rb") as f:
#     data = pickle.load(f, encoding="latin1")

# bvp_raw = np.asarray(data["signal"]["wrist"]["BVP"], dtype=float)

# # === 3) Tai primele 30s (cum faci în preprocesare) ===
# fs = 64
# bvp_raw = bvp_raw[30*fs:]



# # === 4) Filtrare BVP pentru vizualizare ===
# bvp_filt = butter_filter(bvp_raw, fs, low=0.5, high=5, btype="band")

# # === 5) GRAFIC ===
# plt.figure(figsize=(14,4))
# plt.plot(bvp_filt, label="BVP filtrat")
# plt.plot(peaks_idx, bvp_filt[peaks_idx], "ro", markersize=4, label="Bătăi detectate")
# plt.title("Validare HR/IBI — Detectare bătăi în BVP")
# plt.legend()
# plt.show()

# plt.hist(hr_ibi["ibi"], bins=40)
# plt.title("Distribuția intervalelor R-R (IBI)")
# plt.xlabel("IBI (secunde)")
# plt.ylabel("Frecvență")
# plt.show()

# print(f"Ritmul cardiac mediu: {np.mean(hr_ibi['hr']):.2f} BPM")

# fs = 64
# N = len(bvp_raw)          # după cut_30s
# T_sec = N / fs            # durata în secunde
# n_beats = len(hr_ibi["peaks_idx"])

# hr_global = 60 * n_beats / T_sec   # BPM mediu

# print("HR global (din număr bătăi):", hr_global)

# hr = hr_ibi["hr"]
# # curățare outlieri IBI
# ibi = hr_ibi["ibi"]

# mask = (ibi > 0.3) & (ibi < 2.0)   # interval fiziologic ~ 30–200 BPM
# ibi_clean = ibi[mask]
# hr_clean = 60 / ibi_clean

# print("HR mean clean:", np.mean(hr_clean))
# print("HR min clean:", np.min(hr_clean))
# print("HR max clean:", np.max(hr_clean))
# print("Procent valori eliminate:", 100 * (1 - len(ibi_clean)/len(ibi)))


# print("HR mean:", np.mean(hr))
# print("HR min:", np.min(hr))
# print("HR max:", np.max(hr))
# print("IBI min:", np.min(ibi))
# print("IBI max:", np.max(ibi))

import matplotlib.pyplot as plt
from preprocesare._01_incarcare_taiere import load_data_wesad, cut_30s, map_labels_to_binary
from preprocesare._02_filtrare_semnale import filter_eda, filter_bvp, filter_acc, filter_temp

from feature_extraction._01_ferestre_feature import sliding_windows, compute_acc_magnitude, is_window_valid_acc, filter_windows_by_acc
from feature_extraction._05_concateneaza_features import concatenate_features
import numpy as np
import pandas as pd
import os
from vizualizari.plot_features_corr import plot_window, plot_bvp_peaks, plot_window_labels
FS = {
    "ACC": 32,
    "BVP": 64,
    "EDA": 4,
    "TEMP": 4,
}

signals, labels = load_data_wesad('data/WESAD/S2.pkl')
labels_binary = [map_labels_to_binary(L) for L in labels]

cut_signals, cut_labels = cut_30s(signals, labels_binary, FS, fs_label=700)

eda_filt = filter_eda(cut_signals['EDA'], fs=FS['EDA'])
bvp_filt = filter_bvp(cut_signals['BVP'], fs=FS['BVP'])
temp_filt = filter_temp(cut_signals['TEMP'], fs=FS['TEMP'])
acc_filt = filter_acc(cut_signals['ACC'], fs=FS['ACC'])


plt.figure(figsize=(14, 8))
plt.subplot(4, 1, 1)
plt.plot(eda_filt, label='EDA filtrat', color='red')    
plt.title('Semnal EDA filtrat')
plt.subplot(4, 1, 2)
plt.plot(bvp_filt, label='BVP filtrat', color='green')
plt.title('Semnal BVP filtrat')
plt.subplot(4, 1, 3)
plt.plot(temp_filt, label='TEMP filtrat', color='blue')
plt.title('Semnal TEMP filtrat')
plt.subplot(4, 1, 4)
plt.plot(acc_filt, label='ACC filtrat', color='orange')
plt.title('Semnal ACC filtrat')
plt.tight_layout()
plt.show()
WINDOW_SIZE = 30  # secunde
STEP_SIZE   = 5   # secunde

windows_eda, windows_bvp, windows_temp, windows_acc, labels_list = sliding_windows(
    signals={
        "EDA": eda_filt,
        "BVP": bvp_filt,
        "TEMP": temp_filt,
        "ACC": acc_filt,      # ACC încă pe 3 axe aici
    },
    labels=np.asarray(cut_labels),
    fs_dict=FS,
    fs_label=700,
    window_s=WINDOW_SIZE,
    step_s=STEP_SIZE
)

# === 1. Calculăm acc_std pentru fiecare fereastră ===
acc_std_list = []

for acc_w in windows_acc:
    acc_w = np.asarray(acc_w)

    # Magnitudine ACC
    if acc_w.ndim == 2 and acc_w.shape[1] == 3:
        acc_mag = np.sqrt(np.sum(acc_w**2, axis=1))
    else:
        # fallback safe
        acc_mag = np.abs(acc_w).reshape(-1)

    acc_std = np.std(acc_mag)
    acc_std_list.append(acc_std)

acc_std_list = np.array(acc_std_list)

print("Acc std shape:", acc_std_list.shape)
print("Acc std stats:")
print(pd.Series(acc_std_list).describe())

plt.figure(figsize=(8,5))
plt.hist(acc_std_list, bins=40, color='steelblue', alpha=0.7)
plt.xlabel("ACC std per fereastră")
plt.ylabel("Număr ferestre")
plt.title("Distribuția deviației standard a magnitudinii ACC")
plt.grid(True, alpha=0.3)
plt.show()

thr_85 = np.percentile(acc_std_list, 85)
thr_90 = np.percentile(acc_std_list, 90)
thr_98 = np.percentile(acc_std_list, 98)

print("\nPraguri candidate:")
print("Percentila 85:", thr_85)
print("Percentila 90:", thr_90)
print("Percentila 98:", thr_98)


print("Număr ferestre EDA:", len(windows_eda))
print("Număr ferestre BVP:", len(windows_bvp))  
print("Număr ferestre TEMP:", len(windows_temp))  
print("Număr ferestre ACC:", len(windows_acc))  

# 🧠 2. extragem feature-uri pentru TOATE ferestrele
feats = concatenate_features(
    window_eda=windows_eda,
    window_bvp=windows_bvp,
    window_temp=windows_temp,
    window_acc=windows_acc,
    fs_bvp=FS['BVP'],
)
print("A trecut")


# 🧠 2.1. filtrăm ferestrele cu mișcare mare (ACC)
# labels_list este lista de etichete pe fereastră (0/1/-1)
feats_clean, labels_clean, acc_clean = filter_windows_by_acc(
    windows_acc,
    feats,
    labels_list,
    std_threshold=thr_98 ,   # poți ajusta ulterior
)

print(f"Ferestre înainte filtrare: {len(feats)}")
print(f"Ferestre după filtrare ACC: {len(feats_clean)}")


df_feats = pd.DataFrame(feats_clean)

# 🧠 3. atașăm etichetele
assert len(df_feats) == len(labels_clean)
df_feats["label"] = labels_clean

print(df_feats.head())
print(df_feats["label"].value_counts())

print(df_feats["label"][df_feats["label"]==1].describe())
# df_feats["hrv_sdnn"]  = df_feats["hrv_sdnn"].clip(0, 300)
# df_feats["hrv_rmssd"] = df_feats["hrv_rmssd"].clip(0, 300)

#salvare csv valori features
os.makedirs("data/features", exist_ok=True)
df_feats.to_csv("data/features/S2_features_wesad.csv", index=False)




df_clean = df_feats[df_feats["label"] != -1]

features_to_compare = [ "bvp_hr_mean", "bvp_std_pp", "bvp_mean", "bvp_std","bvp_d1_std", "bvp_d2_std"]

index = [250, 500]

for i in index:
    plot_window(i, windows_eda, windows_bvp, windows_temp, windows_acc, labels_list)
    plot_bvp_peaks(i, windows_bvp, fs=FS['BVP'])

for feat in features_to_compare:
    plt.figure(figsize=(6,4))
    df_clean[df_clean["label"] == 0][feat].hist(alpha=0.6, label="non-stress")
    df_clean[df_clean["label"] == 1][feat].hist(alpha=0.6, label="stress")
    plt.title(f"Distribuția {feat} pe clase")
    plt.legend()
    plt.show()
