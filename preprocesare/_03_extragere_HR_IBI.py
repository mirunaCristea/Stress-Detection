import numpy as np
from pathlib import Path
import pickle
from scipy.signal import find_peaks

from _01_incarcare_taiere import load_data_wesad, cut_30s
from _02_filtrare_semnale import butter_filter

FS_DICT = {
    "ACC": 32,
    "BVP": 64,
    "EDA": 4,
    "TEMP": 4,
}
FS_LABEL = 700
FS_BVP = FS_DICT["BVP"]  # frecvența de eșantionare pentru BVP


def extract_hr_ibi_from_bvp(bvp_raw, fs=FS_BVP, do_filter=True):
    """
    Extrage ritmul cardiac (HR) și intervalele R-R (IBI) din semnalul BVP.

    Parametri:
    - bvp_raw: array-like, semnalul BVP brut.
    - fs: float, frecvența de eșantionare a semnalului BVP.
    - do_filter: bool, dacă este True, aplică un filtru bandpass pe semnalul BVP înainte de detectarea vârfurilor.

    Returnează:
    - hr: array-like, ritmul cardiac în bătăi pe minut (BPM).
    - ibi: array-like, intervalele R-R în secunde.
    - peak_indices: array-like, indicii vârfurilor detectate în semnalul BVP.
    """
    bvp = np.asarray(bvp_raw, dtype=float).reshape(-1)   
    bvp = bvp - np.mean(bvp)
    bvp = bvp / np.std(bvp)



    # Filtrare bandpass (0.5-5 Hz)
    if do_filter:
        bvp = butter_filter(bvp, fs=fs, low=0.7, high=3, btype='band', order=4)

    # Detectare vârfuri
    # distanta minimă între vârfuri: 0.3s (200 BPM max)

    peak_indices, _ = find_peaks(bvp, distance=int(fs*0.4), prominence=0.5*np.std(bvp))  # minim 0.4s între vârfuri

    if len(peak_indices) < 2:
        # prea puține bătăi detectate
        return {
            "t_peaks": np.array([]),
            "peaks_idx": peak_indices,
            "t_ibi": np.array([]),
            "ibi": np.array([]),
            "hr": np.array([]),
        }

    # 3. Timpii vârfurilor
    t_peaks = peak_indices / fs      # în secunde

    # 4. IBI (secunde) = diferența dintre timpii vârfurilor consecutive
    ibi = np.diff(t_peaks)

    # 5. HR (bătăi/minut)
    hr = 60.0 / ibi

    # 6. Timp asociat fiecărui IBI/HR
    #    (la mijloc între cele două bătăi)
    t_ibi = t_peaks[:-1] + ibi / 2.0

    return {
        "t_peaks": t_peaks,
        "peaks_idx": peak_indices,
        "t_ibi": t_ibi,
        "ibi": ibi,
        "hr": hr,
    }



# def process_subject(subject_id):
#     pkl_path = WESAD_DIR / f"{subject_id}.pkl"
#     print(f"[*] Procesez {pkl_path}...")

#     # 1) încarci semnalele + label-urile
#     signals, labels = load_data_wesad(pkl_path)

#     # 2) tai primele 30s folosind funcția ta
#     cut_signals, cut_labels = cut_30s(signals, labels, FS_DICT, fs_label=FS_LABEL)

#     # 3) iei BVP-ul tăiat
#     bvp_cut = cut_signals["BVP"]

#     # 4) scoți HR + IBI din BVP tăiat
#     hr_ibi = extract_hr_ibi_from_bvp(bvp_cut, fs=FS_DICT["BVP"], do_filter=True)

#     # 5) salvezi rezultatul într-un fișier separat
#     out_data = {
#         "subject": subject_id,
#         "hr_ibi": hr_ibi,
#         "labels_cut": cut_labels,   # le ții aici, pot fi utile la features
#         "len_bvp": len(bvp_cut),
#     }

#     out_path = HR_OUT / f"{subject_id}_hr_ibi.pkl"
#     with open(out_path, "wb") as f:
#         pickle.dump(out_data, f)

#     print(f"[OK] Salvat → {out_path}")


# if __name__ == "__main__":
#     subjects = ["S2", "S3", "S4", "S5", "S6", "S7",
#                 "S8", "S9", "S10", "S11", "S13",
#                 "S14", "S15", "S16", "S17"]

#     for sid in subjects:
#         process_subject(sid)