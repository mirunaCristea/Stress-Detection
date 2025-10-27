# -*- coding: utf-8 -*-
"""
Vizualizare RAW + FFT (WESAD, brățară)
- Semnale brute (EDA, BVP, TEMP, ACC) cu fundal colorat pe etichete
- FFT pentru fiecare semnal
"""

from preprocesare._01_incarcare_taiere import load_data_wesad, cut_30s
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.patches as mpatches

# ---------------- CONFIG ----------------
SUBJECT_PATH = r"data/WESAD/S2.pkl"     # <- schimbă subiectul
OUT_DIR = Path("figs_raw_signals")      # unde salvăm imaginile
OUT_DIR.mkdir(exist_ok=True, parents=True)

FS_WRIST = {'BVP':64, 'EDA':4, 'TEMP':4, 'ACC':32}
FS_LABEL = 700
LABEL_NAMES = {0:'none', 1:'baseline', 2:'stress', 3:'amusement', 4:'meditation'}
LABEL_COLORS = {0:'lightgray', 1:'green', 2:'red', 3:'orange', 4:'blue'}  # culori soft

# -------------- HELPERI --------------
def savefig_nice(path, tight=True, dpi=180):
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"[SALVAT] {path}")
    plt.close()

def ensure_1d(sig):
    """ACC Nx3 -> normă; altfel vector 1D curat."""
    sig = np.asarray(sig)
    if sig.ndim == 2 and sig.shape[1] == 3:
        return np.linalg.norm(sig, axis=1)
    return sig.reshape(-1)

def mode_in_window(arr):
    if len(arr) == 0:
        return 0
    vals, counts = np.unique(arr.astype(int), return_counts=True)
    return vals[np.argmax(counts)]

def labels_to_signal_axis(lab_cut, fs_sig, n_sig, fs_label=FS_LABEL):
    """Mapează etichetele (700 Hz) la fiecare eșantion dintr-un semnal la fs_sig."""
    t_sig = np.arange(n_sig) / fs_sig
    lab_sig = np.zeros(n_sig, dtype=int)
    for i in range(n_sig):
        t0 = t_sig[i]
        t1 = t_sig[i+1] if i < n_sig-1 else t_sig[i] + (1.0/fs_sig)
        i0 = int(np.floor(t0 * fs_label))
        i1 = int(np.floor(t1 * fs_label))
        if i1 <= i0: i1 = i0 + 1
        i0 = max(i0, 0); i1 = min(i1, len(lab_cut))
        lab_sig[i] = mode_in_window(lab_cut[i0:i1])
    return lab_sig

def shade_by_labels( t, lab_sig):
    """Colorează fundalul axului în funcție de etichetele din lab_sig."""
    lab_sig = np.asarray(lab_sig).astype(int).reshape(-1)
    # găsește segmentele continue
    edges = np.flatnonzero(np.diff(lab_sig) != 0)
    starts = np.r_[0, edges+1]
    ends   = np.r_[edges, len(lab_sig)-1]
    for s, e in zip(starts, ends):
        v = lab_sig[s]
        if v not in LABEL_COLORS: 
            continue
        t0 = t[s]
        t1 = t[e] if e < len(t) else t[-1]
        plt.axvspan(t0, t1, color=LABEL_COLORS[v], alpha=0.25, lw=0)

def fft_plot(x, fs, title, fname, nfft=None):
    """|RFFT| vs frecvență (blindat pt. (N,1), NaN/Inf)."""
    x = np.asarray(x).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 8:
        print(f"[AVERTISMENT] {title}: prea puține puncte.")
        return
    x = x - np.mean(x)

    if nfft is None:
        n = x.size
        nfft = 1 << int(np.floor(np.log2(n)))
        nfft = int(min(max(nfft, 256), 65536))  # limită rezonabilă

    win = np.hanning(nfft)
    xw = x[:nfft] * win
    X = np.fft.rfft(xw)
    freqs = np.fft.rfftfreq(nfft, d=1.0/fs)
    mag = np.abs(X)

    plt.figure(figsize=(12,4))
    plt.plot(freqs, mag, lw=1.0, color='black')
    plt.xlim(0, fs/2)
    plt.xlabel("Frecvență [Hz]")
    plt.ylabel("|X(f)|")
    plt.title(title)
    plt.grid()
   # plt.show()
    
    savefig_nice(OUT_DIR/fname)


def add_label_legend(ax):
    """Afișează legenda în dreapta graficului, integrată elegant."""
    keys = [0, 1, 2, 3, 4]  # ordonare fixă
    handles = [
        mpatches.Patch(
            facecolor=LABEL_COLORS[k],
            edgecolor="none",
            alpha=0.4,
            label=f"{LABEL_NAMES[k].capitalize()}"
        )
        for k in keys if k in LABEL_COLORS
    ]

    ax.legend(
        handles=handles,
        loc="upper right",          # poziție laterală
         # ușor în afara axei
        frameon=True,              # fără chenar
        fontsize=10,
        title="Etichete",
        title_fontsize=11
    )



# -------------- ÎNCĂRCARE + TĂIERE 30s --------------
signals, labels = load_data_wesad(SUBJECT_PATH)
signals, labels = cut_30s(signals, labels, FS_WRIST, FS_LABEL)

# -------------- SEMNALE --------------
eda  = signals.get('EDA',  [])
bvp  = signals.get('BVP',  [])
temp = signals.get('TEMP', [])
acc  = ensure_1d(signals.get('ACC',  []))  # normă dacă a fost Nx3

t_eda  = np.arange(len(eda))  / FS_WRIST['EDA']
t_bvp  = np.arange(len(bvp))  / FS_WRIST['BVP']
t_temp = np.arange(len(temp)) / FS_WRIST['TEMP']
t_acc  = np.arange(len(acc))  / FS_WRIST['ACC']

# mapare etichete pe fiecare axă de timp (după tăiere)
lab_eda  = labels_to_signal_axis(labels, FS_WRIST['EDA'],  len(eda))
lab_bvp  = labels_to_signal_axis(labels, FS_WRIST['BVP'],  len(bvp))
lab_temp = labels_to_signal_axis(labels, FS_WRIST['TEMP'], len(temp))
lab_acc  = labels_to_signal_axis(labels, FS_WRIST['ACC'],  len(acc))

# -------------- PLOTURI RAW + LABELS --------------
plt.figure(figsize=(12,4))

shade_by_labels(t_eda, lab_eda)
plt.plot(t_eda, eda, lw=0.8,color='black')
plt.title("EDA – semnal brut (cu etichete)")
plt.ylabel("µS")
add_label_legend(plt.gca())
plt.grid()
savefig_nice(OUT_DIR/"EDA_raw.png")


plt.figure(figsize=(12,4))
shade_by_labels( t_bvp, lab_bvp)
plt.plot(t_bvp, bvp, lw=0.8,color='black')
plt.title("BVP – semnal brut (cu etichete)")
plt.ylabel("arb. u.")
add_label_legend(plt.gca())
plt.grid()
savefig_nice(OUT_DIR/"BVP_raw.png")


plt.figure(figsize=(12,4))
shade_by_labels( t_temp, lab_temp)
plt.plot(t_temp, temp, lw=0.8,color='black')
plt.title("TEMP – semnal brut (cu etichete)")
plt.ylabel("°C")
add_label_legend(plt.gca())
plt.grid()
savefig_nice(OUT_DIR/"TEMP_raw.png")

plt.figure(figsize=(12,4))
shade_by_labels(t_acc, lab_acc)
plt.plot(t_acc, acc, lw=0.8,color='black')
plt.title("ACC (normă) – semnal brut (cu etichete)")
plt.ylabel("1/64 g")
plt.xlabel("Timp [s]")
add_label_legend(plt.gca())
plt.grid()
savefig_nice(OUT_DIR/"ACC_raw.png")


# -------------- FFT (pe semnalele brute) --------------
fft_plot(eda,  FS_WRIST['EDA'],  "FFT EDA (brut)",   "FFT_EDA_raw.png")
fft_plot(bvp,  FS_WRIST['BVP'],  "FFT BVP (brut)",   "FFT_BVP_raw.png")
fft_plot(temp, FS_WRIST['TEMP'], "FFT TEMP (brut)",  "FFT_TEMP_raw.png")
fft_plot(acc,  FS_WRIST['ACC'],  "FFT ACC (brut)",   "FFT_ACC_raw.png")

print("\nGATA ✅  Vezi imaginile în:", OUT_DIR.resolve())
