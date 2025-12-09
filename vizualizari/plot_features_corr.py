import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
def plot_window(index, windows_eda, windows_bvp, windows_temp, windows_acc, labels_list):
    fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=False)

    axs[0].plot(windows_eda[index])
    axs[0].set_title(f"EDA – fereastra {index}, label = {labels_list[index]}")

    axs[1].plot(windows_bvp[index])
    axs[1].set_title("BVP")

    axs[2].plot(windows_temp[index])
    axs[2].set_title("Temperature")

    axs[3].plot(windows_acc[index])
    axs[3].set_title("ACC (magnitudine)")

    plt.tight_layout()
    plt.show()


def plot_bvp_peaks(index, windows_bvp, fs=64):
    bvp = np.asarray(windows_bvp[index])

    # curățare + peak detection
    bvp_clean = nk.ppg_clean(bvp, sampling_rate=fs)
    info = nk.ppg_findpeaks(bvp_clean, sampling_rate=fs)
    peaks = info["PPG_Peaks"]

    plt.figure(figsize=(12,4))
    plt.plot(bvp_clean, label="BVP clean")
    plt.scatter(peaks, bvp_clean[peaks], color='red', s=20, label='Peaks')
    plt.title(f"BVP clean + peaks pentru fereastra {index}")
    plt.legend()
    plt.show()

def plot_window_labels(labels_list):
    plt.figure(figsize=(14,2))
    plt.plot(labels_list, drawstyle='steps-mid')
    plt.title("Etichete pe ferestre (0=non-stress, 1=stress, -1=tranziție)")
    plt.xlabel("Index fereastră")
    plt.ylabel("Label")
    plt.show()

def plot_window_debug(i, windows_eda, windows_bvp, windows_temp, windows_acc, labels_list):
    fig, axs = plt.subplots(4,1, figsize=(15,10))
    axs[0].plot(windows_eda[i]); axs[0].set_title(f"EDA win {i}, label={labels_list[i]}")
    axs[1].plot(windows_bvp[i]); axs[1].set_title("BVP")
    axs[2].plot(windows_temp[i]); axs[2].set_title("TEMP")
    axs[3].plot(windows_acc[i]); axs[3].set_title("ACC")
    plt.tight_layout()
    plt.show()
