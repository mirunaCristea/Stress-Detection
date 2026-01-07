import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

LABEL_COLORS = {
    1: 'green',     # baseline
    2: 'red',       # stress
    3: 'orange',    # amusement
    4: 'blue',      # meditation
}
LABEL_NAMES = {
    1: 'Baseline',
    2: 'Stress',
    3: 'Amusement',
    4: 'Meditation',
}


def plot_raw_vs_filtered(raw, filtered, fs, title='', unit='', show_signal='both', labels=None, fs_labels=700, show_labels=True):
    """
    Plotează semnalele brute și filtrate pentru comparație.

    Parametri:
    - raw: array-like sau None, semnalul brut de intrare. Dacă este None, nu se plotează.
    - filtered: array-like sau None, semnalul filtrat. Dacă este None, nu se plotează.
    - fs: float, frecvența de eșantionare a semnalului.
    - title: str, titlul graficului.
    - unit: str, unitatea de măsură pentru axa y.
    - show_signal: str, specifică ce semnal să fie afișat. Poate fi 'raw', 'filtered' sau 'both'.
    - show_labels: -afiseara segmentele de etichete pe grafic 
    """
    t = np.arange(len(raw)) / fs 
    plt.figure(figsize=(12, 4))

    if show_signal in 'both':
        plt.plot(t,raw, label='Brut', alpha=0.5) 
        plt.plot(t,filtered, label='Filtrat', linewidth=1.2, color='blue')
    elif show_signal == 'raw':
        plt.plot(t,raw, label='Brut', color='black')
    elif show_signal == 'filtered':
        plt.plot(t,filtered, label='Filtrat', color='blue')

    if show_labels and labels is not None:
        factor = int(fs_labels // fs)  
        labels_down = labels[::factor] # subeșantionare etichete pentru a se potrivi cu frecvența semnalului

        last_label = labels_down[0]
        start_idx = 0
        for i, lbl in enumerate(labels_down):
            if lbl != last_label or i == len(labels_down)-1:
                if last_label in LABEL_COLORS:
                    plt.axvspan(start_idx/fs, i/fs, color=LABEL_COLORS[last_label], alpha=0.2)
                start_idx = i
                last_label = lbl

        # legendă colorată
        
        patches = [mpatches.Patch(color=c, alpha=0.25, label=LABEL_NAMES[l])
               for l, c in LABEL_COLORS.items()]
        plt.legend(handles=patches + [
            plt.Line2D([], [], color='gray', label='Brut'),
            plt.Line2D([], [], color='blue', label='Filtrat')
        ])

    else:
        plt.legend()

    plt.title(title)
    plt.xlabel('Timp (s)')
    plt.ylabel(f'Amplitudine ({unit})')
    plt.grid()
    plt.tight_layout()
    plt.show()