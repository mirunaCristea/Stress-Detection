import pickle
import numpy as np

def load_data_wesad(file_path):
    """
    Deschide fișierul .pkl și returnează semnalele de la brățară + etichetele.
    """

    with open(file_path, 'rb') as file: # deschidem fisierul in modul read-binary
        data = pickle.load(file, encoding='latin1')
    signals = data['signal']['wrist'] # extragem semnalele de la bratara
    labels = np.asarray(data['label']).astype(int)# extragem etichetele
    return signals, labels

def cut_30s(signals,labels,fs_dict,fs_label=700):
    """
    Eliminăm primele 30 de secunde din fiecare semnal (perioada de calibrare + zgomot)
    """
    cut_signals = {}

    for name, signal in signals.items():
        fs = fs_dict[name]
        cut_index = int(fs * 30)  # indexul corespunzator primelor 30 de secunde
        if name == 'ACC':
            #ACC are 3 axe -> vom taia pe fiecare axa in parte
            cut_signals['ACC'] = signal[cut_index:,:]  # taiem semnalul
        else:
            cut_signals[name] = signal[cut_index:]  # taiem semnalul

    cut_labels = labels[int(fs_label * 30):]  # taiem etichetele

    return cut_signals, cut_labels