# metrics.py
"""
Acest fișier conține funcții pentru evaluarea performanței modelelor
de clasificare binară (Stress vs Non-Stress).

Clase:
- 0 = Non-stress
- 1 = Stress

Metricile sunt alese special pentru date dezechilibrate (cum este WESAD).
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix
)


def compute_classification_metrics(y_true, y_pred):
    """
    Calculează principalele metrici de clasificare.

    Parametri:
    ----------
    y_true : array-like
        Etichete reale (0 = Non-stress, 1 = Stress)
    y_pred : array-like
        Etichete prezise de model

    Returnează:
    -----------
    metrics : dict
        Dicționar cu valorile metricilor
    """

    metrics = {
        # Accuracy = proporția totală de predicții corecte
        # Atenție: poate fi înșelătoare când clasele sunt dezechilibrate
        "accuracy": accuracy_score(y_true, y_pred),

        # Balanced accuracy = media recall-ului pe fiecare clasă
        # Foarte potrivită pentru seturi dezechilibrate
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),

        # Precision pentru clasa Stress (1)
        # Din toate ferestrele prezise ca Stress, câte sunt corecte
        "precision_stress": precision_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),

        # Recall pentru clasa Stress (1)
        # Din toate ferestrele Stress reale, câte sunt detectate
        "recall_stress": recall_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),

        # F1-score pentru clasa Stress (1)
        # Compromis între precision și recall (foarte important în acest proiect)
        "f1_stress": f1_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),

        # F1 macro = media F1 pentru ambele clase (0 și 1)
        # Nu este influențată de clasa majoritară
        "f1_macro": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
    }

    return metrics


def compute_confusion_matrix(y_true, y_pred):
    """
    Calculează confusion matrix pentru clasificare binară.

    Formatul matricei:
        [[TN, FP],
         [FN, TP]]

    Unde:
    - TN: Non-stress prezis corect
    - FP: Non-stress prezis greșit ca Stress
    - FN: Stress ratat
    - TP: Stress detectat corect
    """

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "TN": int(cm[0, 0]),
        "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]),
        "TP": int(cm[1, 1]),
        "matrix": cm
    }
