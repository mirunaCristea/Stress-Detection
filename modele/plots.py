# modele/plots.py
# ------------------------------------------------------------
# Funcții de plot pentru evaluarea modelelor (RF + CNN)
# Include:
#  - matricea de confuzie (numărări + opțional normalizare)
#  - curba ROC + AUC
#  - curba Precision–Recall + AP
#  - distribuția probabilităților / scorurilor pe clase
#  - curba de calibrare (reliability curve)
#  - analiză în funcție de prag (threshold sweep)
#  - importanța caracteristicilor (Random Forest) + permutation importance
#  - evoluția loss-ului (CNN) și loss + accuracy pe epoci
# ------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    roc_auc_score,
    average_precision_score,
)
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve


# -------------------------
# Utilitare (conversii + validări)
# -------------------------

def _as_1d_int(y) -> np.ndarray:
    """
    Convertim vectorul y la np.ndarray 1D de int (0/1).
    E util ca să evităm forme ciudate (N,1) sau tipuri float/bool.
    """
    y = np.asarray(y).reshape(-1)
    return y.astype(int)


def _as_1d_float(x) -> np.ndarray:
    """
    Convertim scorurile/probabilitățile la np.ndarray 1D de float.
    """
    x = np.asarray(x).reshape(-1)
    return x.astype(float)


def _get_cm_matrix(cm_obj) -> np.ndarray:
    """
    Acceptă:
      - dict-ul întors de compute_confusion_matrix: {"matrix": [[..],[..]], ...}
      - un np.ndarray brut shape (2,2)

    Returnează:
      - np.ndarray (2,2) cu [ [TN, FP],
                             [FN, TP] ]
    """
    if isinstance(cm_obj, dict):
        m = np.asarray(cm_obj.get("matrix"))
        if m.shape != (2, 2):
            raise ValueError(f"Expected (2,2) confusion matrix, got {m.shape}")
        return m

    m = np.asarray(cm_obj)
    if m.shape != (2, 2):
        raise ValueError(f"Expected (2,2) confusion matrix, got {m.shape}")
    return m


# -------------------------
# Matrice de confuzie
# -------------------------

def plot_confusion_matrix(
    cm: Union[np.ndarray, Dict[str, Any]],
    labels: Tuple[str, str] = ("No stress", "Stress"),
    normalize: Optional[str] = None,  # None | "true" | "pred" | "all"
    title: str = "Confusion matrix",
    ax: Optional[plt.Axes] = None,
    savepath: Optional[str] = None,
    show: bool = True,
):
    """
    Plotează matricea de confuzie.
    - Pe diagonală sunt predicțiile corecte (TN și TP).
    - În afara diagonalei sunt erorile (FP și FN).

    Parametrul normalize:
      - None  -> afișează doar numărări (count-uri)
      - "true"-> normalizează pe rând (adică pe clasa reală): util pentru recall
      - "pred"-> normalizează pe coloană (adică pe clasa prezisă): util pentru precizie
      - "all" -> normalizează global

    Legendă/interpretare rapidă (pentru clasificare stres=1):
      - FN (False Negative) = stres real, prezis ca non-stres (eroare importantă!)
      - FP (False Positive) = non-stres real, prezis ca stres
    """
    m = _get_cm_matrix(cm)

    if ax is None:
        fig, ax = plt.subplots()

    disp = ConfusionMatrixDisplay(confusion_matrix=m, display_labels=list(labels))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")

    # Titlul va fi folosit ca „legendă” principală a figurii
    ax.set_title(
        f"{title}\n"
        f"(Diagonala = corect; Off-diagonala = erori. Stres=1: FN=stres ratat, FP=alarmă falsă)"
    )

    # Dacă normalizezi, suprapunem și procentele peste valori
    if normalize is not None:
        if normalize == "true":
            denom = m.sum(axis=1, keepdims=True)
        elif normalize == "pred":
            denom = m.sum(axis=0, keepdims=True)
        elif normalize == "all":
            denom = m.sum()
        else:
            raise ValueError("normalize must be None, 'true', 'pred', or 'all'")

        denom = np.where(denom == 0, 1, denom)
        mn = m / denom

        for (i, j), _ in np.ndenumerate(m):
            ax.text(j, i + 0.28, f"{mn[i, j]*100:.1f}%", ha="center", va="center")

        ax.set_title(
            f"{title} (normalize={normalize})\n"
            f"Procentele arată proporția relativă (în funcție de normalizare)."
        )

    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return ax


# -------------------------
# ROC / PR
# -------------------------

def plot_roc_curve(
    y_true,
    y_score,
    title: str = "ROC curve",
    ax: Optional[plt.Axes] = None,
    savepath: Optional[str] = None,
    show: bool = True,
):
    """
    Curba ROC (Receiver Operating Characteristic).
    - x: FPR (False Positive Rate) = FP/(FP+TN)
    - y: TPR (True Positive Rate) = Recall = TP/(TP+FN)

    AUC (Area Under Curve):
    - 0.5 ~ aleator
    - 1.0 ~ separare perfectă

    Legendă/interpretare:
    - Cu cât curba e mai aproape de colțul stânga-sus, cu atât mai bine.
    - AUC oferă o măsură globală independentă de prag.
    """
    y_true = _as_1d_int(y_true)
    y_score = _as_1d_float(y_score)

    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan

    if ax is None:
        fig, ax = plt.subplots()

    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)

    ax.set_title(
        f"{title}\n"
        f"AUC={auc:.3f} (0.5=aleator, 1.0=perfect). Curba mai sus/stânga => mai bun."
        if np.isfinite(auc)
        else f"{title}\nAUC=NA (există o singură clasă în y_true)"
    )

    # mică legendă vizuală (în colț) ca să fie explicativ
    ax.legend([f"ROC (AUC={auc:.3f})" if np.isfinite(auc) else "ROC"], loc="lower right")

    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return ax, auc


def plot_pr_curve(
    y_true,
    y_score,
    title: str = "Precision-Recall curve",
    ax: Optional[plt.Axes] = None,
    savepath: Optional[str] = None,
    show: bool = True,
):
    """
    Curba Precision–Recall (PR) – foarte utilă când clasele sunt dezechilibrate.
    - x: Recall = TP/(TP+FN)
    - y: Precision = TP/(TP+FP)

    AP (Average Precision):
    - sumarizează curba PR (similar cu AUC pentru ROC)
    - valori mai mari => mai bine

    Legendă/interpretare:
    - Dacă vrei să nu ratezi stres (recall mare), te uiți în special la partea dreaptă.
    - Dacă ai multe alarme false (FP), precizia scade.
    """
    y_true = _as_1d_int(y_true)
    y_score = _as_1d_float(y_score)

    ap = average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan

    if ax is None:
        fig, ax = plt.subplots()

    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax)

    ax.set_title(
        f"{title}\n"
        f"AP={ap:.3f}. PR e mai relevant decât ROC la clase dezechilibrate."
        if np.isfinite(ap)
        else f"{title}\nAP=NA (există o singură clasă în y_true)"
    )

    ax.legend([f"PR (AP={ap:.3f})" if np.isfinite(ap) else "PR"], loc="lower left")

    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return ax, ap


# -------------------------
# Distribuția scorurilor pe clase
# -------------------------

def plot_score_distributions(
    y_true,
    y_score,
    bins: int = 30,
    title: str = "Score distributions by class",
    ax: Optional[plt.Axes] = None,
    savepath: Optional[str] = None,
    show: bool = True,
    labels: Tuple[str, str] = ("No stress", "Stress"),
):
    """
    Histogramă cu distribuția probabilităților (y_score) pentru fiecare clasă reală.
    - Dacă cele două histograme sunt bine separate => modelul separă bine clasele.
    - Dacă se suprapun mult => modelul e confuz.

    Legendă/interpretare:
    - zona de suprapunere indică unde apar erori (FP/FN)
    - te ajută să alegi un prag (threshold) potrivit
    """
    y_true = _as_1d_int(y_true)
    y_score = _as_1d_float(y_score)

    if ax is None:
        fig, ax = plt.subplots()

    s0 = y_score[y_true == 0]
    s1 = y_score[y_true == 1]

    ax.hist(s0, bins=bins, alpha=0.6, label=f"{labels[0]} (clasa 0)")
    ax.hist(s1, bins=bins, alpha=0.6, label=f"{labels[1]} (clasa 1)")

    ax.set_xlabel("Scor / probabilitate prezisă pentru clasa 'Stres'")
    ax.set_ylabel("Număr de ferestre (count)")
    ax.set_title(
        f"{title}\n"
        f"Separare bună = histograme distincte; suprapunere mare = clasificare dificilă."
    )
    ax.legend()

    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return ax


# -------------------------
# Curba de calibrare (reliability curve)
# -------------------------

def plot_calibration_curve(
    y_true,
    y_score,
    n_bins: int = 10,
    title: str = "Calibration curve",
    ax: Optional[plt.Axes] = None,
    savepath: Optional[str] = None,
    show: bool = True,
):
    """
    Curba de calibrare:
    - pe axa X: probabilitatea medie prezisă într-un bin
    - pe axa Y: fracția reală de pozitive (stres) în acel bin

    Linia diagonală (y=x) = calibrare perfectă.

    Interpretare:
    - dacă punctele sunt sub diagonală => modelul supraestimează probabilitățile
    - dacă punctele sunt peste diagonală => modelul subestimează probabilitățile
    """
    y_true = _as_1d_int(y_true)
    y_score = _as_1d_float(y_score)

    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=n_bins, strategy="uniform")

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(prob_pred, prob_true, marker="o", linewidth=2, label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Calibrare perfectă (y=x)")

    ax.set_xlabel("Probabilitate prezisă (medie pe bin)")
    ax.set_ylabel("Fracție reală de stres (pozitive) în bin")
    ax.set_title(
        f"{title}\n"
        f"Diagonală = perfect; abateri indică supra/sub-estimarea probabilităților."
    )
    ax.legend(loc="upper left")

    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return ax


# -------------------------
# Threshold sweep (metrici în funcție de prag)
# -------------------------

@dataclass
class ThresholdSweepResult:
    thresholds: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    accuracy: np.ndarray
    balanced_accuracy: np.ndarray


def _binary_metrics_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculează metrici simple pentru clasificare binară:
      - precision, recall, f1, accuracy, balanced_accuracy
    pentru clasa pozitivă = 1 (stres).
    """
    y_true = _as_1d_int(y_true)
    y_pred = _as_1d_int(y_pred)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)  # recall pentru stres (clasa 1)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)

    tpr = recall
    tnr = tn / (tn + fp + 1e-12)
    bal_acc = 0.5 * (tpr + tnr)

    return dict(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        accuracy=float(acc),
        balanced_accuracy=float(bal_acc),
    )


def threshold_sweep(
    y_true,
    y_score,
    thresholds: Optional[np.ndarray] = None,
) -> ThresholdSweepResult:
    """
    Calculează metrici pentru o listă de praguri (thresholds).
    E util ca să alegi pragul care maximizează F1/Recall etc.

    Atenție:
    - ideal pragul se alege pe TRAIN/VAL, nu pe TEST, ca să nu „umfli” rezultatele.
    """
    y_true = _as_1d_int(y_true)
    y_score = _as_1d_float(y_score)

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    prec, rec, f1, acc, bal = [], [], [], [], []
    for th in thresholds:
        y_pred = (y_score >= th).astype(int)
        m = _binary_metrics_from_preds(y_true, y_pred)
        prec.append(m["precision"])
        rec.append(m["recall"])
        f1.append(m["f1"])
        acc.append(m["accuracy"])
        bal.append(m["balanced_accuracy"])

    return ThresholdSweepResult(
        thresholds=np.asarray(thresholds),
        precision=np.asarray(prec),
        recall=np.asarray(rec),
        f1=np.asarray(f1),
        accuracy=np.asarray(acc),
        balanced_accuracy=np.asarray(bal),
    )


def plot_threshold_sweep(
    sweep: ThresholdSweepResult,
    title: str = "Metrics vs threshold",
    ax: Optional[plt.Axes] = None,
    savepath: Optional[str] = None,
    show: bool = True,
    which: Iterable[str] = ("f1", "recall", "precision", "balanced_accuracy"),
):
    """
    Plotează metricile în funcție de prag.
    Interpretare:
      - recall crește de obicei când pragul scade (dar cresc FP)
      - precision crește de obicei când pragul crește (dar cresc FN)
      - F1 caută un compromis între precision și recall
    """
    if ax is None:
        fig, ax = plt.subplots()

    for name in which:
        y = getattr(sweep, name)
        ax.plot(sweep.thresholds, y, label=name)

    ax.set_xlabel("Prag (threshold) pentru clasa 'Stres'")
    ax.set_ylabel("Valoarea metricii")
    ax.set_title(
        f"{title}\n"
        f"Ajută la alegerea pragului: F1 = compromis, Recall = cât stres detectezi."
    )
    ax.legend()

    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return ax


# -------------------------
# Interpretabilitate Random Forest
# -------------------------

def plot_rf_feature_importance(
    model,
    feature_names: Optional[Iterable[str]] = None,
    top_k: int = 20,
    title: str = "Random Forest feature importance",
    ax: Optional[plt.Axes] = None,
    savepath: Optional[str] = None,
    show: bool = True,
):
    """
    Plotează importanța caracteristicilor din Random Forest (Gini / impurity-based).
    Funcționează și dacă modelul este Pipeline(scaler + rf).

    Interpretare:
    - arată ce feature-uri au fost folosite cel mai mult în arbori
    - NU e perfect „cauzal”: poate fi bias pentru feature-uri cu variație mare
    """
    rf = model
    if hasattr(model, "named_steps") and "rf" in model.named_steps:
        rf = model.named_steps["rf"]

    if not hasattr(rf, "feature_importances_"):
        raise ValueError("Model has no feature_importances_ (este RandomForest?)")

    importances = np.asarray(rf.feature_importances_)

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(len(importances))]
    else:
        feature_names = list(feature_names)

    idx = np.argsort(importances)[::-1][:top_k]
    imp = importances[idx]
    names = [feature_names[i] for i in idx]

    if ax is None:
        fig, ax = plt.subplots()

    ax.barh(range(len(imp))[::-1], imp[::-1])
    ax.set_yticks(range(len(imp))[::-1], labels=names[::-1])
    ax.set_xlabel("Importanță (impurity-based)")
    ax.set_title(
        f"{title}\n"
        f"Top {top_k} feature-uri. Valoare mai mare => folosit mai des în decizii."
    )

    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return ax


def plot_permutation_importance(
    model,
    X,
    y,
    feature_names: Optional[Iterable[str]] = None,
    scoring: str = "f1",
    n_repeats: int = 10,
    random_state: int = 42,
    top_k: int = 20,
    title: str = "Permutation importance",
    ax: Optional[plt.Axes] = None,
    savepath: Optional[str] = None,
    show: bool = True,
):
    """
    Permutation importance:
    - permutăm valorile unui feature și vedem cât scade scorul (ex. F1).
    - dacă scorul scade mult => feature important.

    Avantaj:
    - e mai „corect” decât feature_importances_ în multe cazuri.

    Recomandare:
    - calculează pe setul de TEST subject (pentru interpretarea generalizării).
    """
    y = _as_1d_int(y)
    X_arr = X.values if hasattr(X, "values") else np.asarray(X)

    if feature_names is None:
        feature_names = getattr(X, "columns", None)
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X_arr.shape[1])]
    else:
        feature_names = list(feature_names)

    r = permutation_importance(
        model,
        X_arr,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    imp_mean = r.importances_mean
    imp_std = r.importances_std

    idx = np.argsort(imp_mean)[::-1][:top_k]
    names = [feature_names[i] for i in idx]
    mean = imp_mean[idx]
    std = imp_std[idx]

    if ax is None:
        fig, ax = plt.subplots()

    ax.barh(range(len(mean))[::-1], mean[::-1], xerr=std[::-1])
    ax.set_yticks(range(len(mean))[::-1], labels=names[::-1])
    ax.set_xlabel(f"Importanță (scădere în {scoring} după permutare)")
    ax.set_title(
        f"{title}\n"
        f"Top {top_k}. Barele = impact mediu; xerr = variabilitate (std) pe permutări."
    )

    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return ax, r


# -------------------------
# CNN training curve (loss history)
# -------------------------

def plot_training_loss(
    loss_history: Iterable[float],
    title: str = "Training loss over epochs",
    ax: Optional[plt.Axes] = None,
    savepath: Optional[str] = None,
    show: bool = True,
):
    """
    Plotează evoluția loss-ului pe epoci.
    Interpretare:
      - scade => modelul învață (pe setul pe care e calculat loss-ul)
      - dacă oscilează mult => poate lr prea mare / batch mic / date zgomotoase
    """
    loss_history = np.asarray(list(loss_history), dtype=float)
    epochs = np.arange(1, len(loss_history) + 1)

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(epochs, loss_history, marker="o")
    ax.set_xlabel("Epocă")
    ax.set_ylabel("Loss (BCEWithLogits)")
    ax.set_title(
        f"{title}\n"
        f"Scădere constantă = convergență; platou = posibilă saturație."
    )

    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return ax


def plot_loss_and_accuracy(
    history: dict,
    title: str = "CNN training: loss & accuracy",
    savepath: str | None = None,
    show: bool = True,
):
    """
    Plotează pe aceeași figură:
      - loss pe epoci (axa stângă)
      - accuracy pe epoci (axa dreaptă)

    Interpretare:
      - loss scade, accuracy crește => în general OK
      - loss scade dar accuracy nu crește => poate pragul nu e bun / date dezechilibrate
      - accuracy crește pe train dar pe test scade => supraînvățare (overfitting)
    """
    loss = np.asarray(history.get("loss", []), dtype=float)
    acc  = np.asarray(history.get("acc",  []), dtype=float)
    epochs = np.arange(1, len(loss) + 1)

    fig, ax1 = plt.subplots()

    ax1.plot(epochs, loss, marker="o", label="Loss (train)")
    ax1.set_xlabel("Epocă")
    ax1.set_ylabel("Loss (BCEWithLogits)")
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(epochs, acc, marker="o", label="Accuracy (train)")
    ax2.set_ylabel("Accuracy (train)")

    # Legendă combinată (de pe ambele axe)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title(
        f"{title}\n"
        f"Axa stângă: loss; Axa dreaptă: accuracy. Urmărește trendul pe epoci."
    )
    fig.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return ax1, ax2
