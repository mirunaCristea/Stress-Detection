# vizualizare/inspect_features.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def quick_report(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, top_k_corr: int = 25):
    """
    Face grafice rapide ca să vezi:
    - distribuția claselor
    - câte ferestre per subiect
    - distribuții (hist) pt câteva feature-uri cheie
    - boxplot pe clase pt EDA/BVP/TEMP/ACC
    - heatmap corelații (subset)
    - scatter ACC vs EDA (artefact check)
    """

    # ---- 1) Class balance
    plt.figure()
    classes, counts = np.unique(y, return_counts=True)
    plt.bar([str(c) for c in classes], counts)
    plt.title("Distribuția claselor (număr ferestre)")
    plt.xlabel("Clasă")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # ---- 2) Windows per subject
    plt.figure()
    subjs, subj_counts = np.unique(groups, return_counts=True)
    order = np.argsort(subj_counts)[::-1]
    plt.bar([str(s) for s in subjs[order]], subj_counts[order])
    plt.title("Număr ferestre per subiect")
    plt.xlabel("Subiect")
    plt.ylabel("Ferestre")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # ---- 3) Selectăm câteva feature-uri cheie (dacă există)
    key_feats = [
        "EDA_mean", "EDA_tonic_mean", "EDA_phasic_mean",
        "BVP_mean", "BVP_peak_freq",
        "TEMP_mean", "TEMP_slope",
        "net_acc_mean",
    ]
    key_feats = [c for c in key_feats if c in X.columns]

    # Histogram + overlay pe clase (2 clase) pentru feature-uri cheie
    for col in key_feats:
        plt.figure()
        # dacă ai mai mult de 2 clase, tot merge, dar va fi mai aglomerat
        for c in np.unique(y):
            vals = X.loc[y == c, col].astype(float).replace([np.inf, -np.inf], np.nan).dropna().values
            if vals.size == 0:
                continue
            plt.hist(vals, bins=40, alpha=0.5, label=f"clasa {c}")
        plt.title(f"Distribuție: {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ---- 4) Boxplot pe clase (ajută să vezi separabilitatea)
    for col in key_feats:
        plt.figure()
        data = []
        labels = []
        for c in np.unique(y):
            vals = X.loc[y == c, col].astype(float).replace([np.inf, -np.inf], np.nan).dropna().values
            if vals.size == 0:
                continue
            data.append(vals)
            labels.append(str(c))
        if len(data) >= 2:
            plt.boxplot(data, labels=labels, showfliers=False)
            plt.title(f"Boxplot pe clase: {col}")
            plt.xlabel("Clasă")
            plt.ylabel(col)
            plt.tight_layout()
            plt.show()

    # ---- 5) Heatmap corelații (subset top_k_corr features după varianță)
    X_num = X.select_dtypes(include=[np.number]).copy()
    X_num = X_num.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

    if X_num.shape[1] >= 3:
        # alegem top_k după varianță, ca să nu fie heatmap uriaș
        variances = X_num.var(axis=0).sort_values(ascending=False)
        chosen = variances.head(min(top_k_corr, len(variances))).index.tolist()
        corr = X_num[chosen].corr()

        plt.figure(figsize=(10, 8))
        plt.imshow(corr.values, aspect="auto")
        plt.colorbar()
        plt.title("Corelații (subset top features după varianță)")
        plt.xticks(range(len(chosen)), chosen, rotation=90, fontsize=7)
        plt.yticks(range(len(chosen)), chosen, fontsize=7)
        plt.tight_layout()
        plt.show()

    # ---- 6) Scatter: net_acc_mean vs EDA_mean (artefact check)
    if "net_acc_mean" in X.columns and "EDA_mean" in X.columns:
        plt.figure()
        for c in np.unique(y):
            xx = X.loc[y == c, "net_acc_mean"].astype(float).replace([np.inf, -np.inf], np.nan)
            yy = X.loc[y == c, "EDA_mean"].astype(float).replace([np.inf, -np.inf], np.nan)
            mask = xx.notna() & yy.notna()
            plt.scatter(xx[mask], yy[mask], s=8, alpha=0.5, label=f"clasa {c}")
        plt.title("Scatter: net_acc_mean vs EDA_mean")
        plt.xlabel("net_acc_mean")
        plt.ylabel("EDA_mean")
        plt.legend()
        plt.tight_layout()
        plt.show()
