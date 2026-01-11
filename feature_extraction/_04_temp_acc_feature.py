import numpy as np

def extract_temp_features(window):
    """
    TEMP: S
      - TEMP_mean  : media temperaturii
      - TEMP_std   : variabilitatea
      - TEMP_slope : panta trendului în fereastră (coeficientul liniar)
    """
    if len(window) == 0:
        return {
            "TEMP_mean": np.nan,
            "TEMP_std": np.nan,
            "TEMP_slope": np.nan,
        }

    temp = np.asarray(window, dtype=float).reshape(-1)
    temp = temp[np.isfinite(temp)]
    if temp.size < 2:
        return {
            "TEMP_mean": np.nan if temp.size == 0 else float(np.mean(temp)),
            "TEMP_std": np.nan,
            "TEMP_slope": np.nan,
        }

    x = np.arange(temp.size, dtype=float)
    slope = float(np.polyfit(x, temp, 1)[0])

    return {
        "TEMP_mean": float(np.mean(temp)),
        "TEMP_std": float(np.std(temp)),
        "TEMP_slope": slope,
    }


def extract_acc_features(window):
    """
    ACC: feature-urile folosite pe 3 axe + net_acc_mean.
    Așteaptă window cu shape (N,3) = [x, y, z].

    Feature-uri pe fiecare axă:
      - ACC_<axis>_mean/std/min/max

    + net_acc_mean:
      - media magnitudinii accelerației: sqrt(x^2 + y^2 + z^2)
    """
    if len(window) == 0:
        feats = {}
        for axis in ["x", "y", "z"]:
            feats[f"ACC_{axis}_mean"] = np.nan
            feats[f"ACC_{axis}_std"]  = np.nan
            feats[f"ACC_{axis}_min"]  = np.nan
            feats[f"ACC_{axis}_max"]  = np.nan
        feats["net_acc_mean"] = np.nan
        return feats

    acc = np.asarray(window, dtype=float)
    if acc.ndim != 2 or acc.shape[1] != 3:
        raise ValueError("ACC window must have shape (N, 3) -> columns [x,y,z]")

    # NaN-safe pe fiecare axă
    def stats(v):
        v = v[np.isfinite(v)]
        if v.size == 0:
            return np.nan, np.nan, np.nan, np.nan
        return float(np.mean(v)), float(np.std(v)), float(np.min(v)), float(np.max(v))

    x_m, x_s, x_min, x_max = stats(acc[:, 0])
    y_m, y_s, y_min, y_max = stats(acc[:, 1])
    z_m, z_s, z_min, z_max = stats(acc[:, 2])

    # magnitudine (net accel) – media pe fereastră
    mag = np.sqrt(acc[:, 0]**2 + acc[:, 1]**2 + acc[:, 2]**2)
    mag = mag[np.isfinite(mag)]
    net_mean = np.nan if mag.size == 0 else float(np.mean(mag))

    return {
        "ACC_x_mean": x_m, "ACC_x_std": x_s, "ACC_x_min": x_min, "ACC_x_max": x_max,
        "ACC_y_mean": y_m, "ACC_y_std": y_s, "ACC_y_min": y_min, "ACC_y_max": y_max,
        "ACC_z_mean": z_m, "ACC_z_std": z_s, "ACC_z_min": z_min, "ACC_z_max": z_max,
        "net_acc_mean": net_mean,
    }
