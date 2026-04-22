# Auto-generated file
"""
Robust Scaling for continuous numerical variables.

Applies to: age, hours-per-week, education-num
- Uses median and IQR instead of mean/std
- Reduces influence of extreme outliers (very high working hours, outlier ages)
- Formula: X_scaled = (X - median) / IQR
"""

import pandas as pd
from sklearn.preprocessing import RobustScaler
import joblib
import os

ROBUST_COLS = ["age", "hours-per-week", "education-num"]


def fit_robust_scaler(df: pd.DataFrame, cols: list = ROBUST_COLS) -> RobustScaler:
    """
    Fit a RobustScaler on specified columns.

    Parameters
    ----------
    df   : DataFrame containing numerical columns.
    cols : Columns to scale.

    Returns
    -------
    Fitted RobustScaler.
    """
    scaler = RobustScaler()
    scaler.fit(df[cols])
    print(f"[RobustScaler] Fitted on columns: {cols}")
    for col, center, scale in zip(cols, scaler.center_, scaler.scale_):
        print(f"  {col}: median={center:.2f}, IQR={scale:.2f}")
    return scaler


def transform_robust(
    df: pd.DataFrame,
    scaler: RobustScaler,
    cols: list = ROBUST_COLS,
) -> pd.DataFrame:
    """
    Apply a fitted RobustScaler; returns a copy with scaled columns.
    """
    df = df.copy()
    df[cols] = scaler.transform(df[cols])
    return df


def fit_transform_robust(
    df: pd.DataFrame,
    cols: list = ROBUST_COLS,
) -> tuple[pd.DataFrame, RobustScaler]:
    """
    Convenience: fit + transform in one call.

    Returns
    -------
    (scaled_df, fitted_scaler)
    """
    scaler = fit_robust_scaler(df, cols)
    df_out = transform_robust(df, scaler, cols)
    return df_out, scaler


def save_scaler(scaler: RobustScaler, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)
    print(f"[RobustScaler] Scaler saved → {path}")


def load_scaler(path: str) -> RobustScaler:
    return joblib.load(path)


# ── Quick smoke-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "age": [17, 25, 38, 52, 90],
            "hours-per-week": [10, 40, 40, 60, 99],
            "education-num": [5, 9, 13, 16, 16],
        }
    )
    out, sc = fit_transform_robust(sample)
    print("\nScaled sample:")
    print(out)