# Auto-generated file
"""
Yeo-Johnson transformation for skewed financial variables.

Applies to: capital-gain, capital-loss
- Handles zero and negative values (extends Box-Cox)
- λ optimised via Maximum Likelihood Estimation
- Compresses extreme financial values and stabilises variance
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer
import joblib
import os

FINANCIAL_COLS = ["capital-gain", "capital-loss"]


def fit_yeo_johnson(df: pd.DataFrame, cols: list = FINANCIAL_COLS) -> PowerTransformer:
    """
    Fit Yeo-Johnson transformer on specified columns.

    Parameters
    ----------
    df   : Raw or cleaned DataFrame containing the financial columns.
    cols : List of column names to transform.

    Returns
    -------
    Fitted PowerTransformer (method='yeo-johnson').
    """
    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    pt.fit(df[cols])
    print(f"[YeoJohnson] Fitted on columns: {cols}")
    for col, lam in zip(cols, pt.lambdas_):
        print(f"  λ({col}) = {lam:.4f}")
    return pt


def transform_yeo_johnson(
    df: pd.DataFrame,
    transformer: PowerTransformer,
    cols: list = FINANCIAL_COLS,
) -> pd.DataFrame:
    """
    Apply a fitted Yeo-Johnson transformer and return the modified DataFrame.
    Original columns are replaced in-place.
    """
    df = df.copy()
    df[cols] = transformer.transform(df[cols])
    return df


def fit_transform_yeo_johnson(
    df: pd.DataFrame,
    cols: list = FINANCIAL_COLS,
) -> tuple[pd.DataFrame, PowerTransformer]:
    """
    Convenience wrapper: fit + transform in one call.

    Returns
    -------
    (transformed_df, fitted_transformer)
    """
    pt = fit_yeo_johnson(df, cols)
    df_out = transform_yeo_johnson(df, pt, cols)
    return df_out, pt


def save_transformer(transformer: PowerTransformer, path: str) -> None:
    """Persist the fitted transformer to disk for use in inference."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(transformer, path)
    print(f"[YeoJohnson] Transformer saved → {path}")


def load_transformer(path: str) -> PowerTransformer:
    """Load a previously saved PowerTransformer."""
    return joblib.load(path)


# ── Quick smoke-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "capital-gain": [0, 0, 0, 2000, 14084, 99999],
            "capital-loss": [0, 0, 1902, 0, 0, 4356],
        }
    )
    out, pt = fit_transform_yeo_johnson(sample)
    print("\nTransformed sample:")
    print(out)