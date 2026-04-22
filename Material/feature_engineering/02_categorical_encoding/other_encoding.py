# Auto-generated file
"""
Encoding for remaining categorical variables.

Variable        Method
-----------     ------------------------------------
workclass       Leave-One-Out (LOO) Encoding
race            One-Hot Encoding
sex             Binary Encoding  (Male=1, Female=0)
"""

import pandas as pd
from category_encoders import LeaveOneOutEncoder
import joblib
import os


# ── workclass – Leave-One-Out Encoding ──────────────────────────────────────

def fit_loo_encoder(
    df: pd.DataFrame,
    target: pd.Series,
    col: str = "workclass",
) -> LeaveOneOutEncoder:
    """
    Fit a Leave-One-Out encoder on workclass using the binary income target.

    Parameters
    ----------
    df     : DataFrame containing the column.
    target : Binary Series (0 / 1) representing income <=50K / >50K.
    col    : Column name (default: 'workclass').
    """
    enc = LeaveOneOutEncoder(cols=[col], sigma=0.05)
    enc.fit(df[[col]], target)
    print(f"[OtherEncoding] LOO encoder fitted on '{col}'.")
    return enc


def transform_loo(
    df: pd.DataFrame,
    encoder: LeaveOneOutEncoder,
    col: str = "workclass",
) -> pd.DataFrame:
    """Apply a fitted LOO encoder."""
    df = df.copy()
    df[col] = encoder.transform(df[[col]])[col]
    return df


# ── race – One-Hot Encoding ──────────────────────────────────────────────────

def one_hot_race(
    df: pd.DataFrame,
    col: str = "race",
    drop_first: bool = False,
) -> pd.DataFrame:
    """
    One-Hot encode the race column.
    Produces dummy columns used for both modelling and fairness interactions.
    """
    df = df.copy()
    df = pd.get_dummies(df, columns=[col], drop_first=drop_first, dtype=int)
    race_dummies = [c for c in df.columns if c.startswith("race_")]
    print(f"[OtherEncoding] One-Hot encoded 'race' → {race_dummies}")
    return df


# ── sex – Binary Encoding ────────────────────────────────────────────────────

def encode_sex(
    df: pd.DataFrame,
    col: str = "sex",
    drop_source: bool = True,
) -> pd.DataFrame:
    """
    Binary encode sex: Male → 1, Female → 0.
    Any unexpected value is mapped to NaN then filled with 0.
    """
    df = df.copy()
    sex_map = {"Male": 1, "Female": 0}
    df["sex_binary"] = df[col].map(sex_map).fillna(0).astype(int)
    if drop_source:
        df = df.drop(columns=[col])
    print("[OtherEncoding] 'sex' → 'sex_binary' (Male=1, Female=0)")
    return df


# ── Persistence helpers ──────────────────────────────────────────────────────

def save_encoder(encoder, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(encoder, path)
    print(f"[OtherEncoding] Encoder saved → {path}")


def load_encoder(path: str):
    return joblib.load(path)


# ── Quick smoke-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "workclass": ["Private", "Self-emp-not-inc", "Gov", "Private", "?"],
            "race": ["White", "Black", "Asian-Pac-Islander", "White", "Other"],
            "sex": ["Male", "Female", "Male", "Female", "Male"],
        }
    )
    target = pd.Series([1, 0, 1, 0, 0])

    enc = fit_loo_encoder(sample, target)
    out = transform_loo(sample.copy(), enc)
    out = one_hot_race(out)
    out = encode_sex(out)
    print(out)