# Auto-generated file
"""
Map occupation → occupation_group (4 skill-level tiers).

Groups
------
  high_skill   : Exec-managerial, Prof-specialty
  office_tech  : Tech-support, Sales, Adm-clerical
  manual       : Craft-repair, Transport-moving, Machine-op-inspct, Other-service,
                 Handlers-cleaners
  specialised  : Farming-fishing, Armed-Forces, Priv-house-serv

Encoding
--------
  CatBoost Encoding (target-based, leave-one-out variant via category_encoders)
  – Allows the model to learn income boundaries between skill tiers
  – Avoids high-cardinality noise from individual occupation labels
"""

import pandas as pd
from category_encoders import CatBoostEncoder
import joblib
import os

# ── Grouping map ─────────────────────────────────────────────────────────────
OCCUPATION_GROUP_MAP = {
    "Exec-managerial":  "high_skill",
    "Prof-specialty":   "high_skill",
    "Tech-support":     "office_tech",
    "Sales":            "office_tech",
    "Adm-clerical":     "office_tech",
    "Craft-repair":     "manual",
    "Transport-moving": "manual",
    "Machine-op-inspct":"manual",
    "Other-service":    "manual",
    "Handlers-cleaners":"manual",
    "Farming-fishing":  "specialised",
    "Armed-Forces":     "specialised",
    "Priv-house-serv":  "specialised",
}


def add_occupation_group(
    df: pd.DataFrame,
    source_col: str = "occupation",
    target_col: str = "occupation_group",
    drop_source: bool = True,
) -> pd.DataFrame:
    """
    Add occupation_group string column from occupation labels.
    Unknown occupations default to 'specialised'.
    """
    df = df.copy()
    df[target_col] = df[source_col].map(OCCUPATION_GROUP_MAP).fillna("specialised")
    if drop_source:
        df = df.drop(columns=[source_col])
    print(f"[OccupationGroup] '{source_col}' → '{target_col}'")
    print(df[target_col].value_counts())
    return df


def fit_catboost_encoder(
    df: pd.DataFrame,
    target: pd.Series,
    col: str = "occupation_group",
) -> CatBoostEncoder:
    """
    Fit a CatBoost encoder on occupation_group using the binary income target.

    Parameters
    ----------
    df     : DataFrame that contains the col.
    target : Binary Series (0 / 1) representing income <=50K / >50K.
    col    : Column to encode.
    """
    enc = CatBoostEncoder(cols=[col], sigma=0.05, a=1)
    enc.fit(df[[col]], target)
    print(f"[OccupationGroup] CatBoost encoder fitted on '{col}'.")
    return enc


def transform_catboost(
    df: pd.DataFrame,
    encoder: CatBoostEncoder,
    col: str = "occupation_group",
) -> pd.DataFrame:
    """Apply a fitted CatBoost encoder."""
    df = df.copy()
    df[col] = encoder.transform(df[[col]])[col]
    return df


def save_encoder(encoder, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(encoder, path)
    print(f"[OccupationGroup] Encoder saved → {path}")


def load_encoder(path: str):
    return joblib.load(path)


# ── Quick smoke-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "occupation": [
                "Exec-managerial",
                "Sales",
                "Craft-repair",
                "Farming-fishing",
                "Unknown-Occ",
            ]
        }
    )
    target = pd.Series([1, 0, 0, 0, 1])
    out = add_occupation_group(sample)
    enc = fit_catboost_encoder(out, target)
    out_enc = transform_catboost(out, enc)
    print(out_enc)