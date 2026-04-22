# Auto-generated file
"""
Map native-country → country_income_group using World Bank income tiers.

Groups
------
  L   = Low income
  LM  = Lower-middle income
  UM  = Upper-middle income
  H   = High income

Encoding
--------
  Ordinal: L=0, LM=1, UM=2, H=3  (preserves economic ordering)

Data source
-----------
  mapping_.csv  –  columns: [country_adult_dataset, income_group]
  Loaded from the path provided at runtime (or the default constant below).
"""

import pandas as pd
import os

# ── Ordinal mapping ──────────────────────────────────────────────────────────
ORDINAL_MAP = {"L": 0, "LM": 1, "UM": 2, "H": 3}

DEFAULT_MAPPING_PATH = (
    r"C:\Users\lanph\OneDrive\Desktop\Introduction to Data Science"
    r"\Final_Project\Adult_Project_Final_Term\data\processed\mapping_.csv"
)


def load_mapping(path: str = DEFAULT_MAPPING_PATH) -> dict:
    """
    Read mapping CSV and return a dict {country_name: income_group_code}.

    Parameters
    ----------
    path : Path to mapping_.csv with columns [country_adult_dataset, income_group].
    """
    df_map = pd.read_csv(path)
    mapping = dict(zip(df_map["country_adult_dataset"], df_map["income_group"]))
    print(f"[CountryIncomeGroup] Loaded {len(mapping)} country mappings from:\n  {path}")
    return mapping


def add_country_income_group(
    df: pd.DataFrame,
    mapping: dict,
    source_col: str = "native-country",
    target_col: str = "country_income_group",
    drop_source: bool = True,
) -> pd.DataFrame:
    """
    Replace native-country with an ordinal country_income_group feature.

    Parameters
    ----------
    df          : Input DataFrame.
    mapping     : {country_name: income_group_code} dict from load_mapping().
    source_col  : Original country column (default: 'native-country').
    target_col  : New ordinal column name.
    drop_source : Whether to drop the original column.

    Returns
    -------
    DataFrame with country_income_group (int 0-3) added.
    """
    df = df.copy()

    # Map to income group code (H / UM / LM / L); fallback 'H' for unknowns
    df[target_col] = (
        df[source_col]
        .map(mapping)
        .fillna("H")   # United-States and most missing → High income
    )

    # Ordinal encode
    df[target_col] = df[target_col].map(ORDINAL_MAP).astype(int)

    if drop_source:
        df = df.drop(columns=[source_col])

    print(f"[CountryIncomeGroup] '{source_col}' → '{target_col}' (ordinal 0-3)")
    print(df[target_col].value_counts().sort_index().rename("count"))
    return df


# ── Quick smoke-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "native-country": [
                "United-States",
                "Mexico",
                "India",
                "Cambodia",
                "Unknown-Country",
            ]
        }
    )
    # Build a minimal mock mapping for the test
    mock_mapping = {
        "United-States": "H",
        "Mexico": "UM",
        "India": "LM",
        "Cambodia": "L",
    }
    out = add_country_income_group(sample, mock_mapping)
    print(out)