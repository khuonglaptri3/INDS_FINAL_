# Auto-generated file
"""
Collapse marital-status → binary married_flag then One-Hot encode.

Married  (married_flag = 1):
  Married-civ-spouse, Married-AF-spouse

Not married (married_flag = 0):
  Never-married, Divorced, Separated, Widowed, Married-spouse-absent

One-Hot encoding is applied to the remaining relationship column
(handled here as a convenience so all marital-related transforms
are in one place).
"""

import pandas as pd

MARRIED_VALUES = {"Married-civ-spouse", "Married-AF-spouse"}


def add_married_flag(
    df: pd.DataFrame,
    source_col: str = "marital-status",
    flag_col: str = "married_flag",
    drop_source: bool = True,
) -> pd.DataFrame:
    """
    Create a binary married_flag from marital-status.

    Parameters
    ----------
    df          : Input DataFrame.
    source_col  : Original marital status column.
    flag_col    : Name of the new binary column.
    drop_source : Whether to drop the original column.

    Returns
    -------
    DataFrame with married_flag (int 0/1) added.
    """
    df = df.copy()
    df[flag_col] = df[source_col].isin(MARRIED_VALUES).astype(int)

    if drop_source:
        df = df.drop(columns=[source_col])

    married_count = df[flag_col].sum()
    total = len(df)
    print(
        f"[MaritalStatus] '{source_col}' → '{flag_col}': "
        f"{married_count}/{total} married ({married_count/total:.1%})"
    )
    return df


def one_hot_marital(
    df: pd.DataFrame,
    cols: list = None,
    drop_first: bool = False,
) -> pd.DataFrame:
    """
    Apply One-Hot encoding to relationship (and optionally other object cols).

    Parameters
    ----------
    df        : DataFrame (after add_married_flag has run).
    cols      : Columns to OHE. Defaults to ['relationship'].
    drop_first: Whether to drop the first dummy category.
    """
    if cols is None:
        cols = ["relationship"]

    df = df.copy()
    df = pd.get_dummies(df, columns=cols, drop_first=drop_first, dtype=int)
    print(f"[MaritalStatus] One-Hot encoded: {cols}")
    return df


# ── Quick smoke-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "marital-status": [
                "Married-civ-spouse",
                "Never-married",
                "Divorced",
                "Married-AF-spouse",
                "Widowed",
            ],
            "relationship": [
                "Husband",
                "Not-in-family",
                "Unmarried",
                "Wife",
                "Other-relative",
            ],
        }
    )
    out = add_married_flag(sample)
    out = one_hot_marital(out)
    print(out)