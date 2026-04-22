# Auto-generated file
"""
Econometric interaction: Household Responsibility & Labour Supply

  F2 = hours_per_week × married_flag   (also exported as F8)

Rationale
---------
Labour economics literature (e.g., Mincer, 1962) documents that married
individuals – particularly male breadwinners in the Adult dataset period –
tend to supply more hours of market labour to meet household financial
obligations.  The interaction term captures this differential effort and
its relationship with high income, beyond what hours or marital status
contribute independently.
"""

import pandas as pd


def add_household_labour(
    df: pd.DataFrame,
    hours_col: str = "hours-per-week",
    married_col: str = "married_flag",
    out_col: str = "household_labour",
) -> pd.DataFrame:
    """
    Add household labour interaction feature.

    Parameters
    ----------
    df          : Input DataFrame.
                  Requires married_flag (0/1) to exist – run
                  marital_status.add_married_flag() first.
    hours_col   : Hours-per-week column (raw or scaled).
    married_col : Binary married flag column.
    out_col     : Output interaction column name.

    Returns
    -------
    DataFrame with `out_col` appended.
    """
    df = df.copy()

    if married_col not in df.columns:
        raise KeyError(
            f"'{married_col}' not found. Run marital_status.add_married_flag() first."
        )

    df[out_col] = df[hours_col] * df[married_col]
    print(
        f"[HouseholdLabour] Created '{out_col}' = {hours_col} × {married_col}  "
        f"| mean={df[out_col].mean():.2f}, std={df[out_col].std():.2f}"
    )
    return df


# ── Quick smoke-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "hours-per-week": [40, 55, 40, 20, 60],
            "married_flag":   [1,  1,  0,  0,  1],
        }
    )
    out = add_household_labour(sample)
    print(out)