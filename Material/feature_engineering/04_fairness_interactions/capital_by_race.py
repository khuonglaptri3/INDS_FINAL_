"""
Fairness interaction: Capital Access Inequality by Race

  F6 = net_capital × race_*   (one interaction per race dummy)

Rationale
---------
Wealth gaps between racial groups are well-documented.  Interacting
net_capital (= capital_gain − capital_loss) with each race dummy
quantifies whether investment-income advantages or disadvantages
accumulate differently across racial groups, enabling direct measurement
of investment inequality in model audits.

Usage
-----
Run *after* both:
  - net_capital.add_net_capital()         → net_capital column
  - other_encoding.one_hot_race()         → race_* dummy columns
"""

import pandas as pd


def add_capital_by_race(
    df: pd.DataFrame,
    capital_col: str = "net_capital",
    race_prefix: str = "race_",
) -> pd.DataFrame:
    """
    Add net_capital × race interaction columns.

    Parameters
    ----------
    df           : DataFrame with net_capital and OHE race_* columns.
    capital_col  : Net capital column (output of net_capital.add_net_capital()).
    race_prefix  : Prefix used to identify OHE race dummy columns.

    Returns
    -------
    DataFrame with new columns named  capital_x_{race_group}.
    """
    df = df.copy()
    race_cols = [c for c in df.columns if c.startswith(race_prefix)]

    if not race_cols:
        raise KeyError(
            f"No columns with prefix '{race_prefix}' found. "
            "Run other_encoding.one_hot_race() before this step."
        )
    if capital_col not in df.columns:
        raise KeyError(
            f"'{capital_col}' not found. "
            "Run net_capital.add_net_capital() before this step."
        )

    created = []
    for rc in race_cols:
        group_label = rc.replace(race_prefix, "")
        new_col = f"capital_x_{group_label}"
        df[new_col] = df[capital_col] * df[rc]
        created.append(new_col)

    print(f"[CapitalByRace] Created {len(created)} interaction columns: {created}")
    return df


# ── Quick smoke-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "net_capital": [0.0, 1.23, -0.8, 3.10, 0.5],
            "race_White":  [1,   0,    1,    0,    0],
            "race_Black":  [0,   1,    0,    1,    0],
            "race_Asian-Pac-Islander": [0, 0, 0, 0, 1],
        }
    )
    out = add_capital_by_race(sample)
    print(out)