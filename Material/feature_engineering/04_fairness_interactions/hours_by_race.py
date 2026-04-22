# Auto-generated file
"""
Fairness interaction: Labour Hour Burden by Race

  F5 = hours_per_week × race_*   (one interaction per race dummy)

Rationale
---------
If certain racial groups must work significantly more hours to achieve
the same income level, this structural disparity should be measurable
and auditable in the model.  Interacting hours-per-week with each race
dummy quantifies this differential labour burden and enables fairness
metrics to be computed per group.

Usage
-----
Run *after* other_encoding.one_hot_race() so that race_* dummy columns
already exist in the DataFrame.
"""

import pandas as pd


def add_hours_by_race(
    df: pd.DataFrame,
    hours_col: str = "hours-per-week",
    race_prefix: str = "race_",
) -> pd.DataFrame:
    """
    Add hours-per-week × race interaction columns.

    Parameters
    ----------
    df          : DataFrame with hours-per-week and OHE race_* columns.
    hours_col   : Hours worked column (raw or scaled).
    race_prefix : Prefix used to identify OHE race dummy columns.

    Returns
    -------
    DataFrame with new columns named  hours_x_{race_group}.
    """
    df = df.copy()
    race_cols = [c for c in df.columns if c.startswith(race_prefix)]

    if not race_cols:
        raise KeyError(
            f"No columns with prefix '{race_prefix}' found. "
            "Run other_encoding.one_hot_race() before this step."
        )

    created = []
    for rc in race_cols:
        group_label = rc.replace(race_prefix, "")
        new_col = f"hours_x_{group_label}"
        df[new_col] = df[hours_col] * df[rc]
        created.append(new_col)

    print(f"[HoursByRace] Created {len(created)} interaction columns: {created}")
    return df


# ── Quick smoke-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "hours-per-week": [40, 50, 60, 35, 45],
            "race_White":     [1,  0,  1,  0,  0],
            "race_Black":     [0,  1,  0,  1,  0],
            "race_Asian-Pac-Islander": [0, 0, 0, 0, 1],
        }
    )
    out = add_hours_by_race(sample)
    print(out)