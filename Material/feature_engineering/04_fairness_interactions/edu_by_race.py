# Auto-generated file
"""
Fairness interaction: Educational Return by Race

  F4 = education_num × race_*   (one interaction per race dummy)

Rationale
---------
Returns to education are not uniform across racial groups.
Interacting education_num with each race dummy allows the model to learn
(and analysts to audit) whether the income payoff for an additional year
of education differs between groups – a direct measure of educational
return inequality.

Usage
-----
Run *after* other_encoding.one_hot_race() so that race_* dummy columns
already exist in the DataFrame.
"""

import pandas as pd


def add_edu_by_race(
    df: pd.DataFrame,
    edu_col: str = "education-num",
    race_prefix: str = "race_",
) -> pd.DataFrame:
    """
    Add education × race interaction columns.

    Parameters
    ----------
    df          : DataFrame with education-num and OHE race_* columns.
    edu_col     : Education column name (raw or scaled).
    race_prefix : Prefix used to identify OHE race dummy columns.

    Returns
    -------
    DataFrame with new columns named  edu_x_{race_group}.
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
        new_col = f"edu_x_{group_label}"
        df[new_col] = df[edu_col] * df[rc]
        created.append(new_col)

    print(f"[EduByRace] Created {len(created)} interaction columns: {created}")
    return df


# ── Quick smoke-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "education-num": [9, 13, 16, 10, 12],
            "race_White":    [1, 0,  1,  0,  0],
            "race_Black":    [0, 1,  0,  1,  0],
            "race_Asian-Pac-Islander": [0, 0, 0, 0, 1],
        }
    )
    out = add_edu_by_race(sample)
    print(out)