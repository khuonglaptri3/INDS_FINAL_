# Auto-generated file
"""
Econometric interaction: Human Capital Accumulation

  F1 = age × education_num   (also exported as F7, same formula)

Rationale
---------
Becker's human capital theory suggests that earnings are a function of
both the level of education attained and the years over which that
education has been applied.  The product (age × education_num) proxies
lifetime accumulation of human capital and is a stronger predictor of
high income than either variable alone.
"""

import pandas as pd


def add_human_capital(
    df: pd.DataFrame,
    age_col: str = "age",
    edu_col: str = "education-num",
    out_col: str = "human_capital",
) -> pd.DataFrame:
    """
    Add human capital interaction feature.

    Parameters
    ----------
    df      : Input DataFrame (numerical columns already processed).
    age_col : Column name for age (raw or scaled).
    edu_col : Column name for education-num (raw or scaled).
    out_col : Name of the new interaction column.

    Returns
    -------
    DataFrame with `out_col` appended.

    Notes
    -----
    - Apply *after* robust scaling if you want the interaction to operate
      on scaled values; apply *before* if you want the raw product.
    - The feature is referred to as F1 / F7 in the research document –
      they are identical; the alias avoids confusion across sections.
    """
    df = df.copy()
    df[out_col] = df[age_col] * df[edu_col]
    print(
        f"[HumanCapital] Created '{out_col}' = {age_col} × {edu_col}  "
        f"| mean={df[out_col].mean():.2f}, std={df[out_col].std():.2f}"
    )
    return df


# ── Quick smoke-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "age": [25, 38, 52, 30],
            "education-num": [9, 13, 16, 10],
        }
    )
    out = add_human_capital(sample)
    print(out)