# Auto-generated file
"""
Econometric interaction: Net Investment Capacity

  F3 = capital_gain − capital_loss   (also exported as F9)

Rationale
---------
capital_gain and capital_loss individually reflect investment income and
losses.  Their difference – net_capital – provides a single composite
measure of financial strength and wealth accumulation, reducing
multicollinearity and providing a cleaner signal for income classification.

Note: Apply this transform *after* Yeo-Johnson has been applied to
capital_gain and capital_loss so that the net value operates on
variance-stabilised inputs.
"""

import pandas as pd


def add_net_capital(
    df: pd.DataFrame,
    gain_col: str = "capital-gain",
    loss_col: str = "capital-loss",
    out_col: str = "net_capital",
) -> pd.DataFrame:
    """
    Add net capital feature.

    Parameters
    ----------
    df       : Input DataFrame.
    gain_col : Capital gains column (Yeo-Johnson transformed recommended).
    loss_col : Capital losses column (Yeo-Johnson transformed recommended).
    out_col  : Output column name.

    Returns
    -------
    DataFrame with `out_col` appended.
    """
    df = df.copy()
    df[out_col] = df[gain_col] - df[loss_col]
    print(
        f"[NetCapital] Created '{out_col}' = {gain_col} − {loss_col}  "
        f"| mean={df[out_col].mean():.4f}, std={df[out_col].std():.4f}"
    )
    return df


# ── Quick smoke-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "capital-gain": [0.0, 1.23, 3.45, 0.0, 2.10],
            "capital-loss": [0.0, 0.0,  0.80, 1.50, 0.0],
        }
    )
    out = add_net_capital(sample)
    print(out)