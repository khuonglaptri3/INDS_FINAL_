

import argparse
import os
import pathlib
import importlib.util
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import sys
# Ensure absolute imports work when running from feature_engineering/.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

_HERE = pathlib.Path(__file__).parent

def _load(rel_path: str):
    p = _HERE / rel_path
    spec = importlib.util.spec_from_file_location(p.stem, str(p))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

yj  = _load("01_nonlinear_transforms/yeo_johnson_transform.py")
rs  = _load("01_nonlinear_transforms/robust_scaling.py")
cig = _load("02_categorical_encoding/country_income_group.py")
og  = _load("02_categorical_encoding/occupation_group.py")
ms  = _load("02_categorical_encoding/marital_status.py")
oe  = _load("02_categorical_encoding/other_encoding.py")
hc  = _load("03_econometric_interactions/human_capital.py")
hl  = _load("03_econometric_interactions/household_labour.py")
nc  = _load("03_econometric_interactions/net_capital.py")
ebr = _load("04_fairness_interactions/edu_by_race.py")
hbr = _load("04_fairness_interactions/hours_by_race.py")
cbr = _load("04_fairness_interactions/capital_by_race.py")

BASE      = r"C:\Users\lanph\OneDrive\Desktop\Introduction to Data Science\Final_Project\Adult_Project_Final_Term"
DATA_IN   = os.path.join(BASE, "data", "processed", "adult_after_eda.csv")
MAP_PATH  = os.path.join(BASE, "data", "processed", "mapping_.csv")
TRAIN_OUT = os.path.join(BASE, "data", "processed", "adult_features_train.csv")
TEST_OUT  = os.path.join(BASE, "data", "processed", "adult_features_test.csv")
MODEL_DIR = os.path.join(BASE, "models", "encoders")


def build_target(df: pd.DataFrame, col: str = "income") -> pd.Series:
    return (df[col].str.strip().str.replace(".", "", regex=False) == ">50K").astype(int)


def _apply_non_target_steps(df: pd.DataFrame, country_mapping: dict,
                              yj_transformer, robust_scaler) -> pd.DataFrame:
    """
    Các bước KHÔNG dùng target — an toàn để áp dụng lên toàn bộ data
    trước khi tách train/test.
    """
    # Yeo-Johnson (transform only, đã fit)
    df = yj.transform_yeo_johnson(df, yj_transformer)
    # Robust Scaling (transform only, đã fit)
    df = rs.transform_robust(df, robust_scaler)
    # Country income group (lookup table, không dùng target)
    df = cig.add_country_income_group(df, country_mapping)
    # Occupation grouping (rule-based, không dùng target)
    df = og.add_occupation_group(df)
    # Marital → binary + OHE relationship (không dùng target)
    df = ms.add_married_flag(df)
    df = ms.one_hot_marital(df, cols=["relationship"])
    # Race OHE (không dùng target)
    df = oe.one_hot_race(df)
    # Sex binary (không dùng target)
    df = oe.encode_sex(df)
    return df


def _apply_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Tương tác kinh tế lượng và fairness — không dùng target."""
    df = hc.add_human_capital(df)
    df = hl.add_household_labour(df)
    df = nc.add_net_capital(df)
    df = ebr.add_edu_by_race(df)
    df = hbr.add_hours_by_race(df)
    df = cbr.add_capital_by_race(df)
    return df


def run_pipeline(
    data_in:   str   = DATA_IN,
    map_path:  str   = MAP_PATH,
    train_out: str   = TRAIN_OUT,
    test_out:  str   = TEST_OUT,
    model_dir: str   = MODEL_DIR,
    test_size: float = 0.2,
    random_state: int = 42,
    fit: bool  = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chạy pipeline có tách train/test đúng cách.

    Returns
    -------
    (df_train_encoded, df_test_encoded)
    """
    os.makedirs(model_dir, exist_ok=True)
    print("=" * 60)
    print("ADULT INCOME – FEATURE ENGINEERING PIPELINE (leak-free)")
    print("=" * 60)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print("\n[1] Loading data …")
    df = pd.read_csv(data_in)
    country_mapping = cig.load_mapping(map_path)
    print(f"    Shape: {df.shape}")

    # ── 2. Fit các scalers phi tuyến trên TOÀN BỘ data ───────────────────────
    # Yeo-Johnson & RobustScaler là unsupervised (không dùng target)
    # → fit trên toàn bộ không gây leakage
    print("\n[2a] Yeo-Johnson fit …")
    yj_path = os.path.join(model_dir, "yeo_johnson.pkl")
    if fit:
        _, yj_transformer = yj.fit_transform_yeo_johnson(df)
        yj.save_transformer(yj_transformer, yj_path)
    else:
        yj_transformer = yj.load_transformer(yj_path)

    print("\n[2b] Robust Scaler fit …")
    rs_path = os.path.join(model_dir, "robust_scaler.pkl")
    if fit:
        _, robust_scaler = rs.fit_transform_robust(df)
        rs.save_scaler(robust_scaler, rs_path)
    else:
        robust_scaler = rs.load_scaler(rs_path)

    # ── 3. Tách train / test TRƯỚC KHI fit target encoders ───────────────────
    print(f"\n[3] Train/test split ({1-test_size:.0%} / {test_size:.0%}) …")
    target_full = build_target(df)
    df_train, df_test, y_train, y_test = train_test_split(
        df, target_full,
        test_size=test_size,
        random_state=random_state,
        stratify=target_full,   # giữ tỉ lệ class
    )
    df_train = df_train.reset_index(drop=True)
    df_test  = df_test.reset_index(drop=True)
    y_train  = y_train.reset_index(drop=True)
    y_test   = y_test.reset_index(drop=True)
    print(f"    Train: {len(df_train):,}  |  Test: {len(df_test):,}")
    print(f"    >50K ratio — Train: {y_train.mean():.3f}  Test: {y_test.mean():.3f}")

    # ── 4. Áp dụng các bước KHÔNG dùng target lên cả hai split ───────────────
    print("\n[4] Non-target transforms (both splits) …")
    df_train = _apply_non_target_steps(df_train, country_mapping, yj_transformer, robust_scaler)
    df_test  = _apply_non_target_steps(df_test,  country_mapping, yj_transformer, robust_scaler)

    # ── 5. Target encoders: FIT chỉ trên TRAIN ───────────────────────────────
    # ★ Đây là điểm mấu chốt: encoder chỉ "thấy" y_train, không thấy y_test
    print("\n[5a] CatBoost encoder — fit on TRAIN only …")
    og_enc_path = os.path.join(model_dir, "occupation_catboost.pkl")
    if fit:
        enc_occ = og.fit_catboost_encoder(df_train, y_train)
        og.save_encoder(enc_occ, og_enc_path)
    else:
        enc_occ = og.load_encoder(og_enc_path)

    # Transform cả hai split bằng encoder đã fit từ train
    df_train = og.transform_catboost(df_train, enc_occ)
    df_test  = og.transform_catboost(df_test,  enc_occ)
    print(" Train encoded with train stats | Test encoded with same stats")

    print("\n[5b] LOO encoder (workclass) — fit on TRAIN only …")
    loo_path = os.path.join(model_dir, "workclass_loo.pkl")
    if fit:
        enc_loo = oe.fit_loo_encoder(df_train, y_train, col="workclass")
        oe.save_encoder(enc_loo, loo_path)
    else:
        enc_loo = oe.load_encoder(loo_path)

    df_train = oe.transform_loo(df_train, enc_loo)
    df_test  = oe.transform_loo(df_test,  enc_loo)
    print("    ✓ Train encoded with train stats | Test encoded with same stats")

    # ── 6. Tương tác kinh tế lượng & fairness ────────────────────────────────
    print("\n[6] Econometric & fairness interactions …")
    df_train = _apply_interactions(df_train)
    df_test  = _apply_interactions(df_test)

    # ── 7. Gắn target & lưu ──────────────────────────────────────────────────
    print("\n[7] Attaching target & saving …")
    df_train["income"] = y_train.values
    df_test["income"]  = y_test.values

    for path, data, label in [(train_out, df_train, "TRAIN"), (test_out, df_test, "TEST")]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data.to_csv(path, index=False)
        print(f"    [{label}] {data.shape} → {path}")
    print("=" * 60)
    return df_train, df_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",        default=DATA_IN)
    parser.add_argument("--mapping",      default=MAP_PATH)
    parser.add_argument("--train_out",    default=TRAIN_OUT)
    parser.add_argument("--test_out",     default=TEST_OUT)
    parser.add_argument("--model_dir",    default=MODEL_DIR)
    parser.add_argument("--test_size",    default=0.2, type=float)
    parser.add_argument("--random_state", default=42,  type=int)
    parser.add_argument("--infer",        action="store_true")
    args = parser.parse_args()

    run_pipeline(
        data_in=args.input,
        map_path=args.mapping,
        train_out=args.train_out,
        test_out=args.test_out,
        model_dir=args.model_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        fit=not args.infer,
    )
