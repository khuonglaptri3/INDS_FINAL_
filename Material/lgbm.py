# =========================================================
# 0. IMPORT
# =========================================================
import os, joblib, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

warnings.filterwarnings("ignore")

# Fix tqdm bug
import os
os.environ["TQDM_DISABLE"] = "1"

# =========================================================
# 1. LOAD MODEL + DATA
# =========================================================
data_dir = r"C:\Users\lanph\OneDrive\Desktop\Introduction to Data Science\Final_Project\Adult_Project_Final_Term\Material\models" 
INPUT_BASE = r"C:\Users\lanph\OneDrive\Desktop\Introduction to Data Science\Final_Project\Adult_Project_Final_Term\data\processed" 
model = joblib.load(os.path.join(data_dir, "lgbm.pkl")) 
test = pd.read_csv(os.path.join(INPUT_BASE, "adult_features_test.csv"))


X_test = test.drop(columns=["income"])
y_test = test["income"].astype(int)

clf = model.named_steps["clf"]

print("Model:", type(clf))
print("Shape:", X_test.shape)

# =========================================================
# 2. SHAP
# =========================================================
print("\n================ SHAP ANALYSIS ================")

explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)
sv = shap_values[1] if isinstance(shap_values, list) else shap_values

# --- Feature importance
plt.figure(figsize=(10, 5))
shap.summary_plot(sv, X_test, plot_type="bar", max_display=15, show=False)
plt.tight_layout()
plt.show()

# --- Summary
plt.figure(figsize=(10, 6))
shap.summary_plot(sv, X_test, max_display=15, show=False)
plt.tight_layout()
plt.show()

# --- Dependence (normal)
pairs = [
    ("education-num", "age"),
    ("age", "hours-per-week"),
    ("occupation_group", "education-num")
]

for feat, interact in pairs:
    if feat in X_test.columns:
        plt.figure(figsize=(8, 5))
        shap.dependence_plot(feat, sv, X_test, interaction_index=interact, show=False)
        plt.tight_layout()
        plt.show()

# =========================================================
# 🔥 SHAP BIAS ANALYSIS (CÂU 3)
# =========================================================
print("\n--- SHAP BIAS ANALYSIS ---")

# Gender vs Education
if "sex_binary" in X_test.columns and "education-num" in X_test.columns:
    plt.figure(figsize=(8, 5))
    shap.dependence_plot(
        "sex_binary",
        sv,
        X_test,
        interaction_index="education-num",
        show=False
    )
    plt.title("Gender impact (controlled by Education)")
    plt.tight_layout()
    plt.show()

# Gender vs Working Hours
if "sex_binary" in X_test.columns and "hours-per-week" in X_test.columns:
    plt.figure(figsize=(8, 5))
    shap.dependence_plot(
        "sex_binary",
        sv,
        X_test,
        interaction_index="hours-per-week",
        show=False
    )
    plt.title("Gender impact (controlled by Working Hours)")
    plt.tight_layout()
    plt.show()

# Race vs Education
if "race_White" in X_test.columns and "education-num" in X_test.columns:
    plt.figure(figsize=(8, 5))
    shap.dependence_plot(
        "race_White",
        sv,
        X_test,
        interaction_index="education-num",
        show=False
    )
    plt.title("Race impact (controlled by Education)")
    plt.tight_layout()
    plt.show()

# --- Ranking
shap_importance = np.abs(sv).mean(axis=0)
feat_rank = pd.DataFrame({
    "feature": X_test.columns,
    "importance": shap_importance
}).sort_values(by="importance", ascending=False)

print("\nTop SHAP Features:")
print(feat_rank.head(10))

# --- Sensitive feature impact
for sensitive in ["sex_binary", "race_White"]:
    if sensitive in X_test.columns:
        idx = list(X_test.columns).index(sensitive)
        print(f"\n{sensitive}:")
        print(f"  Mean |SHAP| = {np.abs(sv[:, idx]).mean():.4f}")
        print(f"  Mean SHAP = {sv[:, idx].mean():.4f}")

# --- Local explanation
y_pred = model.predict(X_test)
wrong_idx_candidates = np.where((y_pred == 1) & (y_test.values == 0))[0]
wrong_idx = wrong_idx_candidates[0] if len(wrong_idx_candidates) > 0 else 0

print("\nExplaining instance index:", wrong_idx)

plt.figure(figsize=(10, 5))
shap.plots.waterfall(
    shap.Explanation(
        values=sv[wrong_idx],
        base_values=explainer.expected_value[1]
        if isinstance(explainer.expected_value, list)
        else explainer.expected_value,
        data=X_test.iloc[wrong_idx],
        feature_names=X_test.columns.tolist()
    ),
    max_display=10
)
plt.show()

# =========================================================
# 3. FAIRNESS
# =========================================================
print("\n================ FAIRNESS =====================")

from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference
)
from sklearn.metrics import accuracy_score, recall_score

y_pred = model.predict(X_test)

print("\n--- FULL DATA ---")
for feat in ["sex_binary", "race_White"]:
    if feat in X_test.columns:
        dpd = demographic_parity_difference(y_test, y_pred,
                                            sensitive_features=X_test[feat])
        dpr = demographic_parity_ratio(y_test, y_pred,
                                      sensitive_features=X_test[feat])
        eod = equalized_odds_difference(y_test, y_pred,
                                        sensitive_features=X_test[feat])

        print(f"\n[{feat}]")
        print(f"DPD = {dpd:.4f}")
        print(f"DPR = {dpr:.4f}")
        print(f"EOD = {eod:.4f}")

# =========================================================
# 4. LIME
# =========================================================
print("\n================ LIME =====================")

from lime.lime_tabular import LimeTabularExplainer

explainer_lime = LimeTabularExplainer(
    training_data=X_test.values,
    feature_names=X_test.columns.tolist(),
    class_names=["<=50K", ">50K"],
    mode="classification"
)

exp_lime = explainer_lime.explain_instance(
    X_test.iloc[wrong_idx].values,
    clf.predict_proba,
    num_features=10
)

print("\nLIME Explanation:")
print(exp_lime.as_list())

# =========================================================
# 5. DiCE
# =========================================================
print("\n================ DiCE =====================")

import dice_ml
from dice_ml import Dice

class NumericPredictWrapper:
    def __init__(self, estimator, cols):
        self.estimator = estimator
        self.cols = cols

    def _fix(self, X):
        X = pd.DataFrame(X, columns=self.cols)
        return X.apply(pd.to_numeric, errors="coerce")

    def predict_proba(self, X):
        return self.estimator.predict_proba(self._fix(X))

    def predict(self, X):
        return self.estimator.predict(self._fix(X))

X_dice = X_test.copy()

d = dice_ml.Data(
    dataframe=pd.concat([X_dice, y_test.rename("income")], axis=1),
    continuous_features=[
        "age", "education-num",
        "hours-per-week",
        "capital-gain", "capital-loss"
    ],
    outcome_name="income"
)

m = dice_ml.Model(
    model=NumericPredictWrapper(clf, X_test.columns.tolist()),
    backend="sklearn"
)

exp = Dice(d, m, method="genetic")

query = X_dice[y_test == 0].iloc[[0]]

try:
    cf = exp.generate_counterfactuals(
        query,
        total_CFs=3,
        desired_class="opposite",
        features_to_vary="all"
    )
    print("\nCF results:")
    cf.visualize_as_dataframe(show_only_changes=True)
except Exception as e:
    print("CF failed:", e)