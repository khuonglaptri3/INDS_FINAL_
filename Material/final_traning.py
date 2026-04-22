import os, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, learning_curve
from sklearn.preprocessing import StandardScaler

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score,
    balanced_accuracy_score, matthews_corrcoef,
    classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, precision_recall_curve
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# ================= PATH =================
INPUT_BASE = "/kaggle/input/datasets/khngtrnnh/processed-dataset"
TRAIN_CSV = os.path.join(INPUT_BASE, "adult_features_train.csv")
TEST_CSV  = os.path.join(INPUT_BASE, "adult_features_test.csv")
MODEL_DIR = "/kaggle/working/models"
TARGET_COL = "income"

# ================= LOAD DATA =================
def load_data():
    train = pd.read_csv(TRAIN_CSV)
    test  = pd.read_csv(TEST_CSV)

    X_train = train.drop(columns=[TARGET_COL])
    y_train = train[TARGET_COL].astype(int)

    X_test  = test.drop(columns=[TARGET_COL])
    y_test  = test[TARGET_COL].astype(int)

    print("Train:",train.shape," Test:",test.shape)
    print("Positive rate:",y_train.mean())

    return X_train,y_train,X_test,y_test

# ================= MODELS =================
MODELS = {

"lr": Pipeline([
    ("smote", SMOTE(random_state=42)),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        penalty="l2",
        C=1,
        solver="liblinear",
        max_iter=2000
    ))
]),

"rf": Pipeline([
    ("smote", SMOTE(random_state=42)),
    ("clf", RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    ))
]),

"xgb": Pipeline([
    ("smote", SMOTE(random_state=42)),
    ("clf", XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    ))
]),

"lgbm": Pipeline([
    ("smote", SMOTE(random_state=42)),
    ("clf", LGBMClassifier(
        n_estimators=300,
        max_depth=8,
        num_leaves=63,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_gain_to_split=0.01,
        random_state=42,
        verbose=-1
    ))
]),
}
# ================= CV =================
def cross_validate_model(model,X,y):
    cv = StratifiedKFold(5,shuffle=True,random_state=42)

    scores = cross_validate(
        model,X,y,cv=cv,
        scoring=["accuracy","roc_auc","f1"],
        n_jobs=-1
    )

    print("CV ROC-AUC:",scores["test_roc_auc"].mean())
    print("CV F1:",scores["test_f1"].mean())

# ================= EVALUATE =================
def evaluate(model,X_test,y_test):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:,1]

    print(classification_report(y_test,pred))

    return {
        "Accuracy": accuracy_score(y_test, pred),
        "Balanced Accuracy": balanced_accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1-Score": f1_score(y_test, pred),
        "AUC": roc_auc_score(y_test, prob),
        "MCC": matthews_corrcoef(y_test, pred)
    }

# ================= CONFUSION =================
def plot_confusion(model,X_test,y_test,name):
    cm = confusion_matrix(y_test, model.predict(X_test))
    ConfusionMatrixDisplay(cm).plot()
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

# ================= LEARNING CURVE =================
def plot_learning_curve(model, X, y, name):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        cv=5, scoring="f1", n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    gap = train_mean[-1] - val_mean[-1]

    plt.figure()
    plt.plot(train_sizes, train_mean, label="Train")
    plt.plot(train_sizes, val_mean, label="Validation")
    plt.title(f"Learning Curve - {name}")
    plt.legend()
    plt.show()

    print(f"{name} Overfitting Gap:", round(gap,4))
    return gap

# ================= ROC =================
def plot_roc(models,X_test,y_test):
    plt.figure()
    for name,model in models.items():
        prob = model.predict_proba(X_test)[:,1]
        fpr,tpr,_ = roc_curve(y_test,prob)
        auc = roc_auc_score(y_test,prob)
        plt.plot(fpr,tpr,label=f"{name} (AUC={auc:.3f})")

    plt.plot([0,1],[0,1],'--')
    plt.legend()
    plt.title("ROC Curve")
    plt.show()

# ================= PR =================
def plot_pr(models,X_test,y_test):
    plt.figure()
    for name,model in models.items():
        prob = model.predict_proba(X_test)[:,1]
        p,r,_ = precision_recall_curve(y_test,prob)
        plt.plot(r,p,label=name)

    plt.legend()
    plt.title("Precision-Recall Curve")
    plt.show()

# ================= METRIC COMPARE =================
def plot_compare(df):
    df.set_index("Model")[["F1-Score","Precision","Recall","Balanced Accuracy","AUC","MCC"]].plot(kind="bar")
    plt.title("Model Comparison")
    plt.show()

# ================= MAIN =================
def run():
    X_train,y_train,X_test,y_test = load_data()
    os.makedirs(MODEL_DIR,exist_ok=True)

    results=[]
    trained={}

    for name,model in MODELS.items():
        print("\n====",name.upper(),"====")

        cross_validate_model(model,X_train,y_train)

        model.fit(X_train,y_train)
        trained[name]=model

        metrics = evaluate(model,X_test,y_test)
        gap = plot_learning_curve(model,X_train,y_train,name)

        plot_confusion(model,X_test,y_test,name)

        metrics["Model"]=name
        metrics["Overfitting Gap"]=gap

        results.append(metrics)

        joblib.dump(model,f"{MODEL_DIR}/{name}.pkl")

    df = pd.DataFrame(results)

    print("\nFINAL TABLE")
    print(df.sort_values("F1-Score",ascending=False).to_string(index=False))

    plot_compare(df)
    plot_roc(trained,X_test,y_test)
    plot_pr(trained,X_test,y_test)

run()