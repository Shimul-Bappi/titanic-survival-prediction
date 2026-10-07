"""Titanic survival prediction: Random Forest vs XGBoost.

Usage:
    python src/titanic_pipeline.py --data-dir data --out-dir outputs

Expects train.csv and test.csv from https://www.kaggle.com/competitions/titanic/data
inside --data-dir. Writes submission.csv (best model by mean CV accuracy),
submission_rf.csv and submission_xgb.csv to --out-dir.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

TITLE_MAPPING = {
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
    "Lady": "Royalty", "Countess": "Royalty", "Capt": "Officer",
    "Col": "Officer", "Major": "Officer", "Dr": "Officer",
    "Rev": "Officer", "Jonkheer": "Royalty", "Don": "Royalty",
    "Sir": "Royalty", "Dona": "Royalty",
}


def build_features(train: pd.DataFrame, test: pd.DataFrame):
    df = pd.concat([train, test], axis=0, sort=False).reset_index(drop=True)

    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False).replace(TITLE_MAPPING)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = np.where(df["FamilySize"] == 1, 1, 0)

    df["Age"] = df.groupby("Title")["Age"].transform(lambda x: x.fillna(x.median()))
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Fare"] = df["Fare"].fillna(df.groupby("Pclass")["Fare"].transform("median"))
    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

    train_c = df[df["Survived"].notnull()].copy()
    test_c = df[df["Survived"].isnull()].copy().drop(columns=["Survived"])

    X = pd.get_dummies(train_c.drop(columns=["Survived"]))
    y = train_c["Survived"].astype(int)
    X_test = pd.get_dummies(test_c)
    X, X_test = X.align(X_test, join="left", axis=1, fill_value=0)
    return X, y, X_test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    X, y, X_test = build_features(train, test)

    rf = RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_split=4, random_state=42)
    xgb = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, random_state=42, eval_metric="logloss")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_scores = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")
    xgb_scores = cross_val_score(xgb, X, y, cv=cv, scoring="accuracy")
    print(f"Random Forest CV accuracy: {rf_scores.mean():.4f} +/- {rf_scores.std(ddof=1):.4f}")
    print(f"XGBoost       CV accuracy: {xgb_scores.mean():.4f} +/- {xgb_scores.std(ddof=1):.4f}")

    rf.fit(X, y)
    xgb.fit(X, y)
    preds = {"rf": rf.predict(X_test), "xgb": xgb.predict(X_test)}

    for name, p in preds.items():
        pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": p}).to_csv(
            out_dir / f"submission_{name}.csv", index=False)

    best = "xgb" if xgb_scores.mean() >= rf_scores.mean() else "rf"
    pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": preds[best]}).to_csv(
        out_dir / "submission.csv", index=False)
    print(f"Wrote submission.csv from the {best.upper()} model to {out_dir}/")


if __name__ == "__main__":
    main()
