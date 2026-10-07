# Titanic Survival Prediction: Random Forest vs XGBoost

Solution for the Kaggle competition [Titanic – Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic).
The pipeline engineers passenger features, imputes missing values by group, and compares a Random Forest and an XGBoost classifier with the same stratified 5-fold cross-validation.

## Results (5-fold stratified CV accuracy)

| Model | Mean accuracy | Std across folds |
|---|---|---|
| Random Forest (150 trees, depth 6) | 0.8305 | 0.0084 |
| XGBoost (150 trees, depth 4, lr 0.05) | 0.8417 | 0.0140 |

The executed notebook's final model (XGBoost) predicts 36.8 % of test passengers survive (training rate ≈ 38 %), and both models agree on 94 % of test predictions.

The 1.1-point gap is within noise for 5 folds (paired t ≈ 1.4), so the models are effectively comparable. The final `submission.csv` uses the model with the higher mean CV accuracy. Public leaderboard scores are typically a few points below CV accuracy because the test set has only 418 passengers.

## Method

1. **Features:** `Title` extracted from `Name` (rare titles grouped into `Officer` / `Royalty`), `FamilySize = SibSp + Parch + 1`, `IsAlone`.
2. **Imputation:** `Age` by median of `Title`, `Fare` by median of `Pclass`, `Embarked` by mode. `Cabin` (~77 % missing), `Ticket`, `Name` and `PassengerId` are dropped.
3. **Encoding:** one-hot encoding of `Sex`, `Embarked`, `Title`; train/test columns aligned.
4. **Models:** Random Forest and XGBoost, evaluated on identical folds.
5. **Interpretation:** Sex and Title carry most of the signal (≈ 56 % of Random Forest importance, ≈ 73 % of XGBoost importance), followed by `Pclass` and `Fare`.

The notebook contains an interpretation cell after every step.

## Repository structure

```
.
├── notebooks/titanic-passenger-survival-prediction.ipynb   # full walkthrough with interpretations
├── src/titanic_pipeline.py                                 # same pipeline as a script
├── data/                                                   # put train.csv / test.csv here (not tracked)
├── outputs/                                                # generated submissions (not tracked)
├── requirements.txt
└── LICENSE
```

## Getting started

```bash
git clone https://github.com/Shimul-Bappi/titanic-survival-prediction.git
cd titanic-survival-prediction
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download `train.csv` and `test.csv` from the [competition data page](https://www.kaggle.com/competitions/titanic/data) into `data/` (the files are not included, per Kaggle's rules).

**Run the script**

```bash
python src/titanic_pipeline.py --data-dir data --out-dir outputs
```

This writes `submission.csv` (best model), `submission_rf.csv` and `submission_xgb.csv` to `outputs/`.

**Or run the notebook**

On Kaggle, import the notebook and use *Save & Run All*. Locally, open it with Jupyter and change the two `pd.read_csv` paths in Step 1 from `/kaggle/input/competitions/titanic/` to `data/`.

## Possible improvements

- Use `Cabin` (deck letter / has-cabin flag) and `Ticket` (shared-ticket group size).
- Tune hyperparameters with repeated stratified CV.
- Add a logistic regression baseline.

## Author

**Shimul Bappi**

- GitHub: [Shimul-Bappi](https://github.com/Shimul-Bappi)
- LinkedIn: [shimul-bappi](https://www.linkedin.com/in/shimul-bappi/)
- Portfolio: [shimul-bappi.github.io](https://shimul-bappi.github.io/Shimul_Bappi-portfolio/)

## License

MIT, see [LICENSE](LICENSE).
