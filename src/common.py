# Shared paths, column definitions and helpers for the loan-default tasks.
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
MODELS = ROOT / 'models'
REPORTS = ROOT / 'reports'

TARGET = 'default'
RAW_NUMERIC = ['age', 'income', 'loan_amount', 'credit_score',
               'years_employed', 'existing_debt']
CATEGORICAL = ['gender', 'region']

# Approving a future defaulter (false negative) costs 5x a needless referral (false positive).
COST_FN = 5.0
COST_FP = 1.0


def load_loans() -> pd.DataFrame:
    return pd.read_csv(DATA / 'loans_clean.csv')


def split(df, drop=('applicant_id',)):
    # Stratified 80/20 split. Same seed => same rows in every task.
    X = df.drop(columns=[TARGET, *drop])
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def make_preprocessor(numeric, categorical):
    # Imputation + scaling + one-hot encoding, fitted only on training folds.
    num = Pipeline([('impute', SimpleImputer(strategy='median')),
                    ('scale', StandardScaler())])
    cat = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    return ColumnTransformer([('num', num, list(numeric)),
                              ('cat', cat, list(categorical))])


def expected_cost(y_true, proba, threshold, cost_fn=COST_FN, cost_fp=COST_FP):
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    fn = int(((pred == 0) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    return cost_fn * fn + cost_fp * fp


def best_threshold(y_true, proba):
    # Scan thresholds 0.05..0.95 and return the one with the lowest cost.
    thresholds = np.round(np.linspace(0.05, 0.95, 91), 2)
    costs = [expected_cost(y_true, proba, t) for t in thresholds]
    i = int(np.argmin(costs))
    return float(thresholds[i]), float(costs[i])
