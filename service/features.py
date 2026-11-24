# Feature engineering shared by training and serving.
import numpy as np
import pandas as pd

ENGINEERED = ['debt_to_income', 'loan_to_income']


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    # Return a copy of X with engineered ratio features added.
    #   debt_to_income = (loan_amount + existing_debt) / income
    #   loan_to_income = loan_amount / income
    #   +inf / -inf (e.g. income == 0) must become NaN; missing income gives NaN ratios.
    #   The input DataFrame must NOT be modified in place.
    X = X.copy()
    X['debt_to_income'] = (X['loan_amount'] + X['existing_debt']) / X['income']
    X['loan_to_income'] = X['loan_amount'] / X['income']
    X[ENGINEERED] = X[ENGINEERED].replace([np.inf, -np.inf], np.nan)
    return X
