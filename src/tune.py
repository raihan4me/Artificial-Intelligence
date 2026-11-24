# Task 4: GridSearchCV over model families + cost-based threshold selection.
import json

import joblib
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from common import (CATEGORICAL, MODELS, RAW_NUMERIC, REPORTS, best_threshold,
                    expected_cost, load_loans, make_preprocessor, split)
from features import ENGINEERED, add_features

X_train, X_test, y_train, y_test = split(load_loans())
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipe = Pipeline([
    ('features', FunctionTransformer(add_features)),
    ('pre', make_preprocessor(RAW_NUMERIC + ENGINEERED, CATEGORICAL)),
    ('clf', LogisticRegression(max_iter=1000)),
])
param_grid = [
    {'clf': [LogisticRegression(max_iter=1000)], 'clf__C': [0.01, 0.1, 1.0, 10.0]},
    {'clf': [RandomForestClassifier(n_estimators=200, random_state=42)],
     'clf__max_depth': [4, 8, 12], 'clf__min_samples_leaf': [5, 20]},
]
search = GridSearchCV(pipe, param_grid, scoring='roc_auc', cv=cv, n_jobs=2)
search.fit(X_train, y_train)


def describe(params):
    p = dict(params)
    name = type(p.pop('clf')).__name__
    return name + ' ' + ', '.join(f"{k.replace('clf__', '')}={v}" for k, v in p.items())


ranked = sorted(zip(search.cv_results_['params'], search.cv_results_['mean_test_score'],
                    search.cv_results_['std_test_score']), key=lambda t: -t[1])
print('Top 5 configurations (5-fold CV ROC AUC):')
for params, mean, std in ranked[:5]:
    print(f'  {mean:.4f} +/- {std:.4f}  {describe(params)}')

best = search.best_estimator_
print('\nSelected:', describe(search.best_params_))

# Threshold chosen on out-of-fold TRAINING predictions (never on the test set)
oof = cross_val_predict(clone(best), X_train, y_train, cv=cv,
                        method='predict_proba', n_jobs=2)[:, 1]
thr, _ = best_threshold(y_train.to_numpy(), oof)

proba = best.predict_proba(X_test)[:, 1]
baseline = json.loads((REPORTS / 'baseline_metrics.json').read_text())['logreg']['roc_auc']
report = {
    'best_model': describe(search.best_params_),
    'cv_auc': round(float(search.best_score_), 4),
    'test_auc': round(float(roc_auc_score(y_test, proba)), 4),
    'baseline_test_auc': baseline,
    'threshold': thr,
    'test_cost_at_0_5': expected_cost(y_test, proba, 0.5),
    'test_cost_at_threshold': expected_cost(y_test, proba, thr),
    'recall_at_0_5': round(float(recall_score(y_test, (proba >= 0.5).astype(int))), 4),
    'recall_at_threshold': round(float(recall_score(y_test, (proba >= thr).astype(int))), 4),
}
print(json.dumps(report, indent=2))
joblib.dump(best, MODELS / 'loan_model.joblib')
(REPORTS / 'tuning.json').write_text(json.dumps(report, indent=2))
print('wrote models/loan_model.joblib and reports/tuning.json')
