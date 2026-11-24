# Task 3: baseline vs logistic regression on raw features.
import json

import joblib
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline

from common import CATEGORICAL, MODELS, RAW_NUMERIC, REPORTS, load_loans, make_preprocessor, split

df = load_loans()
X_train, X_test, y_train, y_test = split(df)
print(f'train={len(X_train)} test={len(X_test)} positive rate train={y_train.mean():.3f}')

pre = make_preprocessor(RAW_NUMERIC, CATEGORICAL)
candidates = {
    'dummy': DummyClassifier(strategy='most_frequent'),
    'logreg': LogisticRegression(max_iter=1000),
}
results = {}
for name, est in candidates.items():
    pipe = Pipeline([('pre', clone(pre)), ('clf', est)])
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = pipe.predict(X_test)
    results[name] = {
        'accuracy': round(accuracy_score(y_test, pred), 4),
        'precision': round(precision_score(y_test, pred, zero_division=0), 4),
        'recall': round(recall_score(y_test, pred), 4),
        'f1': round(f1_score(y_test, pred, zero_division=0), 4),
        'roc_auc': round(roc_auc_score(y_test, proba), 4),
    }
    print(f'\n=== {name} ===')
    print('confusion matrix [[TN FP] [FN TP]]:\n', confusion_matrix(y_test, pred))
    print(results[name])
    if name == 'logreg':
        print(classification_report(y_test, pred, digits=3))
        joblib.dump(pipe, MODELS / 'baseline_logreg.joblib')

(REPORTS / 'baseline_metrics.json').write_text(json.dumps(results, indent=2))
print('wrote reports/baseline_metrics.json')
