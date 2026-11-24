# Task 4: show the bias/variance trade-off with tree depth.
import json

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier

from common import CATEGORICAL, RAW_NUMERIC, REPORTS, load_loans, make_preprocessor, split
from features import ENGINEERED, add_features

X_train, X_test, y_train, y_test = split(load_loans())
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rows = []
print(f"{'depth':>6} {'train_auc':>10} {'val_auc':>8} {'gap':>6}")
for depth in [1, 2, 3, 4, 6, 8, 12, None]:
    pipe = Pipeline([
        ('features', FunctionTransformer(add_features)),
        ('pre', make_preprocessor(RAW_NUMERIC + ENGINEERED, CATEGORICAL)),
        ('clf', DecisionTreeClassifier(max_depth=depth, random_state=42)),
    ])
    res = cross_validate(pipe, X_train, y_train, cv=cv, scoring='roc_auc',
                         return_train_score=True)
    tr, va = res['train_score'].mean(), res['test_score'].mean()
    rows.append({'depth': depth, 'train_auc': round(tr, 4), 'val_auc': round(va, 4),
                 'gap': round(tr - va, 4)})
    print(f'{str(depth):>6} {tr:>10.3f} {va:>8.3f} {tr - va:>6.3f}')

best = max(rows, key=lambda r: r['val_auc'])
print(f"\nBest depth by validation AUC: {best['depth']}")
(REPORTS / 'overfit.json').write_text(json.dumps({'rows': rows, 'best': best}, indent=2))
