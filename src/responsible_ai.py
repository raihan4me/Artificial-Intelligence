# Task 7: fairness review, permutation importance and model card.
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from common import (COST_FN, COST_FP, MODELS, RAW_NUMERIC, REPORTS, best_threshold,
                    expected_cost, load_loans, make_preprocessor, split)
from features import ENGINEERED


def group_metrics(y_true, y_pred, groups):
    # Per-group fairness metrics. Inputs are 1-D numpy arrays of equal length;
    # y_pred is 1 = predicted default. Return a dict keyed by str(group) with:
    #   'n'             number of rows in the group (int)
    #   'approval_rate' fraction of the group with y_pred == 0 (float)
    #   'tpr'           among rows with y_true == 1, fraction with y_pred == 1
    #   'fpr'           among rows with y_true == 0, fraction with y_pred == 1
    # Use float('nan') for tpr/fpr if the group has no positives/negatives.
    out = {}
    for g in np.unique(groups):
        mask = groups == g
        yt, yp = y_true[mask], y_pred[mask]
        pos, neg = yt == 1, yt == 0
        out[str(g)] = {
            'n': int(mask.sum()),
            'approval_rate': float((yp == 0).mean()),
            'tpr': float((yp[pos] == 1).mean()) if pos.any() else float('nan'),
            'fpr': float((yp[neg] == 1).mean()) if neg.any() else float('nan'),
        }
    return out


def disparate_impact(metrics):
    rates = [m['approval_rate'] for m in metrics.values()]
    return min(rates) / max(rates) if max(rates) > 0 else float('nan')


def evaluate(label, model, X_test, y_test, groups, threshold):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    gm = group_metrics(y_test.to_numpy(), pred, groups.to_numpy())
    di = disparate_impact(gm)
    tprs = [m['tpr'] for m in gm.values()]
    res = {'threshold': threshold, 'test_auc': round(float(roc_auc_score(y_test, proba)), 4),
           'test_cost': expected_cost(y_test, proba, threshold),
           'groups': gm, 'disparate_impact': round(float(di), 4),
           'tpr_gap': round(float(max(tprs) - min(tprs)), 4)}
    print(f"\n=== {label} ===  AUC={res['test_auc']}  cost={res['test_cost']}  threshold={threshold}")
    for g, m in gm.items():
        print(f"  {g}: n={m['n']:<4} approval={m['approval_rate']:.3f} "
              f"TPR={m['tpr']:.3f} FPR={m['fpr']:.3f}")
    print(f"  disparate impact={res['disparate_impact']:.3f}  TPR gap={res['tpr_gap']:.3f}")
    return res


if __name__ == '__main__':
    df = load_loans()
    X_train, X_test, y_train, y_test = split(df)
    g_test = X_test['gender']
    print('Observed default rate in TEST labels by gender:',
          y_test.groupby(g_test).mean().round(3).to_dict())

    tuning = json.loads((REPORTS / 'tuning.json').read_text())
    original = joblib.load(MODELS / 'loan_model.joblib')
    res_orig = evaluate('original model (uses gender)', original, X_test, y_test, g_test,
                        tuning['threshold'])

    # Variant without the sensitive attribute: same estimator, new preprocessor
    X_train_f, X_test_f = X_train.drop(columns=['gender']), X_test.drop(columns=['gender'])
    fair = clone(original).set_params(pre=make_preprocessor(RAW_NUMERIC + ENGINEERED, ['region']))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = cross_val_predict(clone(fair), X_train_f, y_train, cv=cv, method='predict_proba')[:, 1]
    thr, _ = best_threshold(y_train.to_numpy(), oof)
    fair.fit(X_train_f, y_train)
    res_fair = evaluate('candidate model (gender excluded)', fair, X_test_f, y_test, g_test, thr)

    pi = permutation_importance(fair, X_test_f, y_test, scoring='roc_auc',
                                n_repeats=5, random_state=42)
    importance = sorted(
        [{'feature': f, 'mean': round(float(m), 4), 'std': round(float(s), 4)}
         for f, m, s in zip(X_test_f.columns, pi.importances_mean, pi.importances_std)],
        key=lambda d: -d['mean'])
    print('\nPermutation importance (drop in ROC AUC when a feature is shuffled):')
    for d in importance:
        print(f"  {d['feature']:<16} {d['mean']:.4f} +/- {d['std']:.4f}")

    meta = {'model_name': 'loan-default-risk', 'model_version': '1.0.0',
            'estimator': type(fair.named_steps['clf']).__name__,
            'features': list(X_train_f.columns), 'threshold': thr,
            'cost_fn': COST_FN, 'cost_fp': COST_FP,
            'test_auc': res_fair['test_auc'], 'disparate_impact': res_fair['disparate_impact'],
            'sklearn_version': sklearn.__version__,
            'trained_at': datetime.now(timezone.utc).isoformat(timespec='seconds')}
    joblib.dump(fair, MODELS / 'loan_model_fair.joblib')
    (MODELS / 'model_meta.json').write_text(json.dumps(meta, indent=2))
    (REPORTS / 'fairness.json').write_text(json.dumps(
        {'original': res_orig, 'fair': res_fair, 'importance': importance}, indent=2))

    top = chr(10).join('- ' + d['feature'] + ': ' + str(d['mean']) for d in importance[:4])
    feats = ', '.join(meta['features'])
    card = f"""# Model Card: {meta['model_name']} v{meta['model_version']}

## Intended use
Rank consumer-loan applications by probability of default. Applications with
probability >= {thr} are REFERRED to a human underwriter; others are approved.
Not for automated rejection without human review.

## Model
- Estimator: {meta['estimator']} in a scikit-learn Pipeline (median imputation,
  scaling, one-hot encoding, debt/loan-to-income ratios)
- Features: {feats}
- Excluded on purpose: gender (sensitive attribute)
- Threshold chosen on out-of-fold training predictions with cost FN={COST_FN}, FP={COST_FP}

## Evaluation (held-out test set, n={len(X_test)})
| Metric | Original (with gender) | Candidate (no gender) |
|---|---|---|
| ROC AUC | {res_orig['test_auc']} | {res_fair['test_auc']} |
| Business cost | {res_orig['test_cost']} | {res_fair['test_cost']} |
| Disparate impact (approval) | {res_orig['disparate_impact']} | {res_fair['disparate_impact']} |
| TPR gap | {res_orig['tpr_gap']} | {res_fair['tpr_gap']} |

## Most influential features (permutation importance)
{top}

## Known limitations and risks
- Training labels contain historical bias against female applicants; removing the
  gender column does not correct biased labels, and proxies may exist in real data.
- Trained on synthetic data; must be re-validated on production data before use.
- Monitor input drift and per-group approval rates after deployment.
"""
    (REPORTS / 'model_card.md').write_text(card)
    print('\nwrote models/loan_model_fair.joblib, models/model_meta.json,')
    print('      reports/fairness.json, reports/model_card.md')
