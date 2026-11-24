# Task 5: TF-IDF ticket routing.
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from common import DATA, MODELS, REPORTS

df = pd.read_csv(DATA / 'tickets.csv')
print(df['category'].value_counts().to_string(), '\n')
print('Example ticket:', df.loc[0, 'text'], '->', df.loc[0, 'category'], '\n')

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['category'], test_size=0.2, stratify=df['category'], random_state=42)


def vectorizer():
    return TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 2),
                           min_df=2, sublinear_tf=True)


candidates = {'naive_bayes': MultinomialNB(alpha=0.5),
              'logreg': LogisticRegression(max_iter=2000, C=5.0)}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = {}
for name, clf in candidates.items():
    pipe = Pipeline([('tfidf', vectorizer()), ('clf', clf)])
    cv_scores[name] = round(float(cross_val_score(pipe, X_train, y_train, cv=cv,
                                                  scoring='f1_macro').mean()), 4)
    print(f'CV macro-F1 {name:<12} {cv_scores[name]:.4f}')

best_name = max(cv_scores, key=cv_scores.get)
model = Pipeline([('tfidf', vectorizer()), ('clf', candidates[best_name])]).fit(X_train, y_train)
pred = model.predict(X_test)
print(f"\nSelected: {best_name}   vocabulary size: {len(model.named_steps['tfidf'].vocabulary_)}")
print(classification_report(y_test, pred, digits=3))
labels = list(model.classes_)
print('Confusion matrix (rows=true, cols=pred):', labels)
print(confusion_matrix(y_test, pred, labels=labels))

terms = model.named_steps['tfidf'].get_feature_names_out()
clf = model.named_steps['clf']
weights = clf.coef_ if hasattr(clf, 'coef_') else clf.feature_log_prob_
print('\nMost predictive terms per class:')
for i, cls in enumerate(clf.classes_):
    top = np.argsort(weights[i])[::-1][:6]
    words = ', '.join(terms[top])
    print(f'  {cls:<10} {words}')

new_tickets = [
    ('I was billed twice this month and want a refund', 'billing'),
    ('The mobile app keeps crashing after the latest update', 'technical'),
    ('I cannot log in because my password reset email never came', 'account'),
    ('Where is my package? Tracking shows nothing for a week', 'shipping'),
]
preds = model.predict([t for t, _ in new_tickets])
print('\nNew tickets:')
out_new = []
for (text, exp), p in zip(new_tickets, preds):
    print(f'  [{p:<9}] (expected {exp:<9}) {text}')
    out_new.append({'text': text, 'predicted': str(p), 'expected': exp})

report = {'cv_macro_f1': cv_scores, 'selected': best_name,
          'test_accuracy': round(float(accuracy_score(y_test, pred)), 4),
          'test_macro_f1': round(float(f1_score(y_test, pred, average='macro')), 4),
          'new_tickets': out_new}
joblib.dump(model, MODELS / 'ticket_clf.joblib')
(REPORTS / 'ticket_metrics.json').write_text(json.dumps(report, indent=2))
print('\nwrote models/ticket_clf.joblib and reports/ticket_metrics.json')
