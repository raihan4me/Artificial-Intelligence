# Task 6: k-means customer segmentation with silhouette-based model selection.
import json

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from common import DATA, REPORTS

df = pd.read_csv(DATA / 'customers.csv')
features = ['monthly_spend', 'visits_per_month', 'avg_basket', 'tenure_months']
X = StandardScaler().fit_transform(df[features])     # k-means is distance based

scores = {}
print(f"{'k':>2} {'inertia':>10} {'silhouette':>10}")
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    scores[k] = {'inertia': round(float(km.inertia_), 1),
                 'silhouette': round(float(silhouette_score(X, km.labels_)), 4)}
    print(f"{k:>2} {scores[k]['inertia']:>10.1f} {scores[k]['silhouette']:>10.4f}")

best_k = max(scores, key=lambda k: scores[k]['silhouette'])
km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(X)
df['cluster'] = km.labels_

profile = df.groupby('cluster')[features].mean().round(1)
profile['size'] = df['cluster'].value_counts().sort_index()
print(f'\nBest k by silhouette: {best_k}\n\nCluster profiles (original units):')
print(profile)

ari = adjusted_rand_score(df['true_segment'], df['cluster'])
print(f'\nAdjusted Rand Index vs hidden true segments: {ari:.3f}')
print(pd.crosstab(df['true_segment'], df['cluster']))

pca = PCA(n_components=2).fit(X)
print('\nPCA explained variance ratio (2 components):', pca.explained_variance_ratio_.round(3),
      'total =', round(float(pca.explained_variance_ratio_.sum()), 3))

df.to_csv(DATA / 'customers_segmented.csv', index=False)

(REPORTS / 'segments.json').write_text(json.dumps({
    'scores': {str(k): v for k, v in scores.items()}, 'best_k': int(best_k),
    'ari': round(float(ari), 4), 'profile': profile.reset_index().to_dict(orient='records'),
}, indent=2))
print('wrote reports/segments.json')
