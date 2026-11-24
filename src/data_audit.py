# Task 2: data-quality audit and de-duplication of the loan dataset.
import pandas as pd

from common import DATA, TARGET, RAW_NUMERIC

df = pd.read_csv(DATA / 'loans.csv')
print('Raw shape:', df.shape)
print('Duplicate rows:', int(df.duplicated().sum()))

clean = df.drop_duplicates().reset_index(drop=True)
print('Shape after de-duplication:', clean.shape)

missing = clean.isna().sum()
missing = missing[missing > 0]
print('\nMissing values (after de-duplication):')
print(pd.DataFrame({'count': missing, 'pct': (missing / len(clean) * 100).round(1)}))

print(f'\nOverall default rate: {clean[TARGET].mean():.3f}')
print('\nDefault rate by gender:')
print(clean.groupby('gender')[TARGET].mean().round(3))
print('\nCorrelation of numeric features with the target:')
print(clean[RAW_NUMERIC + [TARGET]].corr()[TARGET].drop(TARGET).round(3).sort_values())

clean.to_csv(DATA / 'loans_clean.csv', index=False)
print('\nwrote data/loans_clean.csv')
