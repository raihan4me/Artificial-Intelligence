# Task 2: generate seeded synthetic datasets for the lab.
import numpy as np
import pandas as pd

from common import DATA

rng = np.random.default_rng(42)


def make_loans(n=5000):
    age = rng.integers(21, 70, n)
    income = rng.lognormal(mean=10.6, sigma=0.4, size=n).round(0)
    loan_amount = (rng.uniform(0.1, 0.8, n) * income).round(0)
    credit_score = np.clip(rng.normal(650, 70, n), 300, 850).round(0)
    years_employed = np.clip(rng.exponential(6, n), 0, 45).round(1)
    existing_debt = (rng.uniform(0, 0.6, n) * income).round(0)
    gender = rng.choice(['F', 'M'], n)
    region = rng.choice(['north', 'south', 'east', 'west'], n, p=[0.3, 0.3, 0.2, 0.2])

    dti = (existing_debt + loan_amount) / income
    logit = (-1.0 + 4.0 * (dti - 0.7) - 0.012 * (credit_score - 650)
             - 0.05 * (years_employed - 6) - 0.01 * (age - 45)
             + np.where(region == 'south', 0.3, 0.0))
    p = 1 / (1 + np.exp(-logit))
    default = (rng.uniform(size=n) < p).astype(int)

    # Historical labelling bias: some good female applicants recorded as defaults
    flip = (gender == 'F') & (default == 0) & (rng.uniform(size=n) < 0.10)
    default[flip] = 1

    df = pd.DataFrame({
        'applicant_id': [f'A{i:05d}' for i in range(n)],
        'age': age, 'income': income, 'loan_amount': loan_amount,
        'credit_score': credit_score, 'years_employed': years_employed,
        'existing_debt': existing_debt, 'gender': gender, 'region': region,
        'default': default,
    })
    for col, frac in [('income', 0.04), ('credit_score', 0.03), ('years_employed', 0.02)]:
        idx = rng.choice(n, int(n * frac), replace=False)
        df.loc[idx, col] = np.nan
    dupes = df.sample(40, random_state=7)           # simulated double submissions
    return pd.concat([df, dupes], ignore_index=True)


TICKET_PHRASES = {
    'billing': ['I was charged twice for my subscription', 'the invoice amount is wrong',
                'please refund the extra charge', 'my credit card was billed after I cancelled',
                'why did my monthly fee increase'],
    'technical': ['the app crashes when I open it', 'I get error 500 on the dashboard',
                  'the website is very slow today', 'sync is not working on my phone',
                  'the API returns a timeout'],
    'account': ['I forgot my password and cannot reset it', 'please change the email on my account',
                'I want to delete my account', 'my account was locked after failed logins',
                'how do I enable two factor authentication'],
    'shipping': ['my parcel has not arrived yet', 'the tracking number does not work',
                 'the package was delivered to the wrong address', 'the box arrived damaged',
                 'the courier never showed up'],
}
OPENERS = ['Hi team,', 'Hello,', 'Good morning,', 'Urgent:', '', 'Dear support,']
FILLERS = ['I have been a customer for years.', 'I tried again this morning.',
           'My order number is {n}.', '', 'This is really frustrating.']
CLOSERS = ['Thanks.', 'Please help.', 'Regards.', '', 'Can you fix this today?']


def make_tickets(n=2000):
    cats = list(TICKET_PHRASES)
    rows = []
    for i in range(n):
        cat = cats[i % len(cats)]
        parts = [str(rng.choice(OPENERS)), str(rng.choice(TICKET_PHRASES[cat])) + '.']
        if rng.uniform() < 0.10:                       # secondary topic mentioned
            other = str(rng.choice([c for c in cats if c != cat]))
            parts.append('Also, ' + str(rng.choice(TICKET_PHRASES[other])) + '.')
        parts.append(str(rng.choice(FILLERS)).replace('{n}', str(rng.integers(10000, 99999))))
        parts.append(str(rng.choice(CLOSERS)))
        label = cat
        if rng.uniform() < 0.04:                       # human labelling mistakes
            label = str(rng.choice([c for c in cats if c != cat]))
        rows.append({'ticket_id': f'T{i:05d}',
                     'text': ' '.join(p for p in parts if p),
                     'category': label})
    return pd.DataFrame(rows)


def make_customers(per_segment=300):
    # (visits mean, sd, basket mean, sd, tenure mean, sd)
    segs = {'bargain_hunters': (6, 1.0, 12, 3, 24, 6),
            'loyal_families': (8, 1.0, 35, 5, 60, 8),
            'occasional_big_spenders': (1.5, 0.5, 180, 25, 30, 8),
            'new_browsers': (2, 0.7, 20, 5, 3, 1.5)}
    frames = []
    for name, (v, vs, b, bs, t, ts) in segs.items():
        visits = np.clip(rng.normal(v, vs, per_segment), 0.2, None)
        basket = np.clip(rng.normal(b, bs, per_segment), 1, None)
        tenure = np.clip(rng.normal(t, ts, per_segment), 0, None)
        spend = visits * basket * rng.normal(1.0, 0.08, per_segment)
        frames.append(pd.DataFrame({'monthly_spend': spend.round(2),
                                    'visits_per_month': visits.round(2),
                                    'avg_basket': basket.round(2),
                                    'tenure_months': tenure.round(1),
                                    'true_segment': name}))
    df = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=1).reset_index(drop=True)
    df.insert(0, 'customer_id', [f'C{i:05d}' for i in range(len(df))])
    return df


if __name__ == '__main__':
    DATA.mkdir(exist_ok=True)
    for name, frame in [('loans.csv', make_loans()), ('tickets.csv', make_tickets()),
                        ('customers.csv', make_customers())]:
        frame.to_csv(DATA / name, index=False)
        print(f'wrote {name:<14} rows={len(frame)}')
