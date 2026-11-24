# Loan default risk scoring API.
import json
import os
from pathlib import Path
from typing import List, Literal

import features  # noqa: F401  (needed so the pickled pipeline can be loaded)
import joblib
import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_DIR = Path(os.environ.get('MODEL_DIR', '/app/models'))
MODEL = joblib.load(MODEL_DIR / 'loan_model_fair.joblib')
META = json.loads((MODEL_DIR / 'model_meta.json').read_text())
THRESHOLD = float(META['threshold'])
MAX_BATCH = 100

trained = META['sklearn_version']
if trained != sklearn.__version__:
    raise RuntimeError(f'model trained with sklearn {trained}, runtime has {sklearn.__version__}')


class Applicant(BaseModel):
    age: int = Field(..., ge=18, le=100)
    income: float = Field(..., gt=0)
    loan_amount: float = Field(..., gt=0)
    credit_score: float = Field(..., ge=300, le=850)
    years_employed: float = Field(..., ge=0, le=60)
    existing_debt: float = Field(..., ge=0)
    region: Literal['north', 'south', 'east', 'west']


app = FastAPI(title='Loan default risk API', version=META['model_version'])


def score(applicants: List[Applicant]):
    X = pd.DataFrame([a.model_dump() for a in applicants])[META['features']]
    proba = MODEL.predict_proba(X)[:, 1]
    return [{'default_probability': round(float(p), 4), 'threshold': THRESHOLD,
             'decision': 'refer' if p >= THRESHOLD else 'approve',
             'model_version': META['model_version']} for p in proba]


@app.get('/health')
def health():
    return {'status': 'ok', 'model': META['model_name'], 'model_version': META['model_version'],
            'estimator': META['estimator'], 'sklearn_version': sklearn.__version__}


@app.post('/predict')
def predict(applicant: Applicant):
    return score([applicant])[0]


@app.post('/predict/batch')
def predict_batch(applicants: List[Applicant]):
    if not applicants or len(applicants) > MAX_BATCH:
        raise HTTPException(status_code=400, detail=f'batch size must be 1..{MAX_BATCH}')
    return {'results': score(applicants)}
