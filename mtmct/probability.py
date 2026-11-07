"""Supervised pair calibration. Validation records must be held out by recording."""
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


def fit(train, validation, output):
    """Rows: cost, label (0/1), recording. Never fit on validation rows."""
    if not train or not validation:
        raise ValueError('Training and independent validation pairs required')
    if {r['recording'] for r in train} & {r['recording'] for r in validation}:
        raise ValueError('Recording leakage between training and validation')
    def arrays(rows):
        x=np.array([r['cost'] for r in rows],dtype=float).reshape(-1,1)
        y=np.array([r['label'] for r in rows])
        if not np.isfinite(x).all() or not set(y).issubset({0,1}) or len(set(y))!=2:
            raise ValueError('Finite costs and both label classes required')
        return x,y
    x,y=arrays(train)
    model=LogisticRegression(random_state=0).fit(x,y)
    xv,yv=arrays(validation)
    predictions=model.predict_proba(xv)[:,1]
    result=dict(schema='mtmct-pair-logistic-v1',intercept=float(model.intercept_[0]),
                coefficient=float(model.coef_[0,0]),train_pairs=len(train),validation_pairs=len(validation),
                train_recordings=sorted({r['recording'] for r in train}),
                validation_recordings=sorted({r['recording'] for r in validation}),
                brier=float(brier_score_loss(yv,predictions)),log_loss=float(log_loss(yv,predictions)))
    if result['coefficient'] >= 0:
        raise ValueError('Costs do not support a decreasing match probability')
    Path(output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def predict(cost, model):
    if model.get('schema')!='mtmct-pair-logistic-v1' or not np.isfinite(cost):
        raise ValueError('Invalid calibration model or cost')
    return float(expit(model['intercept']+model['coefficient']*cost))
