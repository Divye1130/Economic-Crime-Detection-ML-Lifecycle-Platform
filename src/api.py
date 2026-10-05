from pathlib import Path
import joblib, pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
MODEL_PATH=Path(__file__).resolve().parents[1]/'artifacts'/'fraud_model.joblib'; BUNDLE=joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
class Transaction(BaseModel):
    amount_gbp:float=Field(ge=0); hour:int=Field(ge=0,le=23); account_age_days:int=Field(ge=0); device_age_days:float=Field(ge=0); transactions_last_1h:int=Field(ge=0); transactions_last_24h:int=Field(ge=0); distance_from_home_km:float=Field(ge=0); merchant_risk_score:float=Field(ge=0,le=1); card_present:int=Field(ge=0,le=1); international:int=Field(ge=0,le=1); new_payee:int=Field(ge=0,le=1); auth_failures_24h:int=Field(ge=0); amount_vs_customer_median:float=Field(ge=0); channel:str; country_risk_score:float=Field(ge=0,le=1)
app=FastAPI(title='Economic Crime Risk Scoring API',version='1.0.0')
@app.get('/health')
def health(): return {'status':'ok','model_loaded':BUNDLE is not None}
@app.post('/score')
def score(transaction:Transaction):
    if BUNDLE is None: return {'error':'model artifact not found'}
    row=pd.DataFrame([transaction.model_dump()]); p=float(BUNDLE['model'].predict_proba(row[BUNDLE['features']])[:,1][0])
    return {'fraud_probability':round(p,6),'threshold':round(float(BUNDLE['threshold']),6),'alert':bool(p>=BUNDLE['threshold'])}
