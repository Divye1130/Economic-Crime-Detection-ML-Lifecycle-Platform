import json
from pathlib import Path
import joblib
import pandas as pd
from src.api import score, Transaction

ROOT=Path(__file__).resolve().parents[1]

def test_metrics_artifact_is_complete():
    m=json.loads((ROOT/'artifacts'/'metrics.json').read_text())
    assert m['dataset']['rows']==60000
    assert 0 < m['selected_threshold'] < 1
    assert m['test']['pr_auc'] > 0
    assert m['test']['roc_auc'] > 0.5

def test_model_bundle_has_required_metadata():
    b=joblib.load(ROOT/'artifacts'/'fraud_model.joblib')
    assert {'model','threshold','features'} <= set(b)
    assert len(b['features'])==15

def test_api_scores_valid_transaction():
    t=Transaction(amount_gbp=120,hour=2,account_age_days=700,device_age_days=1,transactions_last_1h=4,transactions_last_24h=12,distance_from_home_km=280,merchant_risk_score=.7,card_present=0,international=1,new_payee=1,auth_failures_24h=2,amount_vs_customer_median=4.0,channel='Mobile',country_risk_score=.6)
    result=score(t)
    assert 0 <= result['fraud_probability'] <= 1
    assert isinstance(result['alert'], bool)

def test_monitoring_outputs_reconcile():
    drift=pd.read_csv(ROOT/'artifacts'/'feature_drift.csv')
    weekly=pd.read_csv(ROOT/'artifacts'/'weekly_monitoring.csv')
    assert len(drift)==6
    assert weekly.transactions.sum()==12000
    assert set(drift.status) <= {'Stable','Watch','Material'}

def test_threshold_curve_contains_selected_region():
    t=pd.read_csv(ROOT/'artifacts'/'threshold_analysis.csv')
    assert t.threshold.min() <= .01
    assert t.threshold.max() >= .95
    assert t.alert_rate.between(0,1).all()
