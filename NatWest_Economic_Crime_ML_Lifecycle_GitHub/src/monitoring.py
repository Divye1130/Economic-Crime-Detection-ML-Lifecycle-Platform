from pathlib import Path
import json, numpy as np, pandas as pd
NUMERIC_MONITOR=['amount_gbp','device_age_days','transactions_last_24h','merchant_risk_score','amount_vs_customer_median','country_risk_score']

def population_stability_index(reference,current,bins=10):
    ref=pd.to_numeric(reference,errors='coerce').dropna().to_numpy(); cur=pd.to_numeric(current,errors='coerce').dropna().to_numpy(); edges=np.unique(np.quantile(ref,np.linspace(0,1,bins+1)))
    if len(edges)<3: return 0.0
    edges[0],edges[-1]=-np.inf,np.inf
    r=np.histogram(ref,bins=edges)[0]/max(len(ref),1); c=np.histogram(cur,bins=edges)[0]/max(len(cur),1); r=np.clip(r,1e-6,None); c=np.clip(c,1e-6,None)
    return float(np.sum((c-r)*np.log(c/r)))

def drift_status(psi):
    return 'Material' if psi>=.25 else ('Watch' if psi>=.10 else 'Stable')

def create_monitoring_outputs(train,test,scored_test,output_dir):
    output_dir=Path(output_dir); rows=[]
    for f in NUMERIC_MONITOR:
        psi=population_stability_index(train[f],test[f]); rows.append({'feature':f,'psi':psi,'status':drift_status(psi)})
    drift=pd.DataFrame(rows).sort_values('psi',ascending=False); drift.to_csv(output_dir/'feature_drift.csv',index=False)
    weekly=scored_test.copy(); weekly['week']=pd.to_datetime(weekly.timestamp).dt.to_period('W').astype(str); weekly=weekly.groupby('week',as_index=False).agg(transactions=('transaction_id','count'),fraud_rate=('is_fraud','mean'),alert_rate=('alert','mean'),mean_score=('fraud_probability','mean')); weekly.to_csv(output_dir/'weekly_monitoring.csv',index=False)
    summary={'max_psi':float(drift.psi.max()),'material_features':int((drift.status=='Material').sum()),'watch_features':int((drift.status=='Watch').sum())}; (output_dir/'monitoring_summary.json').write_text(json.dumps(summary,indent=2)); return drift,weekly,summary
