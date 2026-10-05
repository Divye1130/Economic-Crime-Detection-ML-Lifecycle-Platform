from __future__ import annotations
from pathlib import Path
import json, joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, precision_score, recall_score, f1_score, confusion_matrix

NUMERIC=['amount_gbp','hour','account_age_days','device_age_days','transactions_last_1h','transactions_last_24h','distance_from_home_km','merchant_risk_score','card_present','international','new_payee','auth_failures_24h','amount_vs_customer_median','country_risk_score']
CATEGORICAL=['channel']
FEATURES=NUMERIC+CATEGORICAL

def _preprocessor(dense=False):
    return ColumnTransformer([('num',StandardScaler(),NUMERIC),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=not dense),CATEGORICAL)])

def chronological_split(df):
    n=len(df); a=int(n*.60); b=int(n*.80)
    return df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy()

def expected_review_loss(y,p,amounts,threshold,review_cost=4.50,miss_fraction=.90,fixed_miss_cost=50.0):
    pred=p>=threshold; yb=y.astype(bool)
    fp=(~yb)&pred; fn=yb&(~pred)
    return float(fp.sum()*review_cost + np.sum(amounts[fn]*miss_fraction + fixed_miss_cost))

def threshold_table(y,p,amounts,max_alert_rate=.015):
    rows=[]
    thresholds=np.unique(np.r_[np.linspace(.001,.08,100),np.linspace(.085,.50,40),np.linspace(.55,.995,30)])
    for t in thresholds:
        pred=p>=t
        rows.append({'threshold':float(t),'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'f1':float(f1_score(y,pred,zero_division=0)),'alert_rate':float(pred.mean()),'expected_review_loss_gbp':expected_review_loss(y,p,amounts,float(t))})
    table=pd.DataFrame(rows)
    feasible=table[table.alert_rate<=max_alert_rate]
    choice=(feasible if len(feasible) else table).sort_values(['expected_review_loss_gbp','threshold']).iloc[0]
    return table, choice

def metric_block(y,p,threshold):
    pred=p>=threshold
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'roc_auc':float(roc_auc_score(y,p)),'pr_auc':float(average_precision_score(y,p)),'brier':float(brier_score_loss(y,p)),'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'f1':float(f1_score(y,pred,zero_division=0)),'alert_rate':float(pred.mean()),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}

def train_and_evaluate(df,output_dir):
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    train,valid,test=chronological_split(df)
    Xtr,ytr=train[FEATURES],train.is_fraud.astype(int).to_numpy(); Xv,yv=valid[FEATURES],valid.is_fraud.astype(int).to_numpy(); Xt,yt=test[FEATURES],test.is_fraud.astype(int).to_numpy()
    logistic=Pipeline([('prep',_preprocessor(False)),('model',LogisticRegression(max_iter=1500,C=.45))]); logistic.fit(Xtr,ytr)
    hgb=Pipeline([('prep',_preprocessor(True)),('model',HistGradientBoostingClassifier(max_iter=180,learning_rate=.06,max_leaf_nodes=23,l2_regularization=1.0,random_state=42))])
    hgb.fit(Xtr,ytr)
    models={'Logistic Regression':logistic,'Gradient Boosting':hgb}; comp=[]
    for name,model in models.items():
        pv=model.predict_proba(Xv)[:,1]
        comp.append({'model':name,'validation_pr_auc':float(average_precision_score(yv,pv)),'validation_roc_auc':float(roc_auc_score(yv,pv)),'validation_brier':float(brier_score_loss(yv,pv))})
    comparison=pd.DataFrame(comp).sort_values('validation_pr_auc',ascending=False)
    chosen_name=comparison.iloc[0].model; chosen=models[chosen_name]
    pv=chosen.predict_proba(Xv)[:,1]; thresholds,selected=threshold_table(yv,pv,valid.amount_gbp.to_numpy()); threshold=float(selected.threshold)
    pt=chosen.predict_proba(Xt)[:,1]
    validation_metrics=metric_block(yv,pv,threshold); test_metrics=metric_block(yt,pt,threshold); test_metrics['expected_review_loss_gbp']=expected_review_loss(yt,pt,test.amount_gbp.to_numpy(),threshold)
    scored=test[['transaction_id','timestamp','amount_gbp','channel','is_fraud']].copy(); scored['fraud_probability']=pt; scored['alert']=(pt>=threshold).astype(int)
    scored['outcome']=np.select([(scored.alert==1)&(scored.is_fraud==1),(scored.alert==1)&(scored.is_fraud==0),(scored.alert==0)&(scored.is_fraud==1)],['True Positive','False Positive','False Negative'],default='True Negative')
    joblib.dump({'model':chosen,'threshold':threshold,'features':FEATURES},output_dir/'fraud_model.joblib')
    comparison.to_csv(output_dir/'model_comparison.csv',index=False); thresholds.to_csv(output_dir/'threshold_analysis.csv',index=False); scored.to_csv(output_dir/'test_scored_transactions.csv',index=False)
    result={'dataset':{'rows':int(len(df)),'frauds':int(df.is_fraud.sum()),'fraud_rate':float(df.is_fraud.mean()),'train_rows':int(len(train)),'validation_rows':int(len(valid)),'test_rows':int(len(test))},'selected_model':chosen_name,'selected_threshold':threshold,'validation':validation_metrics,'test':test_metrics}
    (output_dir/'metrics.json').write_text(json.dumps(result,indent=2))
    return result,comparison,thresholds,scored,(train,valid,test)
