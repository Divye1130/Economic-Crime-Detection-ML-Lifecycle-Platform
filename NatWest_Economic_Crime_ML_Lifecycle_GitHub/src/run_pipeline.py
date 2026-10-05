from pathlib import Path
from src.data_builder import build_transactions
from src.modeling import train_and_evaluate
from src.monitoring import create_monitoring_outputs
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'artifacts'; DATA=ROOT/'data'/'outputs'; OUT.mkdir(exist_ok=True); DATA.mkdir(parents=True,exist_ok=True)
if __name__=='__main__':
    df=build_transactions(); result,comparison,thresholds,scored,splits=train_and_evaluate(df,OUT); train,valid,test=splits; drift,weekly,summary=create_monitoring_outputs(train,test,scored,OUT)
    comparison.to_csv(DATA/'model_comparison.csv',index=False); thresholds.to_csv(DATA/'threshold_analysis.csv',index=False); drift.to_csv(DATA/'feature_drift.csv',index=False); weekly.to_csv(DATA/'weekly_monitoring.csv',index=False)
    print(result); print(summary)
