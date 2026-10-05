import pandas as pd
from src.monitoring import population_stability_index, drift_status

def test_psi_identical_is_zero():
    s=pd.Series(range(100)); assert population_stability_index(s,s) < 1e-9

def test_psi_detects_shift():
    a=pd.Series(range(1000)); b=pd.Series(range(500,1500)); assert population_stability_index(a,b) > .1

def test_drift_status_thresholds():
    assert drift_status(.05)=='Stable'; assert drift_status(.15)=='Watch'; assert drift_status(.30)=='Material'
