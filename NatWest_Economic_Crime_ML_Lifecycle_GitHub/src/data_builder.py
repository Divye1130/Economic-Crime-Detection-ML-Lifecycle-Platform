from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.special import expit

FEATURES = [
    'amount_gbp','hour','account_age_days','device_age_days','transactions_last_1h',
    'transactions_last_24h','distance_from_home_km','merchant_risk_score','card_present',
    'international','new_payee','auth_failures_24h','amount_vs_customer_median','channel',
    'country_risk_score'
]

def build_transactions(n: int = 60_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    day = np.sort(rng.integers(0, 180, n))
    late = day >= 120
    amount = np.exp(rng.normal(3.4, 1.0, n)).clip(0.5, 5_000)
    hour = rng.integers(0, 24, n)
    account_age = rng.integers(10, 4_000, n)
    device_age = rng.exponential(200, n)
    device_age[late] = rng.exponential(105, int(late.sum()))
    device_age = device_age.clip(0, 1_500)
    vel1 = rng.poisson(0.5, n)
    vel24 = rng.poisson(3.5, n)
    distance = rng.exponential(35, n)
    merchant_risk = rng.beta(2, 7, n)
    card_present = rng.binomial(1, 0.60, n)
    international = rng.binomial(1, 0.08, n)
    new_payee = rng.binomial(1, np.where(late, 0.13, 0.085))
    auth_failures = rng.poisson(0.15, n)
    amount_ratio = np.exp(rng.normal(0, 0.65, n))
    channel = np.empty(n, dtype=object)
    channel[~late] = rng.choice(['POS','eCommerce','Mobile','ATM'], (~late).sum(), p=[0.46,0.28,0.19,0.07])
    channel[late] = rng.choice(['POS','eCommerce','Mobile','ATM'], late.sum(), p=[0.34,0.30,0.29,0.07])
    country_risk = rng.beta(1.5, 6, n)
    country_risk[late] = rng.beta(1.9, 5.4, int(late.sum()))

    risk_signal = (
        0.00035*amount + 1.20*new_payee + 1.10*international + 0.75*(1-card_present)
        + 1.50*(device_age < 3) + 1.00*(auth_failures >= 2) + 0.85*(vel1 >= 3)
        + 0.65*(vel24 >= 10) + 0.80*(distance > 250) + 2.40*merchant_risk
        + 1.00*(amount_ratio > 3) + 0.45*((hour < 5) | (hour > 22))
        + 0.35*(channel == 'eCommerce') + 1.30*(new_payee.astype(bool) & international.astype(bool))
        + 1.10*((device_age < 3) & (auth_failures >= 1)) + 1.70*country_risk
    )
    behaviour_shift = late*(0.55*(channel == 'Mobile') + 0.80*(device_age < 10) + 0.40*new_payee - 0.25*(amount > 1000))
    logit = -9.10 + 1.50*risk_signal + 1.50*behaviour_shift
    fraud_probability = expit(logit)
    is_fraud = rng.binomial(1, fraud_probability)

    start = pd.Timestamp('2026-01-01')
    seconds = day*86_400 + rng.integers(0, 86_400, n)
    timestamp = start + pd.to_timedelta(seconds, unit='s')

    return pd.DataFrame({
        'transaction_id':[f'TX{i:07d}' for i in range(1, n+1)],
        'timestamp':timestamp,'amount_gbp':amount.round(2),'hour':hour,
        'account_age_days':account_age,'device_age_days':device_age.round(2),
        'transactions_last_1h':vel1,'transactions_last_24h':vel24,
        'distance_from_home_km':distance.round(2),'merchant_risk_score':merchant_risk.round(5),
        'card_present':card_present,'international':international,'new_payee':new_payee,
        'auth_failures_24h':auth_failures,'amount_vs_customer_median':amount_ratio.round(4),
        'channel':channel,'country_risk_score':country_risk.round(5),'is_fraud':is_fraud,
    }).sort_values('timestamp').reset_index(drop=True)
