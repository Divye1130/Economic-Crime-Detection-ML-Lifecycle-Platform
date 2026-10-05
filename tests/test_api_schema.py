from src.api import Transaction

def test_transaction_schema_accepts_valid_input():
    t=Transaction(amount_gbp=25,hour=12,account_age_days=1000,device_age_days=50,transactions_last_1h=0,transactions_last_24h=2,distance_from_home_km=3,merchant_risk_score=.1,card_present=1,international=0,new_payee=0,auth_failures_24h=0,amount_vs_customer_median=1.2,channel='POS',country_risk_score=.1); assert t.amount_gbp==25
