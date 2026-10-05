from src.data_builder import build_transactions

def test_data_shape_and_keys():
    df=build_transactions(5000,seed=7); assert len(df)==5000; assert df.transaction_id.is_unique; assert df.timestamp.is_monotonic_increasing

def test_feature_ranges():
    df=build_transactions(4000,seed=8); assert (df.amount_gbp>=0).all(); assert df.hour.between(0,23).all(); assert df.merchant_risk_score.between(0,1).all(); assert df.country_risk_score.between(0,1).all(); assert set(df.channel.unique()) <= {'POS','eCommerce','Mobile','ATM'}

def test_binary_target_and_rare_event_rate():
    df=build_transactions(20000,seed=42); assert set(df.is_fraud.unique()) <= {0,1}; assert 0.002 < df.is_fraud.mean() < 0.02
