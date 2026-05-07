import lightgbm as lgb

def build_model(params):
    return lgb.LGBMRegressor(**params, n_estimators=5000)