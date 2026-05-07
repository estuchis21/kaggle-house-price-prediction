FEATURES = [
    'OverallQual','OverallQual_sq',
    'GrLivArea','GrLivArea_sq','Qual_x_GrLiv',
    'GarageCars','GarageArea',
    'TotalBsmtSF','1stFlrSF','2ndFlrSF','TotalSF',
    'FullBath','TotRmsAbvGrd',
    'KitchenQual_num','BsmtQual_num',
    'Age','RemodAge','TotalBaths','GarageAge',
    'Neighborhood_TE','LuxuryScore'
]

LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.01,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
    "seed": 42
}