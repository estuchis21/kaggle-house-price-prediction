import numpy as np
from sklearn.model_selection import KFold

def add_features(full, train, y):

    full["TotalSF"] = (
        full["TotalBsmtSF"].fillna(0) +
        full["1stFlrSF"].fillna(0) +
        full["2ndFlrSF"].fillna(0)
    )

    full["Qual_x_GrLiv"] = full["OverallQual"] * full["GrLivArea"]
    full["OverallQual_sq"] = full["OverallQual"] ** 2
    full["GrLivArea_sq"] = full["GrLivArea"] ** 2

    full["Age"] = full["YrSold"] - full["YearBuilt"]
    full["RemodAge"] = full["YrSold"] - full["YearRemodAdd"]

    full["TotalBaths"] = (
        full["FullBath"].fillna(0) +
        0.5 * full["HalfBath"].fillna(0) +
        full["BsmtFullBath"].fillna(0) +
        0.5 * full["BsmtHalfBath"].fillna(0)
    )

    full["GarageAge"] = full["YrSold"] - \
        full["GarageYrBlt"].fillna(full["YearBuilt"])

    # Ordinal mapping
    qual_map = {"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1}
    full["KitchenQual_num"] = full["KitchenQual"].map(qual_map)
    full["BsmtQual_num"] = full["BsmtQual"].map(qual_map)

    # ---------------------------
    # Target Encoding SIN LEAKAGE
    # ---------------------------
    train_rows = train.shape[0]
    full["Neighborhood_TE"] = np.nan  # 🔥 FIX dtype

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for tr_idx, val_idx in kf.split(train):
        means = y.iloc[tr_idx].groupby(
            train.iloc[tr_idx]["Neighborhood"]
        ).mean()

        full.loc[val_idx, "Neighborhood_TE"] = \
            train.iloc[val_idx]["Neighborhood"].map(means)

    # Test → usar medias globales
    global_means = y.groupby(train["Neighborhood"]).mean()

    full.loc[train_rows:, "Neighborhood_TE"] = \
        full.loc[train_rows:, "Neighborhood"].map(global_means)

    # Luxury Score
    full["LuxuryScore"] = (
        full["OverallQual"].rank(pct=True) +
        full["GrLivArea"].rank(pct=True) +
        full["Neighborhood_TE"].rank(pct=True)
    ) / 3

    return full