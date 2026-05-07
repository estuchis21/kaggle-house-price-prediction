import mlflow
import mlflow.lightgbm
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error, r2_score
import lightgbm as lgb

def cross_validate(params, X, y, features):

    mlflow.set_experiment("house_prices_experiment")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    with mlflow.start_run():

        mlflow.log_params(params)
        mlflow.log_param("n_features", len(features))

        rmsle_scores = []
        r2_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):

            model = lgb.LGBMRegressor(**params, n_estimators=5000)

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(200)]
            )

            preds = model.predict(X_val)

            rmsle = np.sqrt(
                mean_squared_log_error(
                    np.expm1(y_val),
                    np.expm1(preds)
                )
            )

            r2 = r2_score(y_val, preds)

            rmsle_scores.append(rmsle)
            r2_scores.append(r2)

            mlflow.log_metric(f"rmsle_fold_{fold}", rmsle)
            mlflow.log_metric(f"r2_fold_{fold}", r2)

        mlflow.log_metric("rmsle_mean", np.mean(rmsle_scores))
        mlflow.log_metric("r2_mean", np.mean(r2_scores))

        # 🔥 Reentrenamiento final en TODO el dataset
        final_model = lgb.LGBMRegressor(**params, n_estimators=5000)
        final_model.fit(X, y)

        mlflow.lightgbm.log_model(final_model, "model")

        print("Mean RMSLE:", np.mean(rmsle_scores))
        print("Mean R2:", np.mean(r2_scores))

    return final_model