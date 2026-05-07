from data_loader import load_data
from feature_engineering import add_features
from config import FEATURES, LGB_PARAMS
from train import cross_validate

def main():

    train, test, full, y = load_data()
    full = add_features(full, train, y)

    train_rows = train.shape[0]

    X = full[FEATURES].fillna(0)
    X_train = X.iloc[:train_rows]

    model = cross_validate(LGB_PARAMS, X_train, y, FEATURES)

if __name__ == "__main__":
    main()