import pandas as pd
import numpy as np

def load_data():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    y = np.log1p(train["SalePrice"])
    train.drop("SalePrice", axis=1, inplace=True)

    full = pd.concat([train, test], axis=0).reset_index(drop=True)
    return train, test, full, y