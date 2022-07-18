import numpy as np
import pandas as pd
from collections import defaultdict
# X 전처리 함수
def preprocess_X(X):
    db = defaultdict(list)
    n_db = defaultdict(list)
    for item in X.itertuples():
        if item[2] == 'LEAGUE':
            db[item[1]] = [item[3],item[4],item[5],item[6],item[7],item[8]]
        else:
            n_db['AVG'].append(item[3]/db[item[1]][0])
            n_db['OBP'].append(item[4]/db[item[1]][1])
            n_db['SLG'].append(item[5]/db[item[1]][2])
            n_db['ERA'].append(item[6]/db[item[1]][3])
            n_db['FIP'].append(item[7]/db[item[1]][4])
            n_db['WHIP'].append(item[8]/db[item[1]][5])
 
    return pd.DataFrame(n_db)

# y 전처리 함수
def preprocess_y(y):
    y.dropna(inplace=True)

    return y

# 스케일링 진행해주는 함수
def standard_X(X, **kwargs):
    mean, std = 0, 0
    if not kwargs:
        mean = X.mean(axis=0, skipna=False)
        std = X.std()
    else:
        mean = kwargs["mean"]
        std = kwargs["std"]
    for col in X.columns:
        X[col] = (X[col] - mean[col]) / std[col]

    return X, mean, std