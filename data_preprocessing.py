import numpy as np
import pandas as pd
from collections import defaultdict
# 인풋모양으로 바꿔주는 함수
def preprocess(df):
    db = defaultdict(list)
    n_db = defaultdict(list)
    for item in df.itertuples():
        if item[2] == 'LEAGUE':
            db[item[1]] = [item[3],item[4],item[5],item[6],item[7],item[8]]
        else:
            # n_db['SEASON'].append(item[1])
            # n_db['TEAM'].append(item[2])
            n_db['AVG'].append(item[3]/db[item[1]][0])
            n_db['OBP'].append(item[4]/db[item[1]][1])
            n_db['SLG'].append(item[5]/db[item[1]][2])
            n_db['ERA'].append(item[6]/db[item[1]][3])
            n_db['FIP'].append(item[7]/db[item[1]][4])
            n_db['WHIP'].append(item[8]/db[item[1]][5])
            n_db['GRADE'].append(item[9])
 
    return pd.DataFrame(n_db)

# 정규화진행해주는 함수
def normalization_df(df):

    return df