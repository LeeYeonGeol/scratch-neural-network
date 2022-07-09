import numpy as np
import pandas as pd
import neural_network
import data_preprocessing

# input
df = pd.read_csv("data/KBO_DATA.csv")

# preprocessing
df = data_preprocessing.preprocess(df)

# 정규화 안한 상태
# 뉴럴네트워크 생성
nn = neural_network.neuralnetwork()

# 학습
output = nn.train(df)


epochs = 20

