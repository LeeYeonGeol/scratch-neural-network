import numpy as np
import pandas as pd
import neural_network
import data_preprocessing

# input
data = pd.read_csv("data/KBO_DATA.csv")

# preprocessing
X, y = data.drop('GRADE', axis=1), data['GRADE']
X = data_preprocessing.preprocess_X(X)
y = data_preprocessing.preprocess_y(y)
# 스케일링
X, mean, std = data_preprocessing.standard_X(X)

# 뉴럴네트워크 생성
nn = neural_network.neuralnetwork()

# 학습
learning_rate = 0.01

epochs = 829
output = nn.train(X, y, learning_rate, epochs)

#output.sort(key = lambda x : (x[0], x[1]))

print(output[0], output[-1])



