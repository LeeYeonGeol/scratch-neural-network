import numpy as np
import pandas as pd
import random
np.random.seed(42)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(z):
    exp_z = np.exp(z)
    sum_exp_z = np.sum(exp_z)
    return exp_z / sum_exp_z

def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

class neuralnetwork:
    def __init__(self):
        self.w1 = np.random.rand(6, 5)
        self.w2 = np.random.rand(5, 5)
        self.w3 = np.random.rand(5, 10)
        self.b1 = np.random.rand(5)
        self.b2 = np.random.rand(5)
        self.b3 = np.random.rand(10)

    def train(self, df):
    
        for idx in range(len(df)):
            # 순전파
            ## 초기화
            h1, h2, o = np.zeros(5), np.zeros(5), np.zeros(10)
            # 계산
            h1 = df.iloc[idx][:-1].dot(self.w1) + self.b1
            h2 = sigmoid(h1).dot(self.w2) + self.b2
            o = sigmoid(h2).dot(self.w3) + self.b3
            y_pred = softmax(o)
          
            # 역전파
            y_true = np.zeros(10)
            y_true[int(df.iloc[idx][-1])-1] = 1
            print(y_true, y_pred)
            
            print(-np.sum(y_true * np.log(y_pred + 10**-100)))

            
            break

        return 0


    