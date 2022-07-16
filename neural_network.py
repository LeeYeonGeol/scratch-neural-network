import numpy as np
import pandas as pd
from tqdm import tqdm
np.random.seed(42)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)

def softmax(z):
    exp_z = np.exp(z)
    sum_exp_z = np.sum(exp_z)
    return exp_z / sum_exp_z

def ce_loss(y_true, y_pred):
    return -(y_true * np.log(y_pred + 10**-100) + (1-y_true) * np.log(1 - y_pred + 10**-100))

class neuralnetwork:
    def __init__(self):
        self.w1 = np.random.rand(6, 5)
        self.w2 = np.random.rand(5, 5)
        self.w3 = np.random.rand(5, 10)
        self.b1 = np.random.rand(5)
        self.b2 = np.random.rand(5)
        self.b3 = np.random.rand(10)

    def train(self, X, y, lr, epochs):
        output = []
        for epoch in tqdm(range(epochs)):
            total_loss = 0
            for idx in range(len(X)):
                # 순전파
                # 계산
                inp = np.array(X.iloc[idx])
                h1 = inp.dot(self.w1) + self.b1
                h2 = sigmoid(h1).dot(self.w2) + self.b2
                o = h2.dot(self.w3) + self.b3
                y_pred = softmax(o)

                y_true = np.zeros(10)
                y_true[int(y.iloc[idx])-1] = 1
                loss = ce_loss(y_true, y_pred)
                total_loss += np.sum(loss)

                # 역전파
                # b3 & w3 역전파
                de_dz3 = y_pred-y_true
                deriv_3 = np.reshape(h2, (len(h2), 1)) * de_dz3

                # b2 & w2 역전파
                de_dz2 = self.w3.dot(np.reshape(de_dz3, (len(de_dz3), 1)))
                de_dz2 = de_dz2.squeeze()
                deriv_2 = np.reshape(h1, (len(h1), 1)) * (de_dz2 * deriv_sigmoid(h2))

                # b1 & w1 역전파
                de_dz1 = self.w2.dot(np.reshape(de_dz2, (len(de_dz2), 1)))
                de_dz1 = de_dz1.squeeze()
                deriv_1 = np.reshape(inp, (len(inp), 1)) * (de_dz1 * deriv_sigmoid(h1))

                # 업데이트
                self.w3 -= lr*deriv_3
                self.b3 -= lr*(de_dz3) 
                self.w2 -= lr*deriv_2
                self.b2 -= lr*(de_dz2) 
                self.w1 -= lr*deriv_1
                self.b1 -= lr*(de_dz1) 

            output.append((total_loss/len(X), epoch))
        return output





    