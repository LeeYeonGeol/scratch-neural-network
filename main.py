import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import neural_network
import data_preprocessing

# np & pd 관련 설정
np.random.seed(42)
np.set_printoptions(precision=2, suppress=True)
pd.options.display.float_format = '{:.3f}'.format

# 훈련 데이터 로드
train_data = pd.read_csv("data/KBO_TRAIN.csv")

# 전처리
X_train, y_train = train_data.drop('GRADE', axis=1), train_data['GRADE']
X_train = data_preprocessing.preprocess_X(X_train)
y_train = data_preprocessing.preprocess_y(y_train)

# 스케일링
X_train, mean, std = data_preprocessing.standard_X(X_train)

# 뉴럴네트워크 생성
nn = neural_network.neuralnetwork()

# 학습
learning_rate = 0.01
epochs = 829
output = nn.train(X_train, y_train, learning_rate, epochs)

# loss 그래프 그리기
sns.lineplot(data=output, x="epoch", y="loss")
plt.ylim([0, max(output["loss"])+0.5])
plt.show()

# 테스트 데이터 로드
test_data = pd.read_csv("data/KBO_TEST.csv")

# 전처리
X_test, y_test = test_data.drop('GRADE', axis=1), test_data['GRADE']
X_test = data_preprocessing.preprocess_X(X_test)
y_test = data_preprocessing.preprocess_y(y_test)

# 스케일링
X_test, mean, std = data_preprocessing.standard_X(X_test, mean = mean, std = std)

# 예측
pred_top3, pred_detail = nn.predict(X_test, y_test, TEAM = test_data.dropna()['TEAM'])
print(pred_top3)
print(pred_detail)