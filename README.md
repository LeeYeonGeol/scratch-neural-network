# scratch-neural-network
## 1. Introduction
- 넘파이와 판다스로 간단히 구현한 뉴럴 네트워크입니다.
- 프로야구 데이터를 통해, 각 팀별 순위를 예측하는 프로그램입니다.

## 2. Dataset
- 데이터는 [스탯티즈](http://www.statiz.co.kr/)를 참고하여 직접 만든 데이터입니다. (양이 많지 않아 크롤링은 하지 않음)
- Input: 팀별 AVG, OBP, SLG, ERA, FIP, WHIP를 리그의 평균값으로 나눈 값이 들어갑니다.
- Output: 등수에 대한 예측 확률이 원-핫 벡터 형태로 나옵니다.

## 3. Neural-Network
### 구조도
<img src="assets/architecture.PNG" width=80%>

- Input Layer: 1개
- Hidden Layer: 2개
- Output Layer: 1개
  
6개의 값이 Input Layer로 들어와서 2개의 Hidden Layer를 거치고 softmax 연산을 통해, 최종 로짓값이 생성됩니다.


### Hyper Parameter
- learning Rate: 0.01
- epoch: 829
- loss function: Cross Entrophy

## 4. Result
### 학습과정
<img src="assets/loss_graph_1.PNG" width=60%>

## More Information


## Reference
### 역전파 구현
- https://towardsdatascience.com/deriving-backpropagation-with-cross-entropy-loss-d24811edeaf9
- https://www.mldawn.com/back-propagation-with-cross-entropy-and-softmax/