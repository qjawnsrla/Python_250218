# 회귀는 입력 데이터(독립변수) 를 기반으로 출력변수(종속변수) 를 예측하는 모델을 만드는 과정

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor   # k-NN 회귀 모델

perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

# 훈련 세트와 테스트 세트로 나눕니다
train_input, test_input, train_target, test_target = (
    train_test_split(perch_length, perch_weight, random_state=42)
)

# 훈련세트와 테스트세트를 2차원 배열로 변환합니다.
train_input = train_input.reshape(-1, 1)    # 행은 자동계산, 열은 1개로 변환
test_input = test_input.reshape(-1, 1)

## k-최근접 이웃 회귀 모델 훈련
knr = KNeighborsRegressor(n_neighbors=3)

# k-최근접 이웃 회귀 모델을 훈련합니다.
knr.fit(train_input, train_target)

# 농어의 무게 예측하기
print(knr.predict([[50]]))

# 선형 회귀 (Linear Regression)
# - 입력 데이터와 출력 데이터의 선형 관계를 학습
# - 가장 적합한 직선을 찾아 새로운 데이터를 예측
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

# 선형 회귀 모델을 훈련
lr.fit(train_input, train_target)
print(lr.predict([[50]]))

# 훈련 세트의 산점도
plt.scatter(train_input, train_target)

# 15에서 50까지의 1차 방정식 그래프 그리기
plt.plot([15, 50], [15*lr.coef_+lr.intercept_, 50*lr.coef_+lr.intercept_])

# 50cm 농어 데이터
plt.scatter(50, 1241.8, marker='^', color='red')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 다항 회귀 : 2차 방정식 그래프를 그리기 위해서 길이의 제곱한 항이 훈련 세트에 추가되어야 함.
train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))

print(train_poly.shape, test_poly.shape)

lr = LinearRegression()
lr.fit(train_poly, train_target)

print(lr.predict([[50**2, 50]]))

# 시각화하기
point = np.arange(15, 50)   # 구간별 직선을 그리기 위해 15에서 49까지 정수 배열 생성

# 후련 세트 산점도 그리기
plt.plot(point, 1.01*point**2 - 21.6*point + 116.05)

# 50cm 농어 데이터
plt.scatter(50, 1574, marker='^', color='skyblue')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()