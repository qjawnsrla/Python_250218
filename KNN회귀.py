# 회귀 ( Regression ) : 연속적인 수치 값 예측
# k-최근접 이웃 회귀 : 가장 가까운 k개의 이웃 값의 평균으로 예측
import numpy as np
import matplotlib.pyplot as plt
from keras.src.losses import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor   # k-NN 회귀 모델
from sklearn.metrics import mean_absolute_error     # 평균 절댓값 오차 관련 모듈

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

plt.scatter(perch_length, perch_weight)
plt.xlabel('length(cm)')
plt.ylabel('weight(g)')
plt.show()

# 훈련 세트와 테스트 세트 분리
train_input, test_input, train_target, test_target = (
    train_test_split(perch_length, perch_weight, random_state=42)
)

# 2차원 배열로 변환
train_input = train_input.reshape(-1, 1)    # 행은 자동계산, 열은 1개로 변환
test_input = test_input.reshape(-1, 1)

## k-최근접 이웃 회귀 모델 훈련
knr = KNeighborsRegressor()

# 모델 훈련
knr.fit(train_input, train_target)

# 모델 평가 : 1에 가까울수록 좋음
test_score = knr.score(test_input, test_target)
print(f"테스트 세트 결정계수(R^2): {test_score}")

# 평균 절댓값 오차 : 값이 작을수록 예측이 실제값에 가까움
# 테스트 세트에 대한 예측 생성
test_prediction = knr.predict(test_input)

# 테스트 세트에 대한 평균 절댓값 오차 계산
mae = mean_absolute_error(test_target, test_prediction)
print(f"평균 절댓값 오차 : {mae}")

# 과대 적합 vs 과소 적합
# 과대 적합 : 모델이 훈련 세트에 너무 맞춰져 새로운 데이터에 대한 예측력이 떨어지는 것
# - 훈련 세트 점수는 높지만 테스트 세트 점수가 낮게 나옴
# 과소 적합 : 모델이 충분히 훈련 되지 않아 데이터 패턴을 잘 학습하지 못한 경우
# - 훈련 세트와 테스트 세트 점수가 모두 낮은 경우, 테스트 세트 점수가 훈련 세트 보다 높은 경우

# 훈련 세트 점수
train_score = knr.score(train_input, train_target)
print(f"훈련 세트 결정 계수 (R^2): {train_score}")
# 테스트 세트 점수
test_score = knr.score(test_input, test_target)
print(f"테스트 세트 결정 계수 (R^2): {test_score}")

# 모델 개선 : 이웃 수 변경
# 과소 적합을 해결하기 위해서는 모델의 복잡도를 높여야하고, 복잡도를 높이기 위해서는 k(이웃의 개수) 를 줄이면 복잡도 증가
knr.n_neighbors = 3
knr.fit(train_input, train_target)
train_score = knr.score(train_input, train_target)
print(f"훈련 세트 결정 계수 (R^2): {train_score}")
# 테스트 세트 점수
test_score = knr.score(test_input, test_target)
print(f"테스트 세트 결정 계수 (R^2): {test_score}")

