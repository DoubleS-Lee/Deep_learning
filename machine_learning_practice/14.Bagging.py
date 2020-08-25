import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 현재경로 확인
os.getcwd()

# 데이터 불러오기
data = pd.read_csv("./data/kc_house_data.csv") 
data.head() # 데이터 확인

'''
id: 집 고유아이디
date: 집이 팔린 날짜 
price: 집 가격 (타겟변수)
bedrooms: 주택 당 침실 개수
bathrooms: 주택 당 화장실 개수
floors: 전체 층 개수
waterfront: 해변이 보이는지 (0, 1)
condition: 집 청소상태 (1~5)
grade: King County grading system 으로 인한 평점 (1~13)
yr_built: 집이 지어진 년도
yr_renovated: 집이 리모델링 된 년도
zipcode: 우편번호
lat: 위도
long: 경도
'''

nCar = data.shape[0] # 데이터 개수
nVar = data.shape[1] # 변수 개수
print('nCar: %d' % nCar, 'nVar: %d' % nVar )

## 의미가 없다고 판단되는 변수 제거

data = data.drop(['id', 'date', 'zipcode', 'lat', 'long'], axis = 1) # id, date, zipcode, lat, long  제거

## 범주형 변수를 이진형 변수로 변환
- 범주형 변수는 waterfront 컬럼 뿐이며, 이진 분류이기 때문에 0, 1로 표현한다.
- 데이터에서 0, 1로 표현되어 있으므로 과정 생략

## 설명변수와 타겟변수를 분리, 학습데이터와 평가데이터 분리

feature_columns = list(data.columns.difference(['price'])) # Price를 제외한 모든 행
X = data[feature_columns]
y = data['price']
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state = 42) # 학습데이터와 평가데이터의 비율을 7:3
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape) # 데이터 개수 확인

## 학습 데이터를 선형 회귀 모형에 적합 후 평가 데이터로 검증 (Stats_Models)

import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

sm_train_x = sm.add_constant(train_x, has_constant = 'add') # Bias 추가
sm_model = sm.OLS(train_y, sm_train_x) # 모델 구축
fitted_sm_model = sm_model.fit() # 학습 진행
fitted_sm_model.summary() # 학습 모델 구조 확인

# 결과 확인
sm_test_x = sm.add_constant(test_x, has_constant = 'add') # 테스트 데이터에 Bias 추가
sm_model_predict = fitted_sm_model.predict(sm_test_x) # 테스트 데이터 예측
print("RMSE: {}".format(sqrt(mean_squared_error(sm_model_predict, test_y)))) # RMSE
print(fitted_sm_model.params) # 회귀계수

## Bagging 한 결과가 일반적인 결과보다 좋은지 확인

import random
bagging_predict_result = [] # 빈 리스트 생성
for _ in range(10):
    data_index = [data_index for data_index in range(train_x.shape[0])] # 학습 데이터의 인덱스를 리스트로 변환
    random_data_index = np.random.choice(data_index, train_x.shape[0]) # 데이터의 1/10 크기만큼 랜덤 샘플링, // 는 소수점을 무시하기 위함
    print(len(set(random_data_index)))
    sm_train_x = train_x.iloc[random_data_index, ] # 랜덤 인덱스에 해당되는 학습 데이터 중 설명 변수
    sm_train_y = train_y.iloc[random_data_index, ] # 랜덤 인덱스에 해당되는 학습 데이터 중 종속 변수
    sm_train_x = sm.add_constant(sm_train_x, has_constant = 'add') # Bias 추가
    sm_model = sm.OLS(sm_train_y, sm_train_x) # 모델 구축
    fitted_sm_model = sm_model.fit() # 학습 진행
    
    sm_test_x = sm.add_constant(test_x, has_constant = 'add') # 테스트 데이터에 Bias 추가
    sm_model_predict = fitted_sm_model.predict(sm_test_x) # 테스트 데이터 예측
    bagging_predict_result.append(sm_model_predict) # 반복문이 실행되기 전 빈 리스트에 결과 값 저장

bagging_predict_result[0] # 0 ~ 9, 10번의 예측을 하였기 때문에 10개의 결과가 생성

# Bagging을 바탕으로 예측한 결과값에 대한 평균을 계산
bagging_predict = [] # 빈 리스트 생성
for lst2_index in range(test_x.shape[0]): # 테스트 데이터 개수만큼의 반복
    temp_predict = [] # 임시 빈 리스트 생성 (반복문 내 결과값 저장)
    for lst_index in range(len(bagging_predict_result)): # Bagging 결과 리스트 반복
        temp_predict.append(bagging_predict_result[lst_index].values[lst2_index]) # 각 Bagging 결과 예측한 값 중 같은 인덱스를 리스트에 저장
    bagging_predict.append(np.mean(temp_predict)) # 해당 인덱스의 30개의 결과값에 대한 평균을 최종 리스트에 추가

bagging_predict

# 예측한 결과값들의 평균을 계산하여 실제 테스트 데이트의 타겟변수와 비교하여 성능 평가
print("RMSE: {}".format(sqrt(mean_squared_error(bagging_predict, test_y)))) # RMSE

## 학습 데이터를 선형 회귀 모형에 적합 후 평가 데이터로 검증 (Scikit-Learn)

from sklearn.linear_model import LinearRegression
regression_model = LinearRegression() # 선형 회귀 모형
linear_model1 = regression_model.fit(train_x, train_y) # 학습 데이터를 선형 회귀 모형에 적합
predict1 = linear_model1.predict(test_x) # 학습된 선형 회귀 모형으로 평가 데이터 예측
print("RMSE: {}".format(sqrt(mean_squared_error(predict1, test_y)))) # RMSE 결과

## Bagging 을 이용하여 선형 회귀 모형에 적합 후 평가 (Sampling 10번)

from sklearn.ensemble import BaggingRegressor
bagging_model = BaggingRegressor(base_estimator = regression_model, # 선형회귀모형
                                 n_estimators = 5, # 5번 샘플링
                                 verbose = 1) # 학습 과정 표시
linear_model2 = bagging_model.fit(train_x, train_y) # 학습 진행
predict2 = linear_model2.predict(test_x) # 학습된 Bagging 선형 회귀 모형으로 평가 데이터 예측
print("RMSE: {}".format(sqrt(mean_squared_error(predict2, test_y)))) # RMSE 결과

## 그렇다면 Sampling을 많이 해보자!

bagging_model2 = BaggingRegressor(base_estimator = regression_model, # 선형 회귀모형
                                  n_estimators = 30, # 30번 샘플링
                                  verbose = 1) # 학습 과정 표시
linear_model3 = bagging_model2.fit(train_x, train_y) # 학습 진행
predict3 = linear_model3.predict(test_x) # 학습된 Bagging 선형 회귀 모형으로 평가 데이터 예측
print("RMSE: {}".format(sqrt(mean_squared_error(predict3, test_y)))) # RMSE 결과

## 학습 데이터를 의사결정나무모형에 적합 후 평가 데이터로 검증

from sklearn.tree import DecisionTreeRegressor
decision_tree_model = DecisionTreeRegressor() # 의사결정나무 모형
tree_model1 = decision_tree_model.fit(train_x, train_y) # 학습 데이터를 의사결정나무 모형에 적합
predict1 = tree_model1.predict(test_x) # 학습된 의사결정나무 모형으로 평가 데이터 예측
print("RMSE: {}".format(sqrt(mean_squared_error(predict1, test_y)))) # RMSE 결과

import random
bagging_predict_result = [] # 빈 리스트 생성
for _ in range(30):
    data_index = [data_index for data_index in range(train_x.shape[0])] # 학습 데이터의 인덱스를 리스트로 변환
    random_data_index = np.random.choice(data_index, train_x.shape[0]) # 데이터의 1/10 크기만큼 랜덤 샘플링, // 는 소수점을 무시하기 위함
    print(len(set(random_data_index)))
    sm_train_x = train_x.iloc[random_data_index, ] # 랜덤 인덱스에 해당되는 학습 데이터 중 설명 변수
    sm_train_y = train_y.iloc[random_data_index, ] # 랜덤 인덱스에 해당되는 학습 데이터 중 종속 변수
    decision_tree_model = DecisionTreeRegressor() # 의사결정나무 모형
    tree_model1 = decision_tree_model.fit(sm_train_x, sm_train_y) # 학습 데이터를 의사결정나무 모형에 적합
 
    predict1 = tree_model1.predict(test_x) # 테스트 데이터 예측
    bagging_predict_result.append(predict1) # 반복문이 실행되기 전 빈 리스트에 결과 값 저장

# Bagging을 바탕으로 예측한 결과값에 대한 평균을 계산
bagging_predict = [] # 빈 리스트 생성
for lst2_index in range(test_x.shape[0]): # 테스트 데이터 개수만큼의 반복
    temp_predict = [] # 임시 빈 리스트 생성 (반복문 내 결과값 저장)
    for lst_index in range(len(bagging_predict_result)): # Bagging 결과 리스트 반복
        temp_predict.append(bagging_predict_result[lst_index][lst2_index]) # 각 Bagging 결과 예측한 값 중 같은 인덱스를 리스트에 저장
    bagging_predict.append(np.mean(temp_predict)) # 해당 인덱스의 30개의 결과값에 대한 평균을 최종 리스트에 추가

# 예측한 결과값들의 평균을 계산하여 실제 테스트 데이트의 타겟변수와 비교하여 성능 평가

print("RMSE: {}".format(sqrt(mean_squared_error(bagging_predict, test_y)))) # RMSE

bagging_predict_result[29]

## Bagging 을 이용하여 의사결정나무모형에 적합 후 평가 (Sampling 10번)

bagging_decision_tree_model1 = BaggingRegressor(base_estimator = decision_tree_model, # 의사결정나무 모형
                                                n_estimators = 5, # 5번 샘플링
                                                verbose = 1) # 학습 과정 표시
tree_model2 = bagging_decision_tree_model1.fit(train_x, train_y) # 학습 진행
predict2 = tree_model2.predict(test_x) # 학습된 Bagging 의사결정나무 모형으로 평가 데이터 예측
print("RMSE: {}".format(sqrt(mean_squared_error(predict2, test_y)))) # RMSE 결과

bagging_decision_tree_model2 = BaggingRegressor(base_estimator = decision_tree_model, # 의사결정나무 모형
                                                n_estimators = 30, # 30번 샘플링
                                                verbose = 1) # 학습 과정 표시
tree_model3 = bagging_decision_tree_model2.fit(train_x, train_y) # 학습 진행
predict3 = tree_model3.predict(test_x) # 학습된 Bagging 의사결정나무 모형으로 평가 데이터 예측
print("RMSE: {}".format(sqrt(mean_squared_error(predict3, test_y)))) # RMSE 결과

