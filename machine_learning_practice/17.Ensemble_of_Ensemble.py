import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 현재경로 확인
os.getcwd()

# 데이터 불러오기
data = pd.read_csv("./data/otto_train.csv") # Product Category
data.head() # 데이터 확인

'''
id: 고유 아이디
feat_1 ~ feat_93: 설명변수
target: 타겟변수 (1~9)
'''

nCar = data.shape[0] # 데이터 개수
nVar = data.shape[1] # 변수 개수
print('nCar: %d' % nCar, 'nVar: %d' % nVar )

## 의미가 없다고 판단되는 변수 제거

data = data.drop(['id'], axis = 1) # id 제거

## 타겟 변수의 문자열을 숫자로 변환

mapping_dict = {"Class_1": 1,
                "Class_2": 2,
                "Class_3": 3,
                "Class_4": 4,
                "Class_5": 5,
                "Class_6": 6,
                "Class_7": 7,
                "Class_8": 8,
                "Class_9": 9}
after_mapping_target = data['target'].apply(lambda x: mapping_dict[x])

## 설명변수와 타겟변수를 분리, 학습데이터와 평가데이터 분리

feature_columns = list(data.columns.difference(['target'])) # target을 제외한 모든 행
X = data[feature_columns] # 설명변수
y = after_mapping_target # 타겟변수
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 42) # 학습데이터와 평가데이터의 비율을 8:2 로 분할| 
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape) # 데이터 개수 확인

## 1. XGBoost

# !pip install xgboost
import xgboost as xgb
import time
start = time.time() # 시작 시간 지정
xgb_dtrain = xgb.DMatrix(data = train_x, label = train_y) # 학습 데이터를 XGBoost 모델에 맞게 변환
xgb_dtest = xgb.DMatrix(data = test_x) # 평가 데이터를 XGBoost 모델에 맞게 변환
xgb_param = {'max_depth': 10, # 트리 깊이
         'learning_rate': 0.01, # Step Size
         'n_estimators': 100, # Number of trees, 트리 생성 개수
         'objective': 'multi:softmax', # 목적 함수
        'num_class': len(set(train_y)) + 1} # 파라미터 추가, Label must be in [0, num_class) -> num_class보다 1 커야한다.
xgb_model = xgb.train(params = xgb_param, dtrain = xgb_dtrain) # 학습 진행
xgb_model_predict = xgb_model.predict(xgb_dtest) # 평가 데이터 예측
print("Accuracy: %.2f" % (accuracy_score(test_y, xgb_model_predict) * 100), "%") # 정확도 % 계산
print("Time: %.2f" % (time.time() - start), "seconds") # 코드 실행 시간 계산

xgb_model_predict

## 2. LightGBM

# !pip install lightgbm
import lightgbm as lgb
start = time.time() # 시작 시간 지정
lgb_dtrain = lgb.Dataset(data = train_x, label = train_y) # 학습 데이터를 LightGBM 모델에 맞게 변환
lgb_param = {'max_depth': 10, # 트리 깊이
            'learning_rate': 0.01, # Step Size
            'n_estimators': 100, # Number of trees, 트리 생성 개수
            'objective': 'multiclass', # 목적 함수
            'num_class': len(set(train_y)) + 1} # 파라미터 추가, Label must be in [0, num_class) -> num_class보다 1 커야한다.
lgb_model = lgb.train(params = lgb_param, train_set = lgb_dtrain) # 학습 진행
lgb_model_predict = np.argmax(lgb_model.predict(test_x), axis = 1) # 평가 데이터 예측, Softmax의 결과값 중 가장 큰 값의 Label로 예측
print("Accuracy: %.2f" % (accuracy_score(test_y, lgb_model_predict) * 100), "%") # 정확도 % 계산
print("Time: %.2f" % (time.time() - start), "seconds") # 코드 실행 시간 계산

lgb_model.predict(test_x)

## 3. Catboost

# !pip install catboost
import catboost as cb
start = time.time() # 시작 시간 지정
cb_dtrain = cb.Pool(data = train_x, label = train_y) # 학습 데이터를 Catboost 모델에 맞게 변환
cb_param = {'max_depth': 10, # 트리 깊이
            'learning_rate': 0.01, # Step Size
            'n_estimators': 100, # Number of trees, 트리 생성 개수
            'eval_metric': 'Accuracy', # 평가 척도
            'loss_function': 'MultiClass'} # 손실 함수, 목적 함수
cb_model = cb.train(pool = cb_dtrain, params = cb_param) # 학습 진행
cb_model_predict = np.argmax(cb_model.predict(test_x), axis = 1) + 1 # 평가 데이터 예측, Softmax의 결과값 중 가장 큰 값의 Label로 예측, 인덱스의 순서를 맞추기 위해 +1
print("Accuracy: %.2f" % (accuracy_score(test_y, cb_model_predict) * 100), "%") # 정확도 % 계산
print("Time: %.2f" % (time.time() - start), "seconds") # 코드 실행 시간 계산

cb_model.predict(test_x)

# 데이터 불러오기
data = pd.read_csv("./data/kc_house_data.csv") 
data.head() # 데이터 확인

data = data.drop(['id', 'date', 'zipcode', 'lat', 'long'], axis = 1) # id, date, zipcode, lat, long  제거

feature_columns = list(data.columns.difference(['price'])) # Price를 제외한 모든 행
X = data[feature_columns]
y = data['price']
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state = 42) # 학습데이터와 평가데이터의 비율을 7:3
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape) # 데이터 개수 확인

# !pip install lightgbm
import lightgbm as lgb
start = time.time() # 시작 시간 지정
lgb_dtrain = lgb.Dataset(data = train_x, label = train_y) # 학습 데이터를 LightGBM 모델에 맞게 변환
lgb_param = {'max_depth': 10, # 트리 깊이
            'learning_rate': 0.01, # Step Size
            'n_estimators': 500, # Number of trees, 트리 생성 개수
            'objective': 'regression'} # 파라미터 추가, Label must be in [0, num_class) -> num_class보다 1 커야한다.
lgb_model = lgb.train(params = lgb_param, train_set = lgb_dtrain) # 학습 진행


from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

sqrt(mean_squared_error(lgb_model.predict(test_x),test_y))

## Ensemble의 Ensemble

import random
bagging_predict_result = [] # 빈 리스트 생성
for _ in range(10):
    data_index = [data_index for data_index in range(train_x.shape[0])] # 학습 데이터의 인덱스를 리스트로 변환
    random_data_index = np.random.choice(data_index, train_x.shape[0]) # 데이터의 1/10 크기만큼 랜덤 샘플링, // 는 소수점을 무시하기 위함
    print(len(set(random_data_index)))
    lgb_dtrain = lgb.Dataset(data = train_x.iloc[random_data_index,], label = train_y.iloc[random_data_index,]) # 학습 데이터를 LightGBM 모델에 맞게 변환
    lgb_param = {'max_depth': 14, # 트리 깊이
            'learning_rate': 0.01, # Step Size
            'n_estimators': 500, # Number of trees, 트리 생성 개수
            'objective': 'regression'} # 파라미터 추가, Label must be in [0, num_class) -> num_class보다 1 커야한다.
    lgb_model = lgb.train(params = lgb_param, train_set = lgb_dtrain) # 학습 진행
    predict1 = lgb_model.predict(test_x) # 테스트 데이터 예측
    bagging_predict_result.append(predict1) # 반복문이 실행되기 전 빈 리스트에 결과 값 저장

bagging_predict_result

# Bagging을 바탕으로 예측한 결과값에 대한 평균을 계산
bagging_predict = [] # 빈 리스트 생성
for lst2_index in range(test_x.shape[0]): # 테스트 데이터 개수만큼의 반복
    temp_predict = [] # 임시 빈 리스트 생성 (반복문 내 결과값 저장)
    for lst_index in range(len(bagging_predict_result)): # Bagging 결과 리스트 반복
        temp_predict.append(bagging_predict_result[lst_index][lst2_index]) # 각 Bagging 결과 예측한 값 중 같은 인덱스를 리스트에 저장
    bagging_predict.append(np.mean(temp_predict)) # 해당 인덱스의 30개의 결과값에 대한 평균을 최종 리스트에 추가

# 예측한 결과값들의 평균을 계산하여 실제 테스트 데이트의 타겟변수와 비교하여 성능 평가

print("RMSE: {}".format(sqrt(mean_squared_error(bagging_predict, test_y)))) # RMSE

bagging_predict

