# 다중 회귀 모델 해석 및 다중공선성 진단
import os
import pandas as pd 
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# 현재경로 확인
os.getcwd()

# 데이터 불러오기
boston = pd.read_csv("C:/Users/DoubleS/Documents/[패캠]딥러닝-인공지능-강의자료/머신러닝과 데이터분석 A_Z/2. [Machine Learning]/PART 2) 회귀분석/3. 실습데이터/Boston_house.csv")
boston

boston_data = boston.drop(['Target'],axis=1)
# boston_data

'''
타겟 데이터
1978 보스턴 주택 가격
506개 타운의 주택 가격 중앙값 (단위 1,000 달러)

특징 데이터
CRIM: 범죄율
INDUS: 비소매상업지역 면적 비율
NOX: 일산화질소 농도
RM: 주택당 방 수
LSTAT: 인구 중 하위 계층 비율
B: 인구 중 흑인 비율
PTRATIO: 학생/교사 비율
ZN: 25,000 평방피트를 초과 거주지역 비율
CHAS: 찰스강의 경계에 위치한 경우는 1, 아니면 0
AGE: 1940년 이전에 건축된 주택의 비율
RAD: 방사형 고속도로까지의 거리
DIS: 직업센터의 거리
TAX: 재산세율'''

## crim, rm, lstat을 통한 다중 선형 회귀분석

x_data=boston[['CRIM','RM','LSTAT']] ##변수 여러개
target = boston[['Target']]
x_data.head()

x_data1 = sm.add_constant(x_data, has_constant='add')


multi_model = sm.OLS(target,x_data1)
fitted_multi_model=multi_model.fit()

fitted_multi_model.summary()

## crim, rm, lstat, b, tax, age, zn, nox, indus 변수를 통한 다중선형회귀분석 

x_data2=boston[['CRIM','RM','LSTAT','B','TAX','AGE','ZN','NOX','INDUS']]  ##변수 추가
x_data2.head()

x_data2_ = sm.add_constant(x_data2, has_constant='add')


multi_model2 = sm.OLS(target,x_data2_)
fitted_multi_model2=multi_model2.fit()

fitted_multi_model2.summary()

fitted_multi_model.params

fitted_multi_model2.params

import matplotlib.pyplot as plt
fitted_multi_model.resid.plot(label="full")
fitted_multi_model2.resid.plot(label="full_add")
plt.legend()

## 상관계수/산점도를 통해 다중공선성 확인

x_data2.corr()

import seaborn as sns;
cmap = sns.light_palette("darkgray", as_cmap=True)
sns.heatmap(x_data2.corr(), annot=True, cmap=cmap)
plt.show()

sns.pairplot(x_data2)
plt.show()

# VIF를 통한 다중공선성 확인
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(x_data2.values, i) for i in range(x_data2.shape[1])]
vif["features"] = x_data2.columns
vif

vif = pd.DataFrame()
x_data3= x_data2.drop('NOX',axis=1)
vif["VIF Factor"] = [variance_inflation_factor(
    x_data3.values, i) for i in range(x_data3.shape[1])]
vif["features"] = x_data3.columns
vif

vif = pd.DataFrame()
x_data4= x_data3.drop('RM',axis=1)
vif["VIF Factor"] = [variance_inflation_factor(
    x_data4.values, i) for i in range(x_data4.shape[1])]
vif["features"] = x_data4.columns
vif

x_data5 = sm.add_constant(x_data4, has_constant='add')
model_vif = sm.OLS(target,x_data5)
fitted_model_vif=model_vif.fit()

fitted_model_vif.summary()

fitted_multi_model2.summary()


# 학습 / 검증데이터 분할
from sklearn.model_selection import train_test_split
X = x_data2_
y = target
train_x, test_x, train_y, test_y = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state = 1)
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)

test_y
train_x.head()
train_y.head()

train_x2 = sm.add_constant(train_x, has_constant='add')
fit_1 = sm.OLS(train_y,train_x)
fit_1=fit_1.fit()

plt.plot(np.array(fit_1.predict(test_x)),label="pred")
plt.plot(np.array(test_y),label="true")
plt.legend()
plt.show()

X = x_data3
y = target
train_x2, test_x2, train_y2, test_y2 = train_test_split(X, y, train_size=0.7, test_size=0.3,random_state = 1)
print(train_x2.shape, test_x2.shape, train_y2.shape, test_y2.shape)

X = x_data4
y = target
train_x3, test_x3, train_y3, test_y3 = train_test_split(X, y, train_size=0.7, test_size=0.3,random_state = 1)
print(train_x2.shape, test_x2.shape, train_y2.shape, test_y2.shape)

test_y2

fit_2 = sm.OLS(train_y2,train_x2)
fit_2=fit_2.fit()
fit_3 = sm.OLS(train_y3,train_x3)
fit_3=fit_3.fit()

## true값과 예측값 비교 
plt.plot(np.array(fit_2.predict(test_x2)),label="pred_2")
plt.plot(np.array(fit_3.predict(test_x3)),label="pred_3")
plt.plot(np.array(test_y2),label="true")
plt.legend()
plt.show()

## full모델 추가해서 비교 
plt.plot(np.array(fit_1.predict(test_x)),label="pred")
plt.plot(np.array(fit_2.predict(test_x2)),label="pred_vif")
plt.plot(np.array(fit_2.predict(test_x2)),label="pred_vif2")
plt.plot(np.array(test_y2),label="true")
plt.legend()
plt.show()

plt.plot(np.array(test_y2['Target']-fit_1.predict(test_x)),label="pred_full")
plt.plot(np.array(test_y2['Target']-fit_2.predict(test_x2)),label="pred_vif")
plt.plot(np.array(test_y2['Target']-fit_3.predict(test_x3)),label="pred_vif2")
plt.legend()
plt.show()

# MSE를 통한 검증데이터에 대한 성능비교

from sklearn.metrics import mean_squared_error

 mean_squared_error(y_true= test_y2['Target'], y_pred= fit_2.predict(test_x2))

 mean_squared_error(y_true= test_y2['Target'], y_pred= fit_3.predict(test_x3))

 mean_squared_error(y_true= test_y2['Target'], y_pred= fit_1.predict(test_x))

