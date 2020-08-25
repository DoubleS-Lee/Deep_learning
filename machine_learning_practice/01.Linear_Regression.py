# 단순선형회귀 적합 및 해석
import os
import pandas as pd 
import numpy as np
import statsmodels.api as sm

# 현재경로 확인
os.getcwd()

# 데이터 불러오기
boston = pd.read_csv("C:/Users/DoubleS/Documents/[패캠]딥러닝-인공지능-강의자료/머신러닝과 데이터분석 A_Z/2. [Machine Learning]/PART 2) 회귀분석/3. 실습데이터/Boston_house.csv")

boston.head()

# target을 제외한 데이터만 뽑기
boston_data = boston.drop(['Target'],axis=1)
# boston_data

boston_data.describe()

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

# crim/rm/lstat 세게의 변수로 각각 단순 선형 회귀 분석하기

target = boston[['Target']]
# boston_target
crim=boston[['CRIM']]
rm=boston[['RM']]
lstat=boston[['LSTAT']]

## target ~ crim 선형회귀분석
# crim 변수에 상수항 추가하기
crim1 = sm.add_constant(crim, has_constant='add')
crim1

model1 = sm.OLS(target,crim1)
fitted_model1=model1.fit()


fitted_model1.summary()

fitted_model1.params

## y_hat=beta0 + beta1 * X 계산해보기
np.dot(crim1,fitted_model1.params)

len(np.dot(crim1,fitted_model1.params))

# prediction
pred1=fitted_model1.predict(crim1)

pred1-np.dot(crim1,fitted_model1.params)

## 적합시킨 직선 시각화

import matplotlib.pyplot as plt
plt.yticks(fontname = "Arial") #
plt.scatter(crim,target,label="data")
plt.plot(crim,pred1,label="result")
plt.legend()
plt.show()

plt.scatter(target,pred1)
plt.xlabel("real_value")
plt.ylabel("pred_value")
plt.show()

fitted_model1.resid.plot()
plt.xlabel("residual_number")
plt.show()

##잔차의 합계산해보기
sum(fitted_model1.resid)

## 위와 동일하게 rm변수와 lstat 변수로 각각 단순선형회귀분석 적합시켜보기
rm1 = sm.add_constant(rm, has_constant='add')
lstat1 = sm.add_constant(lstat, has_constant='add')

model2 = sm.OLS(target,rm1)
fitted_model2=model2.fit()
model3 = sm.OLS(target,lstat1)
fitted_model3=model3.fit()

fitted_model2.summary()

fitted_model3.summary()

pred2=fitted_model2.predict(rm1)
pred3=fitted_model3.predict(lstat1)


import matplotlib.pyplot as plt
plt.scatter(rm,target,label="data")
plt.plot(rm,pred2,label="result")
plt.legend()
plt.show()

import matplotlib.pyplot as plt
plt.scatter(lstat,target,label="data")
plt.plot(lstat,pred3,label="result")
plt.legend()
plt.show()

fitted_model2.resid.plot()
plt.xlabel("residual_number")
plt.show()

fitted_model3.resid.plot()
plt.xlabel("residual_number")
plt.show()

fitted_model1.resid.plot(label="crim")
fitted_model2.resid.plot(label="rm")
fitted_model3.resid.plot(label="lstat")
plt.legend()