# 다중회구분석 적합 및 단순선형회귀와의 비교
import os
import pandas as pd 
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
# 현재경로 확인
os.getcwd()

# 단순선형회귀분석(선형회귀실습자료와동일)

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

target = boston[['Target']]

# 다중선형회귀분석
## crim, rm, lstat 세개의 변수를 통해 다중회귀적합
x_data=boston[['CRIM','RM','LSTAT']] ##변수 여러개
x_data.head()

# 상수항 추가
x_data1 = sm.add_constant(x_data, has_constant='add')

# 회귀모델 적합
multi_model = sm.OLS(target,x_data1)
fitted_multi_model=multi_model.fit()

fitted_multi_model.summary()

## 회귀계수
print(fitted_multi_model.params)  

## 행렬연산을 통해 beta구하기

from numpy import linalg ##행렬연산을 통해 beta구하기 
ba=linalg.inv((np.dot(x_data1.T,x_data1))) ## (X'X)-1
np.dot(np.dot(ba,x_data1.T),target) ##(X'X)-1X'y

pred4=fitted_multi_model.predict(x_data1)

## residual plot

fitted_multi_model.resid.plot()
plt.xlabel("residual_number")
plt.show()

fitted_multi_model.resid.plot(label="full")
plt.legend()

