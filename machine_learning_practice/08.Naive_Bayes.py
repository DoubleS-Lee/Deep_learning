# Naive Bayes 실습

# 1. Gaussian Naive Bayes

# - 데이터, 모듈 불러오기

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

import pandas as pd

iris = datasets.load_iris()
df_X=pd.DataFrame(iris.data)
df_Y=pd.DataFrame(iris.target)

df_X.head()

df_Y.head()

# - 모델 피팅

gnb=GaussianNB()
fitted=gnb.fit(iris.data,iris.target)
y_pred=fitted.predict(iris.data)

fitted.predict_proba(iris.data)[[1,48,51,100]]

fitted.predict(iris.data)[[1,48,51,100]]

# - Confusion matrix 구하기

from sklearn.metrics import confusion_matrix

confusion_matrix(iris.target,y_pred)

# - Prior 설정하기

gnb2=GaussianNB(priors=[1/100,1/100,98/100])
fitted2=gnb2.fit(iris.data,iris.target)
y_pred2=fitted2.predict(iris.data)
confusion_matrix(iris.target,y_pred2)

gnb2=GaussianNB(priors=[1/100,98/100,1/100])
fitted2=gnb2.fit(iris.data,iris.target)
y_pred2=fitted2.predict(iris.data)
confusion_matrix(iris.target,y_pred2)

# 2. Multinomial naive bayes

# - 모듈 불러오기 및 데이터 생성

from sklearn.naive_bayes import MultinomialNB

import numpy as np

X = np.random.randint(5, size=(6, 100))
y = np.array([1, 2, 3, 4, 5, 6])

X

y

# - Multinomial naive bayes 모델 생성

clf = MultinomialNB()
clf.fit(X, y)

print(clf.predict(X[2:3]))

clf.predict_proba(X[2:3])

# - prior 변경해보기

clf2 = MultinomialNB(class_prior=[0.1,0.5,0.1,0.1,0.1,0.1])
clf2.fit(X, y)

clf2.predict_proba(X[2:3])