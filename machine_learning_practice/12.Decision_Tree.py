# Decision Tree 실습

# 1. 함수 익히기 및 모듈 불러오기

# - 함수 익히기

from sklearn import tree
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

clf.predict([[1, 1]])

# - 모듈 불러오기

from sklearn.datasets import load_iris
from sklearn import tree
from os import system

system("pip install graphviz")

import graphviz 

# - 데이터 로드

iris=load_iris()

iris.data

iris.feature_names

iris.target

iris.target_names

# 2. 의사결정나무 구축 및 시각화

# - 트리 구축

clf=tree.DecisionTreeClassifier()
clf=clf.fit(iris.data,iris.target)

# - 트리의 시각화

dot_data=tree.export_graphviz(clf,out_file=None,
                             feature_names=iris.feature_names,
                            class_names=iris.target_names,
                              filled=True, rounded=True,
                              special_characters=True
                             )
graph=graphviz.Source(dot_data)

graph

# - 엔트로피를 활용한 트리

clf2=tree.DecisionTreeClassifier(criterion="entropy")

clf2.fit(iris.data,iris.target)

dot_data2=tree.export_graphviz(clf2,out_file=None,
                             feature_names=iris.feature_names,
                            class_names=iris.target_names,
                              filled=True, rounded=True,
                              special_characters=True
                             )
graph2=graphviz.Source(dot_data2)

graph2

# - 프루닝

clf3=tree.DecisionTreeClassifier(criterion="entropy",max_depth=2)

clf3.fit(iris.data,iris.target)

dot_data3=tree.export_graphviz(clf3,out_file=None,
                             feature_names=iris.feature_names,
                            class_names=iris.target_names,
                              filled=True, rounded=True,
                              special_characters=True
                             )
graph3=graphviz.Source(dot_data3)
graph3

# - Confusion Matrix 구하기

from sklearn.metrics import confusion_matrix
confusion_matrix(iris.target,clf.predict(iris.data))

confusion_matrix(iris.target,clf2.predict(iris.data))

confusion_matrix(iris.target,clf3.predict(iris.data))

# 3. Training - Test 구분 및 Confusion matrix 계산

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris.data,iris.target, stratify=iris.target,random_state=1)

clf4=tree.DecisionTreeClassifier(criterion="entropy")

clf4.fit(X_train,y_train)

confusion_matrix(y_test,clf4.predict(X_test))

# 4. Decision regression tree

# - 모듈 불러오기 및 데이터 생성

import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

# - Regression tree 구축

regr1=tree.DecisionTreeRegressor(max_depth=2)
regr2=tree.DecisionTreeRegressor(max_depth=5)

regr1.fit(X,y)

regr2.fit(X,y)

X_test=np.arange(0.0,5.0,0.01)[:,np.newaxis]
X_test

y_1=regr1.predict(X_test)
y_2=regr2.predict(X_test)

y_1

plt.figure()
plt.scatter(X, y, s=20, edgecolor="black",
            c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue",
         label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()

dot_data4 = tree.export_graphviz(regr2, out_file=None, 
                                filled=True, rounded=True,  
                                special_characters=True)

graph4 = graphviz.Source(dot_data4) 
graph4

dot_data5 = tree.export_graphviz(regr1, out_file=None, 
                                filled=True, rounded=True,  
                                special_characters=True)

graph5=graphviz.Source(dot_data5)

graph5