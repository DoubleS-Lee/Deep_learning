#%%
# SVM
import numpy as np
import pandas as pd
from sklearn import svm

train = pd.read_csv('C:/Users/6533567/Documents/AI_fast_track_DS40/02.실습자료/titanic/train_pre.csv')
test = pd.read_csv('C:/Users/6533567/Documents/AI_fast_track_DS40/02.실습자료/titanic/test_pre.csv')
target = pd.read_csv('C:/Users/6533567/Documents/AI_fast_track_DS40/02.실습자료/titanic/train_label_pre.csv')
test_ori = pd.read_csv('C:/Users/6533567/Documents/AI_fast_track_DS40/02.실습자료/titanic/test.csv')

target = target.to_numpy()[:,1:]

clf = svm.SVC(gamma='scale')
clf.fit(train,target)
y_pred = clf.predict(test)

y_pred_1 = np.round(y_pred).astype(np.int)

submission = pd.DataFrame({'passengerId':test_ori['PassengerId'],'Survived':y_pred_1[:,1]})

print(y_pred_1)

 


#%%
#Naive bayes classification


from sklearn.naive_bayes import GaussianNB


clf = GaussianNB()
clf.fit(X, Y)

 

 

#%%
GA (Genetic Algorithm)
처음에 랜덤으로 뽑아서 보고
제일 우수한 모델과 유사한걸 재생산해서 학습시키고
거기서 또 우수한 모델을 가려내서 재생산하여 학습시키는 것을 반복

단점
어느 봉우리에 수렴할지 몰라서 로컬미니멈에 빠질수 있음

GA 알고리즘
가장 우수한 놈을 뽑는다 + 우수하지 않은 놈들중에서도 랜덤으로 뽑아준다 (확률기반으로 뽑음) - selection
부모의 형질을 그대로 갖고 있다. 우수한 놈들을 기반으로 부모의 형질을 그대로 물려받은 자식을 생산한다 - crossover
부모와 형질이 완전히 다른 자식을 생성한다 - mutation

#%%
#GA
import pandas as pd
import numpy as np
from geneticalgorithm import geneticalgorithm as ga
from sklearn.model_selection import train_test_split


# 이 모델의 조건은 x1,x2,x3이 있는거고 이거의 범위는 0<= x <=10이고 x1 + x2 >=3 을 만족해야한다면
# 밑에 objective function 을 다음과 같이 짜주면 된다 (패널티를 내가 만들어서 먹이는 개념이다)

#objective function : 이걸 minimalize를 하는것이 목적
def f(X):
    if X[0] + X[1] <3:
        pen=20
    else:
        pen=0
    return np.sum(X)

# x1,x2,x3의 범위를 만들어냈다
varbound = np.array([[0,10]]*3)

algorithm_param = {'max_num_iteration': None,
                   'population_size':100,
                   'mutation_probability':0.1,
                   'elit_ratio': 0.01,
                   'crossover_probability': 0.5,
                   'parents_portion': 0.3,
                   'crossover_type':'uniform',
                   'max_iteration_without_improv':None}

model = ga(function = f, dimension = 3, variable_type = 'real', variable_boundaries = varbound, algorithm_parameters = algorithm_param)

model.run()


#%%
#GA
#Boston dataset
import pandas as pd
import numpy as np
from geneticalgorithm import geneticalgorithm as ga
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm


# 여기에서의 인자인 x는 varbound로부터 온다
def f(x):
    X,y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.33, random_state=42)
    X_train *= x
    X_test *= x
   
    mdl = KNeighborsRegressor(n_neighbors = 5)
    mdl.fit(X_train, y_train)
    # mdl.score 는 1-R^2 이다
    error = np.sum(y_test - mdl.predict(X_test))**2
    return error


#
varbound = np.array([[0,1]]*13)

algorithm_param = {'max_num_iteration': None,
                   'population_size':100,
                   'mutation_probability':0.1,
                   'elit_ratio': 0.01,
                   'crossover_probability': 0.5,
                   'parents_portion': 0.3,
                   'crossover_type':'uniform',
                   'max_iteration_without_improv':None}

 

model = ga(function = f, dimension = 13, variable_type = 'real', variable_boundaries = varbound, algorithm_parameters = algorithm_param)

model.run()

 

#%%
import pandas as pd
import numpy as np
from geneticalgorithm import geneticalgorithm as ga
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm

# 여기에서의 인자인 x는 varbound로부터 온다
def f(x):
    X,y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.33, random_state=42)
    E = []
   
    for i in range(4):
        X_train *= x[13*i:13*(i+1)]
        X_test *= x[13*i:13*(i+1)]
        mdl = KNeighborsRegressor(n_neighbors = 5)
        mdl.fit(X_train, y_train)
        error = np.sum(y_test - mdl.predict(X_test))**2
       
        E.append(error)
    E = np.array(E)
   
    return np.mean(E)

algorithm_param = {'max_num_iteration': None,
                   'population_size':100,
                   'mutation_probability':0.1,
                   'elit_ratio': 0.01,
                   'crossover_probability': 0.5,
                   'parents_portion': 0.3,
                   'crossover_type':'uniform',
                   'max_iteration_without_improv':None}

model = ga(function = f, dimension = 13*4, variable_type = 'bool', algorithm_parameters = algorithm_param)

model.run()

#%%
# Artifitial Neural Network
# AND Gate
import numpy as np
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])
dir(x)
learing_rate = 0.1
epoch = 1000
W = np.random.random((1,2))

def sigmoid(x):
    return 1/(1+np.exp(-x))

for epochs in range(epoch):
    for i in range(0,len(y)):
        y_pred = sigmoid(np.dot(W,x[i,:]))
        error = y[i]-y_pred
        dW = learing_rate*(1-y_pred)*y_pred*error*x[i,:]
        W = W+dW
W.shape
x.shape
Y = []
for i in range(0,len(y)):
    Y = np.dot(W, np.transpose(x[i,:]))
    print(Y)

print(W)

#%%

Stochastic Gradient Descent
Batch Gradient Descent
mini-Batch Gradient Descent


#%%
#bike

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv('C:/Users/6533567/Documents/AI_fast_track_DS40/02.실습자료/bike/train.csv')
test = pd.read_csv('C:/Users/6533567/Documents/AI_fast_track_DS40/02.실습자료/bike/test.csv')

train.info()
test.info()

train.head()

fig = plt.figure(figsize=[20,20])
ax = sns.heatmap(train.corr(),annot=True,square=True)

 

 

 

 


#%%
# titanic Example
import pandas as pd
from sklearn import datasets
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

X = pd.read_csv('C:/Users/6533567/Documents/AI_fast_track_DS40/02.실습자료/titanic/train_pre.csv')
test = pd.read_csv('C:/Users/6533567/Documents/AI_fast_track_DS40/02.실습자료/titanic/test_pre.csv')
y = pd.read_csv('C:/Users/6533567/Documents/AI_fast_track_DS40/02.실습자료/titanic/train_label_pre.csv')
test_ori = pd.read_csv('C:/Users/6533567/Documents/AI_fast_track_DS40/02.실습자료/titanic/test.csv')
fig = plt.figure(figsize=[20,20])
ax = sns.heatmap(X.corr(),annot=True,square=True)

train.shape
test.shape
target.shape

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

 

forest = RandomForestClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


X.info()


y = y.to_numpy()[:,1:]

X = X.to_numpy()[:,:4]

test = test.to_numpy()[:,:4]

clf = svm.SVC(gamma='scale')
clf.fit(X,y)
y_pred = clf.predict(test)
y_pred_1 = np.round(y_pred).astype(np.int)

submission = pd.DataFrame({'passengerId':test_ori['PassengerId'],'Survived':y_pred_1[:]})

test_ori.shape
y_pred_1.shape

 

 


submission.to_csv('titanic/submission_lee_DT.csv', index=False)

 


y_pred_1 = np.round(y_pred).astype(np.int)

submission = pd.DataFrame({'passengerId':test_ori['PassengerId'],'Survived':y_pred_1[:,1]})


#%%

## Titanic

import numpy as np
from geneticalgorithm import geneticalgorithm as ga
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd

global n_classifier
global n_feature
global X_train
global y


train = pd.read_csv('titanic/train_preprocessing.csv')
test = pd.read_csv('titanic/test_preprocessing.csv')
target = pd.read_csv('titanic/target_preprocessing.csv')

X_train = train.to_numpy()[:,1:]
X_test = test.to_numpy()[:,2:]
y = target.to_numpy()[:,1]
n_classifier = 5
n_feature = X_train.shape[1]

def f(x):
    x_train,x_test,y_train, y_test = train_test_split(
        X_train, y, test_size=0.2)
   
    E = []
    for i in range(n_classifier) :
        x_train *= x[n_feature*i:n_feature*(i+1)]
        x_test *= x[n_feature*i:n_feature*(i+1)]
        clf = LogisticRegression()
        clf.fit(x_train, y_train)
        error = np.sum(y_test-clf.predict(x_test))**2
   
        E.append(error)
    E = np.array(E)
   
    return np.sum(E)


algorithm_param = {'max_num_iteration': 100,\
                   'population_size':100,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}


model=ga(function=f,dimension=n_feature*n_classifier,variable_type='bool',
         algorithm_parameters=algorithm_param)
model.run()

 