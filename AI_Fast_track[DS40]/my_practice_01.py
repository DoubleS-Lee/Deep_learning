
#%%
복습
1. AI(Artifitial Intelligence) : Rule base 인공지능(소프트웨어 1.0) : 사람이 직접 룰을 만들어서 답을 낸다
2. 머신러닝 : 데이터를 줬을때 컴퓨터가 룰을 찾아가는것
3. 딥러닝 : 데이터를 줬을때 컴퓨터가 룰을 찾아가는것

1. 지도학습 : 입력과 출력사이의 관계 찾기 : 분류, 회귀
2. 비지도학습 : 데이터의 분포를 알 수 있다 : 군집화, 특징추출, 모델생성
3. 강화학습 : 최적화의 관점 : 그 중에 GA라는게 있다

likelihood : 이런 식으로 사건이 일어날 것 같다

linear regression
k-means (실루엣기법)











#%%
공부할 것들

벡터
고유값(t)
AK = tK
고유벡터(K)
det(A-tI) = 0

기본적으로 행렬은 벡터를 변환시켜주는 연산자이다 (크기와 방향 모두)
vector * matrix = another vector

하지만 어떤 행렬은 벡터의 크기만 변환시키기도 한다
마치 그냥 상수(t)만 곱해준것 처럼
입력 벡터(K)를 A로 선형변환 시킨 결과(AK) 그 결과가 상수배라는 뜻이다(tK)

이때 어떤 행렬(A)에 대해 방향은 그대로고 크기만 변환하게 되는 벡터(K)가 존재한다
이를 고유벡터(K)라고 하고
이에 해당하는 스칼라 값을 고유값(t)이라고 한다

행렬의 성질에 따라 다음을 만족해야한다
det(A-tI) = 0
이때 행렬이 2x2행렬이라면 이 방정식을 풀면 t1과 t2가 나오게 된다.
이제 t1인 경우, t2인 경우에 대해 AK = tK 연립방정식을 풀면
고유벡터가 2개가 나온다
각각 t1인 경우의 고유벡터 K1, t2인 경우의 고유벡터 K2가 나온다

이 의미는 K1벡터는 여기에 행렬(A)를 취해주면 방향은 변하지 않고 크기가 t1배가 된다
이 의미는 K2벡터는 여기에 행렬(A)를 취해주면 방향은 변하지 않고 크기가 t2배가 된다

이 의미는 A행렬은 고유벡터가 존재하는데
이 고유벡터는 각각 t1배와 t2배가 된다

방향은 변하지 않고 크기만 변하게 하는 벡터

 

확률이론
bayesian
likelihood

kalman 필터 = 노이즈의 패턴을 찾아내서 이를 기반으로 노이즈를 필터치는 필터링 방법

PCA
LDA

numpy
pandas
matplotlib


#%%
수업내용


벡터
1. 데이터 - 벡터로 만들어줘야함(크기, 방향)
2. 거리 - 얼만큼 떨어져있는가 = 유사도(Kmean, KNN)
3. y=Ax 고유값, 고유벡터

확률, 통계
통계적 개념
매번 일어나는 사건은 예측이 불가능하지만 일정치 이상 쌓이게 되면 일정한 분포를 따른다

수학적 확률
발생할 가능성을 기반으로 확률을 구함

통계적 확률
실제로 일어난 횟수를 기반으로 확률을 구한것

%% n이 무한대로 갈수록 수학적 확률과 통계적 확률이 같아진다

covariance(=공분산) = x와 y의 상관관계 (=퍼짐도, 산포도를 나타냄)
correlation = 공분산을 정규화(nomalization) 해줌

독립사건
이전의 시행의 결과가 다음 시행의 결과에 영향을 미치지 않는 것

종속사건
이전의 시행의 결과가 다음 시행의 결과에 영향을 미치는 것

마르코프 프로세스
다음 사건이 발생할때 전 사건만 관여를 하고 전전 사건은 관여를 하지 않는다


bayesian 정리
사전확률과 사후확률 사이의 관계를 나타내는 정리
새로운 정보를 토대로 어떤 사건이 발생했다는 주장에 대한 신뢰도를 갱신해 나가는 방법
주장에 대한 신뢰도
사전확률과 우도확률을 안다면 사후확률을 알 수 있다
P(H|E) = {P(E|H)*P(H)}/P(E)
P(H|E) : 사후확률
P(H) : 사전확률
P(E|H) : 우도확률
H : Hypothesis (경험)
E : Evidence(증거)

evidence를 관측하여 갱신하기 전 후의 내 주장에 대한 신뢰도
확률론의 패러다임을 전환했다(연역적추론 -> 귀납적추론으로)
베이지안 관점의 통계학은 사전 확률과 같은 경험에 기반한 선험적인,
혹은 불확실성을 내포하는 수치를 기반으로 하고 거기에 추가 정보를 바탕으로 사전 확률을 갱신한다


bayesizn optimizer
sequential bayesian filter = kalman filter

 


사후확률 = likelihood * 사전확률

likelihood(우도) : 가능도 = ~할것같다
동전을 10번 던졌을때 앞면이 반드시 5번 나오라는 법은 없지만 그럴 가능성이 가장 높다
100% 이렇게 된다는 법은 없지만 이럴 가능성이 아주 높다
확률분포가 아니기 때문에 다 합쳤을때 1이 아닐수 있다
확률과 가능도의 차이
확률은 전체 파라미터 값들이 다 주어지고 여기서 랜덤변수 x가 될 확률을 구하는 것
가능도는 x라는 랜덤 변수가 주어졌을때 파라미터의 확률이 된다
x라는 랜덤변수가 주어졌다는 것은 전체 모집단에 대한 랜덤변수x를 모두 안다는 것이 불가능하다는 뜻이다
그래서 x는 전체라기 보다는 측정하거나 알고있는 샘플들에 해당한다
따라서 해당 샘플이 있을때 파라미터들이 될 수 있는 확률이 가능도이다

확률이 모델 파라미터값이 관측 데이터 없이 주어진 상태에서 랜덤한  출력이 일어날 가능성이라면,
가능도(likelihood) 는 특정 관측 결과가 주어진 상태에서 모델 파라미터 값들이 나타날 가능성이다.

확률과 가능도는 반대되는 개념이 된다.
확률이 모 파라미터를 알고 랜덤 변수 X 를 미지수로 표현되는 식이라면
가능도는 관측치 X 가 주어진 상황에서 파라미터 \thetaθ 를 변수로하는 식으로 표현된다.

동전을 찌그러트리면 앞면이 몇의 확률로 나올지 모른다
그래서 실제로 이걸 다 던져보고 이 결과를 가지고 가능도를 추정한다

Maximum Likelihood Estimator
반드시 그런 확률로 결과가 나올지는 모르겠지만 가장 그럴것 같은 결과를 나타내준다
결과를 보고 가장 합리적인 결과를 나타내준다
어떤 현상에 대해 데이터를 모았을때 이게 일어날 확률
내가 가진 데이터셋을 가장 잘 표현하는 확률분포가 어떤건지 찾아주는 것
데이터의 분포를 끄집어낸다

가우시안확률밀도함수
적분이 된다


공분산

#%%
정방행렬이 아니어서 역행렬이 안 구해지면
pseudo Inverse를 만들어야함

#%%
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = 1 * x_0 + 2 * x_1 + 3
y = np.dot(X, np.array([1, 2])) + 3

reg = LinearRegression(fit_intercept = True)
reg.fit(X,y)

print(reg.score(X, y),
      reg.coef_ ,
      reg.intercept_,
      reg.predict(np.array([[3, 5]])))

#%%
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# Create linear regression object : fill here
regr = linear_model.LinearRegression()

# Train the model using the training sets : fill here
regr.fit(diabetes_X_train,diabetes_y_train)

# Make predictions using the testing set : fill here
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)

# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

#%%
#overfitting vs Underfitting
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


def true_fun(X):
    return np.cos(1.5 * np.pi * X)

np.random.seed(0)

n_samples = 30
degrees = [1, 4, 15, 30]

X = np.sort(np.random.rand(n_samples))
y = true_fun(X) + np.random.randn(n_samples) * 0.1

plt.figure(figsize=(14, 5))
for i in range(len(degrees)):
    ax = plt.subplot(1, len(degrees), i + 1)
    plt.setp(ax, xticks=(), yticks=())

    polynomial_features = PolynomialFeatures(degree=degrees[i],
                                             include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    pipeline.fit(X[:, np.newaxis], y)

    # Evaluate the models using crossvalidation
    scores = cross_val_score(pipeline, X[:, np.newaxis], y,
                             scoring="neg_mean_squared_error", cv=10)

    X_test = np.linspace(0, 1, 100)
    plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
    plt.plot(X_test, true_fun(X_test), label="True function")
    plt.scatter(X, y, edgecolor='b', s=20, label="Samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim((0, 1))
    plt.ylim((-2, 2))
    plt.legend(loc="best")
    plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
        degrees[i], -scores.mean(), scores.std()))
plt.show()

 


#%%
# IRIS Example

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split


# np.random.seed(10) : random.seed 안에 숫자를 넣으면 그 뒤로 생성하는 랜덤 값은 동일하게 나오게 된다
np.random.seed(5)

iris = datasets.load_iris()
x = iris.data
y = iris.target

print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

iris_reg = linear_model.LinearRegression()
iris_reg.fit(x_train,y_train)
y_pred = iris_reg.predict(x_test)

print(y_test,y_pred)

y_pred = np.round(iris_reg.predict(x_test))

print(y_test-y_pred)


#%%
# titanic Example
import pandas as pd
from sklearn import datasets
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

train = pd.read_csv('C:/Users/6533567/Documents/AI_fast_track_DS40/02.실습자료/titanic/train_pre.csv')
test = pd.read_csv('C:/Users/6533567/Documents/AI_fast_track_DS40/02.실습자료/titanic/test_pre.csv')
target = pd.read_csv('C:/Users/6533567/Documents/AI_fast_track_DS40/02.실습자료/titanic/train_label_pre.csv')
test_ori = pd.read_csv('C:/Users/6533567/Documents/AI_fast_track_DS40/02.실습자료/titanic/test.csv')

train.shape
test.shape
target.shape

titanic_reg = linear_model.LinearRegression()
titanic_reg.fit(train, target)
y_pred = titanic_reg.predict(test)

y_pred_1 = np.round(y_pred).astype(np.int)

submission = pd.DataFrame({'passengerId':test_ori['PassengerId'],'Survived':y_pred_1[:,1]})

submission.to_csv('titanic/submission_lee.csv', index=False)


#%%
Kmeans 클러스터링
학습 : 각 클러스터의 중심점을 찾는다(클러스터에 속하는 점들을 중심으로 평균을 구해서 찾는다), 이를 기반으로 각 클러스터의 경계를 찾는다

kmeans 클러스터링에서의 distance는 유사도 라고 할 수 있다

k개의 클러스터의 중심점을 찾게 되면 그 중심을 기준으로 경계를 만들게 된다

clustering = data의 distribution을 알게 된다

Kmeans 클러스터링의 원리와 작동 알고리즘
1. 각 데이터 포인트 i에 대해 가장 가까운 중심점을 찾고, 그 중심점에 해당하는 군집 할당
2. 중심점 업데이트 : 할당된 군집을 기반으로 새로운 중심 계산, 중심점은 군집 내부 점들 좌표의 평균(mean)으로 함
3. 각 클러스터의 할당이 바뀌지 않을 때까지 반복

Kmeans 클러스터링의 단점
1. 초기 중심 값에 민감한 반응을 보임
2. 노이즈와 아웃라이어에 민감함
3. 군집의 개수 K를 설정하는 것이 어려움

최적 클러스터의 개수

결과가 잘 됐는지 안 됐는지 어떻게 알지

클러스터의 중심점을 찾는 것이 kmeans 클러스터링의 최종 목적


많은 데이터들이 한 클러스터에는 거리가 짧고 다른 클러스터와는 거리가 멀수록 군집화가 잘 됐다고 평가

. 파라미터 업데이트를 위한 loss 값
 - 실루엣 파라미터 (k의 개수를 측정할 수 있는 방법)
1에 가까울수록 군집화가 잘 됐다,
자신의 클러스터 중심과는 가깝고 다른 집단인 클러스터의 중심과는 멀수록 군집화가 잘됐다


#%%
# K-means clustering
# IRIS Example

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# np.random.seed(10) : random.seed 안에 숫자를 넣으면 그 뒤로 생성하는 랜덤 값은 동일하게 나오게 된다
np.random.seed(5)

iris = datasets.load_iris()
x = iris.data
y = iris.target

print(x)
print(y)

kmeans = KMeans(n_clusters = 3,n_init=20, random_state=50)
kmeans.fit(x)

print(kmeans.labels_)

print(kmeans.labels_, y)
print(kmeans.labels_ - y)

print(kmeans.cluster_centers_)

#%%
GMM
클러스터링
Gaussian Mixture Model
Kmeans에서 각 cluster의 분포가 Gaussian 일때 이를 GMM 이라고 한다
(kmeans는 데이터의 분포가 뭔지모를때 사용)

데이터간 거리는 데이터간 상관도를 나타낸다

가우시안 분포

마할라노비스 디스턴스

Maximum Likelihood Estimator


GMM 학습후 평균과 공분산을 얻을 수 있다
각 데이터들이 가우시안에 속할 확률을 구할 수 있다


EM 방법
Expectation  Maximization 기법
E-step : conditional probabilities
M-step : parameters update

 

#%%
PCA
Principle Component Analysis(주성분 분석법)
주성분 분석법
: 고차원 특징 벡터를 저차원 특징 벡터로 축소하는 특징 벡터의 차원 축소 뿐만아니라 데이터 시각화
그리고 특징 추출에도 유용하게 사용되는 데이터 처리 기법 중의 하나임

데이터의 global한 특성을 찾기 위함

PCA의 목적 : 1) 데이터가 가장 잘 보이는 축을 찾는 것
 2) 데이터가 가장 많이 흩어져 있는 축
 3) 분산의 주축으로 전환
    데이터의 Global한 특성을 찾는데 유효하다
 - 분산이 위 내용을 대변해주니까 분산을 계산해야한다
 - 공분산의 고유벡터를 구해야한다
 - 고유값 중 일정치를 넘기는 큰 놈만 남기고 나머지는 지워서 큰 고유값이 포함하고 있는 축을 기준으로 만들어주면
    축 2개짜리 데이터를 축 1개로 표현할 수 있다
 
차원축소
속도가 빨라지는 대신 정확도가 떨어지게 된다

 

#%%
# PCA
# titanic Example
import pandas as pd
from sklearn import datasets
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

train = pd.read_csv('C:/Users/6533567/Documents/AI_fast_track_DS40/02.실습자료/titanic/train_pre.csv')
test = pd.read_csv('C:/Users/6533567/Documents/AI_fast_track_DS40/02.실습자료/titanic/test_pre.csv')
target = pd.read_csv('C:/Users/6533567/Documents/AI_fast_track_DS40/02.실습자료/titanic/train_label_pre.csv')
test_ori = pd.read_csv('C:/Users/6533567/Documents/AI_fast_track_DS40/02.실습자료/titanic/test.csv')

train.shape
test.shape
target.shape


pca = PCA(n_components=5)
pca.fit(train)
train = pca.transform(train)
print(pca.explained_variance_ratio_) 

pca = PCA(n_components=5)
pca.fit(test)
test = pca.transform(test)
print(pca.explained_variance_ratio_) 


titanic_reg = linear_model.LinearRegression()
titanic_reg.fit(train, target)
y_pred = titanic_reg.predict(test)

y_pred_1 = np.round(y_pred).astype(np.int)

submission = pd.DataFrame({'passengerId':test_ori['PassengerId'],'Survived':y_pred_1[:,1]})

print(y_pred_1)

 

#%%
Kernel approach
비선형성을 가지고 있는 데이터를 선형적으로 만들어 버리는 것

 

 

#%%
LDA
선형판별분석법
Linear Discriminant Analysis
PCA는 데이터가 가장 잘보이는 축, 분산의 주축으로 변환, 분산이 가장 큰 축을 찾는다
LDA는 데이터들을 잘 분리할 수 있는 축을 찾는다

데이터의 Local한 특성을 찾기 위한 방법

LDA의 목적: 1) 데이터를 잘 분류해주는 축을 찾는 것 (1번 데이터와 2번 데이터를 선으로 나눠줄수있게 하는 것)
    2) 데이터를 잘 나눌수있는 축을 찾는 것
LDA의 결과물 : 축
LDA는 비지도학습이다

LDA의 진행방향

LDA 계산의 목적
1),2)를 만족시키는 축을 찾는다
1) 다른 클래스의 데이터끼리는 떨어져있어야한다
따라서 평균을 크게하는 축을 찾는다
2) 같은 클래스끼리는 모여있어야한다
따라서 분산은 작게 되는 축을 찾는다

LDA의 분리 목표 : 클래스간 분산 최대 and 클래스 내부 분산 최소
   
   
#%%
# LDA
# titanic Example
import pandas as pd
from sklearn import datasets
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

train = pd.read_csv('C:/Users/6533567/Documents/AI_fast_track_DS40/02.실습자료/titanic/train_pre.csv')
test = pd.read_csv('C:/Users/6533567/Documents/AI_fast_track_DS40/02.실습자료/titanic/test_pre.csv')
target = pd.read_csv('C:/Users/6533567/Documents/AI_fast_track_DS40/02.실습자료/titanic/train_label_pre.csv')
test_ori = pd.read_csv('C:/Users/6533567/Documents/AI_fast_track_DS40/02.실습자료/titanic/test.csv')

train.shape
test.shape
target.shape

target = target.to_numpy()[:,1:]

lda = LDA(n_components=1)
lda.fit(train,target)
train = lda.transform(train)
print(lda.explained_variance_ratio_) 


titanic_reg = linear_model.LinearRegression()
titanic_reg.fit(train, target)
y_pred = titanic_reg.predict(test)

y_pred_1 = np.round(y_pred).astype(np.int)

submission = pd.DataFrame({'passengerId':test_ori['PassengerId'],'Survived':y_pred_1[:,1]})

print(y_pred_1)

   
#%%
KNN
K-Nearest Neighbor
지도학습
classification을 함

내 근처에 있는 놈들은 나와 비슷할 가능성이 크다

서로 가까운 점들은 유사하다는 가정
데이터셋에 내재된 패턴을 찾기 위해 데이터셋 전체를 봐야하지만
KNN은 내 주변에 점만 확인하면 되기 때문에 전체의 점을 볼 필요가 없음

K는 나로부터 가장 가까운 점의 개수를 말한다

#%%
# KNN
# IRIS Example

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier , NearestNeighbors
from sklearn.model_selection import train_test_split


# np.random.seed(10) : random.seed 안에 숫자를 넣으면 그 뒤로 생성하는 랜덤 값은 동일하게 나오게 된다

iris = datasets.load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

iris_knn = NearestNeighbors(n_neighbors=1)
iris_knn.fit(x_train, y_train)

print(x_test,y_test)


#%%
train
validation
test

N-fold cross validation


#%%
Naive Bayesian filter
classification

입력 데이터간의 상관성이 없는 경우(독립)에 사용하는게 Naive이다
입력 데이터간에 연관성이 강한 경우에는 사용하면 안된다

smoothing

더 많은 정보 활용
outlier 제거
clustering


#%%
Decision Tree
20고개 같은 개념
핵심
- 중요도가 높은 것들을 위에 올려줘야 적게 계산을 하면서 해답을 도출할 수 있다

중요도가 높은 인자를 선택하는 방법
- 엔트로피 (인자들의 확률이 고만고만하면 엔트로피는 커진다, 뭐가 나올지 모르기 때문(불확실성이 높다))
- gain index


#%%
Random Forest


#%%
Ensemble
























