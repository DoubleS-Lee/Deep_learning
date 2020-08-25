from sklearn import datasets
import pandas as pd

iris = datasets.load_iris()
labels = pd.DataFrame(iris.target)
labels.columns=['labels']
data = pd.DataFrame(iris.data)
data.columns=['Sepal length','Sepal width','Petal length','Petal width']
data = pd.concat([data,labels],axis=1)

data.head()


feature = data[ ['Sepal length','Sepal width','Petal length','Petal width']]
feature.head()

from sklearn.cluster import DBSCAN
import matplotlib.pyplot  as plt
import seaborn as sns

# create model and prediction
model = DBSCAN(eps=0.5,min_samples=5)
predict = pd.DataFrame(model.fit_predict(feature))
predict.columns=['predict']

# concatenate labels to df as a new column
r = pd.concat([feature,predict],axis=1)

print(r)

## DBSCAN 결과 시각화


#pairplot with Seaborn
sns.pairplot(r,hue='predict')
plt.show()

## 실제 데이터 시각화


#pairplot with Seaborn
sns.pairplot(data,hue='labels')
plt.show()

## Kmeans 결과와 비교 

from sklearn.cluster import KMeans
km = KMeans(n_clusters = 3, n_jobs = 4, random_state=21)
km.fit(feature)

new_labels =pd.DataFrame(km.labels_)
new_labels.columns=['predict']

r2 = pd.concat([feature,new_labels],axis=1)



#pairplot with Seaborn
sns.pairplot(r2,hue='predict')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
import time
%matplotlib inline
sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}

data = np.load('./data/clusterable_data.npy')

plt.scatter(data.T[0], data.T[1], c='b', **plot_kwds)
frame = plt.gca()
frame.axes.get_xaxis().set_visible(False)
frame.axes.get_yaxis().set_visible(False)

def plot_clusters(data, algorithm, args, kwds):
    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    end_time = time.time()
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
    plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)

plot_clusters(data, cluster.KMeans, (), {'n_clusters':3})

plot_clusters(data, cluster.KMeans, (), {'n_clusters':4})

plot_clusters(data, cluster.KMeans, (), {'n_clusters':5})

plot_clusters(data, cluster.KMeans, (), {'n_clusters':6})

plot_clusters(data, cluster.DBSCAN, (), {'eps':0.020})

plot_clusters(data, cluster.DBSCAN, (), {'eps':0.03})

dbs = DBSCAN(eps=0.03)
dbs2=dbs.fit(data)


dbs2.labels_

### HDBSCAN
#### DBSCAN의 발전된 버젼, 하이퍼 파라미터에 덜민감함 

import hdbscan

plot_clusters(data, hdbscan.HDBSCAN, (), {'min_cluster_size':45})

