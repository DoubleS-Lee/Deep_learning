# pyplot 공식 문서
# https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#한글 폰트 깨짐 현상 해결방법
# matplotlib 폰트설정
plt.rc('font', family='NanumGothic') # For Windows
print(plt.rcParams['font.family'])
plt.rcParams["figure.figsize"] = (12, 9)
#%matplotlib inline
# 브라우저에서 바로 이미지를 그린다.

#%%
# sctterplot
# x,y,colors,area 설정하기
x = np.random.rand(50)
y = np.random.rand(50)
colors = np.arange(50)
area = x * y * 1000
plt.scatter(x, y, s=area, c=colors)

#%%
# cmap(컬러지정)과 alpha(투명도(0~1))
np.random.rand(50)
plt.figure(figsize=(12,6))
plt.subplot(1,3,1)
plt.scatter(x, y, s=area, cmap='blue', alpha=0.1)
plt.title('alpha = 0.1')

plt.subplot(1,3,2)
plt.scatter(x, y, s=area, cmap='blue', alpha=0.5)
plt.title('alpha = 0.5')

plt.subplot(1,3,3)
plt.scatter(x, y, s=area, cmap='blue', alpha=1.0)
plt.title('alpha = 1.0')

#%%
# barplot
# 기본 Barplot 그리기
x = ['Math', 'Programming', 'Data Science', 'Art', 'English', 'Physics']
y = [66, 80, 60, 50, 80, 10]

plt.figure(figsize=(6, 3))
# plt.bar(x, y)
plt.bar(x, y, align='center', alpha=0.7, color='red')
plt.xticks(x)
plt.ylabel('Number of Students')
plt.title('Subjects')

plt.show()

#%%
# 기본 Barhplot 그리기
# barh 함수에서는 xticks로 설정했던 부분을 yticks로 변경합니다.
x = ['Math', 'Programming', 'Data Science', 'Art', 'English', 'Physics']
y = [66, 80, 60, 50, 80, 10]

plt.barh(x, y, align='center', alpha=0.7, color='green')
plt.yticks(x)
plt.xlabel('Number of Students')
plt.title('Subjects')

plt.show()

#%%
# Batplot에서 비교 그래프 그리기
x_label = ['Math', 'Programming', 'Data Science', 'Art', 'English', 'Physics']
x = np.arange(len(x_label))
y_1 = [66, 80, 60, 50, 80, 10]
y_2 = [55, 90, 40, 60, 70, 20]

# 넓이 지정
width = 0.35

# subplots 생성
fig, axes = plt.subplots()

# 넓이 설정
axes.bar(x - width/2, y_1, width, align='center', alpha=0.5)
axes.bar(x + width/2, y_2, width, align='center', alpha=0.8)

# xtick 설정
plt.xticks(x)
axes.set_xticklabels(x_label)
plt.ylabel('Number of Students')
plt.title('Subjects')

plt.legend(['john', 'peter'])

plt.show()

#%%
x_label = ['Math', 'Programming', 'Data Science', 'Art', 'English', 'Physics']
x = np.arange(len(x_label))
y_1 = [66, 80, 60, 50, 80, 10]
y_2 = [55, 90, 40, 60, 70, 20]

# 넓이 지정
width = 0.35

# subplots 생성
fig, axes = plt.subplots()

# 넓이 설정
axes.barh(x - width/2, y_1, width, align='center', alpha=0.5, color='green')
axes.barh(x + width/2, y_2, width, align='center', alpha=0.8, color='red')

# xtick 설정
plt.yticks(x)
axes.set_yticklabels(x_label)
plt.xlabel('Number of Students')
plt.title('Subjects')

plt.legend(['john', 'peter'])

plt.show()


#%%
# Line Plot
# 기본 lineplot 그리기
x = np.arange(0, 10, 0.1)
y = 1 + np.sin(x)

plt.plot(x, y)

plt.xlabel('x value', fontsize=15)
plt.ylabel('y value', fontsize=15)
plt.title('sin graph', fontsize=18)

plt.grid()

plt.show()

#%%
#2개 이상의 그래프 그리기
x = np.arange(0, 10, 0.1)
y_1 = 1 + np.sin(x)
y_2 = 1 + np.cos(x)

plt.plot(x, y_1, label='1+sin', color='blue', alpha=0.3)
plt.plot(x, y_2, label='1+cos', color='red', alpha=0.7)

plt.xlabel('x value', fontsize=15)
plt.ylabel('y value', fontsize=15)
plt.title('sin and cos graph', fontsize=18)

plt.grid()
plt.legend()

plt.show()

#%%
# 마커 스타일링
x = np.arange(0, 10, 0.1)
y_1 = 1 + np.sin(x)
y_2 = 1 + np.cos(x)

plt.plot(x, y_1, label='1+sin', color='blue', alpha=0.3, marker='o')
plt.plot(x, y_2, label='1+cos', color='red', alpha=0.7, marker='+')

plt.xlabel('x value', fontsize=15)
plt.ylabel('y value', fontsize=15)
plt.title('sin and cos graph', fontsize=18)

plt.grid()
plt.legend()

plt.show()

#%%
# 라인 스타일 변경하기
# linestyle : 라인 스타일 변경 옵션
x = np.arange(0, 10, 0.1)
y_1 = 1 + np.sin(x)
y_2 = 1 + np.cos(x)

plt.plot(x, y_1, label='1+sin', color='blue', linestyle=':')
plt.plot(x, y_2, label='1+cos', color='red', linestyle='-.')

plt.xlabel('x value', fontsize=15)
plt.ylabel('y value', fontsize=15)
plt.title('sin and cos graph', fontsize=18)

plt.grid()
plt.legend()

plt.show()

#%%
# Areaplot(filled Area)
# matplotlib에서 area plot을 그리고자 할 때는 fill_between 함수를 사용합니다.
x = np.arange(1,21)
y =  np.random.randint(low=5, high=10, size=20)

# fill_between으로 색칠하기
plt.fill_between(x, y, color="green", alpha=0.6)
plt.show()

#%%
x = np.arange(1,21)
y = np.random.randint(low=5, high=10, size=20)
# 경계선을 굵게 그리고 area는 옅게 그리는 효과 적용
plt.fill_between( x, y, color="green", alpha=0.3)
# 경계선 옵션
plt.plot(x, y, color="green", alpha=0.8)

#%%
# 여러 그래프를 겹쳐서 표현
x = np.arange(1, 10, 0.05)
y_1 =  np.cos(x)+1
y_2 =  np.sin(x)+1
y_3 = y_1 * y_2 / np.pi

plt.fill_between(x, y_1, color="green", alpha=0.1)
plt.fill_between(x, y_2, color="blue", alpha=0.2)
plt.fill_between(x, y_3, color="red", alpha=0.3)


#%%
# Histogram
# 기본 histogram
N = 100000
# bins 구간 개수 설정
bins = 30

x = np.random.randn(N)

plt.hist(x, bins=bins)

plt.show()

#%%
# 다중 Histogram
N = 100000
bins = 30

x = np.random.randn(N)

# sharey : y축을 다중그래프가 share
# tight_layout : graph의 패딩을 자동으로 조절해주어 fit한 graph를 생성
fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)

fig.set_size_inches(12, 5)

axs[0].hist(x, bins=bins)
axs[1].hist(x, bins=bins*2)
axs[2].hist(x, bins=bins*4)

plt.show()

#%%
# y축에 Density 표기
N = 100000
bins = 30

x = np.random.randn(N)

fig, axs = plt.subplots(1, 2, tight_layout=True)
fig.set_size_inches(9, 3)

# density=True 값을 통하여 Y축에 density를 표기할 수 있습니다.
axs[0].hist(x, bins=bins, density=True, cumulative=True)
axs[1].hist(x, bins=bins, density=True)

plt.show()

#%%
# Pie Chart
# pie chart 옵션

# explode: 파이에서 툭 튀어져 나온 비율
# autopct: 퍼센트 자동으로 표기
# shadow: 그림자 표시
# startangle: 파이를 그리기 시작할 각도
# texts, autotexts 인자를 리턴 받습니다.

# texts는 label에 대한 텍스트 효과를

# autotexts는 파이 위에 그려지는 텍스트 효과를 다룰 때 활용합니다.

labels = ['Samsung', 'Huawei', 'Apple', 'Xiaomi', 'Oppo', 'Etc']
sizes = [20.4, 15.8, 10.5, 9, 7.6, 36.7]
explode = (0.3, 0, 0, 0, 0, 0)

# texts, autotexts 인자를 활용하여 텍스트 스타일링을 적용합니다
patches, texts, autotexts = plt.pie(sizes, explode=explode, labels=labels,  autopct='%1.1f%%',shadow=True, startangle=90)
plt.title('Smartphone pie', fontsize=15)

# label 텍스트에 대한 스타일 적용
for t in texts:
    t.set_fontsize(12)
    t.set_color('gray')
    
# pie 위의 텍스트에 대한 스타일 적용
for t in autotexts:
    t.set_color("white")
    t.set_fontsize(18)

plt.show()

#%%
# Box plot
from IPython.display import Image
Image('https://justinsighting.com/wp-content/uploads/2016/12/boxplot-description.png')


# 샘플 데이터 생성
spread = np.random.rand(50) * 100
center = np.ones(25) * 50
flier_high = np.random.rand(10) * 100 + 100
flier_low = np.random.rand(10) * -100
data = np.concatenate((spread, center, flier_high, flier_low))

# 기본 박스플롯 생성
plt.boxplot(data)
plt.tight_layout()
plt.show()

#%%
# 다중 박스플롯 생성
# 샘플 데이터 생성
spread = np.random.rand(50) * 100
center = np.ones(25) * 50
flier_high = np.random.rand(10) * 100 + 100
flier_low = np.random.rand(10) * -100
data = np.concatenate((spread, center, flier_high, flier_low))

spread = np.random.rand(50) * 100
center = np.ones(25) * 40

flier_high = np.random.rand(10) * 100 + 100
flier_low = np.random.rand(10) * -100

d2 = np.concatenate((spread, center, flier_high, flier_low))

data.shape = (-1, 1)
d2.shape = (-1, 1)

data = [data, d2, d2[::2,0]]

# boxplot()으로 매우 쉽게 생성할 수 있습니다.

# 다중 그래프 생성을 위해서는 data 자체가 2차원으로 구성되어 있어야 합니다.

# row와 column으로 구성된 DataFrame에서 Column은 X축에 Row는 Y축에 구성된다고 이해하시면 됩니다.

plt.boxplot(data)
plt.show()

#%%
# Box Plot 축 바꾸기
# vert=False 옵션을 통해 표시하고자 하는 축을 바꿀 수 있습니다.
plt.title('Horizontal Box Plot', fontsize=15)
plt.boxplot(data, vert=False)

plt.show()


#%%
# Outlier 마커 심볼과 컬러 변경
outlier_marker = dict(markerfacecolor='r', marker='D')
plt.title('Changed Outlier Symbols', fontsize=15)
plt.boxplot(data, flierprops=outlier_marker)

plt.show()

#%%
# 3D 그래프 그리기
# 3d 로 그래프를 그리기 위해서는 mplot3d를 추가로 import 합니다

from mpl_toolkits import mplot3d

#밑그림 그리기 (캔버스)
fig = plt.figure()
ax = plt.axes(projection='3d')

#%%
# 3d plot 그리기
# project=3d로 설정합니다
ax = plt.axes(projection='3d')

# x, y, z 데이터를 생성합니다
z = np.linspace(0, 15, 1000)
x = np.sin(z)
y = np.cos(z)

ax.plot(x, y, z, 'gray')
plt.show()


#%%
# project=3d로 설정합니다
ax = plt.axes(projection='3d')

sample_size = 100
x = np.cumsum(np.random.normal(0, 1, sample_size))
y = np.cumsum(np.random.normal(0, 1, sample_size))
z = np.cumsum(np.random.normal(0, 1, sample_size))

ax.plot3D(x, y, z, alpha=0.6, marker='o')

plt.title("ax.plot")
plt.show()

#%%
# 3d-scatter 그리기
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d') # Axe3D object

sample_size = 500

x = np.cumsum(np.random.normal(0, 5, sample_size))
y = np.cumsum(np.random.normal(0, 5, sample_size))
z = np.cumsum(np.random.normal(0, 5, sample_size))

ax.scatter(x, y, z, c = z, s=20, alpha=0.5, cmap='Greens')

plt.title("ax.scatter")
plt.show()

#%%
# contour3D 그리기 (등고선)
x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)
x, y = np.meshgrid(x, y)

z = np.sin(np.sqrt(x**2 + y**2))

fig = plt.figure(figsize=(12, 6))
ax = plt.axes(projection='3d')

ax.contour3D(x, y, z, 20, cmap='Reds')

plt.title("ax.contour3D")
plt.show()

#%%
# imshow
# 이미지(image) 데이터와 유사하게 행과 열을 가진 2차원의 데이터를 시각화 할 때는 imshow를 활용합니다.

from sklearn.datasets import load_digits

digits = load_digits()
X = digits.images[:10]
X.shape
X[0]

# load_digits는 0~16 값을 가지는 array로 이루어져 있습니다.
# 1개의 array는 8 X 8 배열 안에 표현되어 있습니다.
# 숫자는 0~9까지 이루어져있습니다.

fig, axes = plt.subplots(nrows=2, ncols=5, sharex=True, figsize=(12, 6), sharey=True)

for i in range(10):
    axes[i//5][i%5].imshow(X[i], cmap='Blues')
    axes[i//5][i%5].set_title(str(i), fontsize=20)
    
plt.tight_layout()
plt.show()










