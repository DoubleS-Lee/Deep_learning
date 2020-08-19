import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 적용
plt.rc('font', family='NanumBarunGothic') 
# 캔버스 사이즈 적용
plt.rcParams["figure.figsize"] = (12, 9)

## 통계 기반의 시각화를 제공해주는 Seaborn

# `seaborn` 라이브러리가 매력적인 이유는 바로 **통계 차트** 입니다.

# 이번 실습에서는 `seaborn`의 다양한 통계 차트 중 대표적인 차트 몇 개를 뽑아서 다뤄볼 예정입니다.

# 더 많은 통계 차트를 경험해보고 싶으신 분은 [공식 도큐먼트](https://seaborn.pydata.org/api.html)에서 확인하실 수 있습니다.

titanic = sns.load_dataset('titanic')
titanic

# * survived: 생존여부
# * pclass: 좌석등급
# * sex: 성별
# * age: 나이
# * sibsp: 형제자매 + 배우자 숫자
# * parch: 부모자식 숫자
# * fare: 요금
# * embarked: 탑승 항구
# * class: 좌석등급 (영문)
# * who: 사람 구분
# * deck: 데크
# * embark_town: 탑승 항구 (영문)
# * alive: 생존여부 (영문)
# * alone: 혼자인지 여부

tips = sns.load_dataset('tips')
tips

# * total_bill: 총 합계 요금표
# * tip: 팁
# * sex: 성별
# * smoker: 흡연자 여부
# * day: 요일
# * time: 식사 시간
# * size: 식사 인원

## 1. Countplot

# 항목별 갯수를 세어주는 `countplot` 입니다.

# 알아서 해당 column을 구성하고 있는 value들을 구분하여 보여줍니다.

# [countplot 공식 도큐먼트](https://seaborn.pydata.org/generated/seaborn.countplot.html)

# 배경을 darkgrid 로 설정
sns.set(style='darkgrid')

### 1-1 세로로 그리기

sns.countplot(x="class", hue="who", data=titanic)
plt.show()

### 1-2. 가로로 그리기

sns.countplot(y="class", hue="who", data=titanic)
plt.show()

### 1-3. 색상 팔레트 설정

sns.countplot(x="class", hue="who", palette='copper', data=titanic)
plt.show()

## 2. distplot

# matplotlib의 `hist` 그래프와 `kdeplot`을 통합한 그래프 입니다.

# **분포**와 **밀도**를 확인할 수 있습니다.

# [distplot 도큐먼트](https://seaborn.pydata.org/generated/seaborn.distplot.html?highlight=distplot#seaborn.distplot)

# 샘플데이터 생성
x = np.random.randn(100)
x

### 2-1. 기본 distplot  

sns.distplot(x)
plt.show()

### 2-2. 데이터가 Series 일 경우

x = pd.Series(x, name="x variable")
x

sns.distplot(x)
plt.show()

### 2-3. rugplot

# `rug`는 `rugplot`이라고도 불리우며, 데이터 위치를 x축 위에 **작은 선분(rug)으로 나타내어 데이터들의 위치 및 분포**를 보여준다.

sns.distplot(x, rug=True, hist=False, kde=True)
plt.show()

### 2-4. kde (kernel density)

# `kde`는 histogram보다 **부드러운 형태의 분포 곡선**을 보여주는 방법

sns.distplot(x, rug=False, hist=False, kde=True)
plt.show()

### 2-5. 가로로 표현하기

 sns.distplot(x, vertical=True)

### 2-6. 컬러 바꾸기

sns.distplot(x, color="y")
plt.show()

## 3. heatmap

# 색상으로 표현할 수 있는 다양한 정보를 **일정한 이미지위에 열분포 형태의 비쥬얼한 그래픽**으로
# 출력하는 것이 특징이다

[heatmap 도큐먼트](https://seaborn.pydata.org/generated/seaborn.heatmap.html?highlight=heatmap#seaborn.heatmap)

### 3-1. 기본 heatmap

uniform_data = np.random.rand(10, 12)
sns.heatmap(uniform_data, annot=True)
plt.show()

### 3-2. pivot table을 활용하여 그리기

tips

pivot = tips.pivot_table(index='day', columns='size', values='tip')
pivot

sns.heatmap(pivot, cmap='Blues', annot=True)
plt.show()

### 3-3. correlation(상관관계)를 시각화

# **corr()** 함수는 데이터의 상관관계를 보여줍니다.

titanic.corr()

sns.heatmap(titanic.corr(), annot=True, cmap="YlGnBu")
plt.show()

## 4. pairplot

# [pairplot 도큐먼트](https://seaborn.pydata.org/generated/seaborn.pairplot.html?highlight=pairplot#seaborn.pairplot)

# pairplot은 그리도(grid) 형태로 각 **집합의 조합에 대해 히스토그램과 분포도**를 그립니다.

# 또한, 숫자형 column에 대해서만 그려줍니다.

tips.head()

### 4-1. 기본 pairplot 그리기

sns.pairplot(tips)
plt.show()

### 4-2. hue 옵션으로 특성 구분

sns.pairplot(tips, hue='size')
plt.show()

### 4-3. 컬러 팔레트 적용

sns.pairplot(tips, hue='size', palette="rainbow")
plt.show()

### 4-4. 사이즈 적용

sns.pairplot(tips, hue='size', palette="rainbow", height=5,)
plt.show()

## 5. violinplot

# 바이올린처럼 생긴 violinplot 입니다.

# column에 대한 **데이터의 비교 분포도**를 확인할 수 있습니다.

# * 곡선진 부분 (뚱뚱한 부분)은 데이터의 분포를 나타냅니다.
# * 양쪽 끝 뾰족한 부분은 데이터의 최소값과 최대값을 나타냅니다.

# [violinplot 도큐먼트](https://seaborn.pydata.org/generated/seaborn.violinplot.html)

### 5-1. 기본 violinplot 그리기

sns.violinplot(x=tips["total_bill"])
plt.show()

### 5-2. 비교 분포 확인

# x, y축을 지정해줌으로썬 바이올린을 분할하여 **비교 분포**를 볼 수 있습니다.

sns.violinplot(x="day", y="total_bill", data=tips)
plt.show()

### 5-3. 가로로 뉘인 violinplot

sns.violinplot(y="day", x="total_bill", data=tips)
plt.show()

### 5-4. hue 옵션으로 분포 비교

# 사실 hue 옵션을 사용하지 않으면 바이올린이 대칭이기 때문에 비교 분포의 큰 의미는 없습니다.

# 하지만, hue 옵션을 주면, **단일 column에 대한 바이올린 모양의 비교**를 할 수 있습니다.

sns.violinplot(x="day", y="total_bill", hue="smoker", data=tips, palette="muted")
plt.show()

### 5-5. split 옵션으로 바이올린을 합쳐서 볼 수 있습니다.

sns.violinplot(x="day", y="total_bill", hue="smoker", data=tips, palette="muted", split=True)
plt.show()

## 6. lmplot

# `lmplot`은 column 간의 **선형관계를 확인하기에 용이한 차트**입니다.

# 또한, **outlier**도 같이 짐작해 볼 수 있습니다.

# [lmplot 도큐먼트](https://seaborn.pydata.org/generated/seaborn.lmplot.html)

### 6-1. 기본 lmplot

sns.lmplot(x="total_bill", y="tip", height=8, data=tips)
plt.show()

### 6-2. hue 옵션으로 다중 선형관계 그리기

# 아래의 그래프를 통하여 비흡연자가, 흡연자 대비 좀 더 **가파른 선형관계**를 가지는 것을 볼 수 있습니다.

sns.lmplot(x="total_bill", y="tip", hue="smoker", height=8, data=tips)
plt.show()

### 6-3. col 옵션을 추가하여 그래프를 별도로 그려볼 수 있습니다

# 또한, `col_wrap`으로 한 줄에 표기할 column의 갯수를 명시할 수 있습니다.

sns.lmplot(x='total_bill', y='tip', hue='smoker', col='day', col_wrap=2, height=6, data=tips)
plt.show()

## 7. relplot

# 두 column간 상관관계를 보지만 `lmplot`처럼 선형관계를 따로 그려주지는 않습니다.

# [relplot 도큐먼트](https://seaborn.pydata.org/generated/seaborn.relplot.html?highlight=relplot#seaborn.relplot)

### 7-1. 기본 relplot

sns.relplot(x="total_bill", y="tip", hue="day", data=tips)
plt.show()

### 7-2. col 옵션으로 그래프 분할

sns.relplot(x="total_bill", y="tip", hue="day", col="time", data=tips)
plt.show()

### 7-3. row와 column에 표기할 데이터 column 선택

sns.relplot(x="total_bill", y="tip", hue="day", row="sex", col="time", data=tips)
plt.show()

### 7-4. 컬러 팔레트 적용

sns.relplot(x="total_bill", y="tip", hue="day", row="sex", col="time", palette='CMRmap_r', data=tips)
plt.show()

## 8. jointplot

# scatter(산점도)와 histogram(분포)을 동시에 그려줍니다.

# 숫자형 데이터만 표현 가능하니, 이 점 유의하세요.

# [jointplot 도큐먼트](https://seaborn.pydata.org/generated/seaborn.jointplot.html?highlight=jointplot#seaborn.jointplot)

### 8-1. 기본 jointplot 그리기

sns.jointplot(x="total_bill", y="tip", height=8, data=tips)
plt.show()

### 8-2. 선형관계를 표현하는 regression 라인 그리기

# 옵션에 **kind='reg'**을 추가해 줍니다.

sns.jointplot("total_bill", "tip", height=8, data=tips, kind="reg")
plt.show()

### 8-3. hex 밀도 보기

# **kind='hex'** 옵션을 통해 hex 모양의 밀도를 확인할 수 있습니다.

sns.jointplot("total_bill", "tip", height=8, data=tips, kind="hex")
plt.show()

### 8-4. 등고선 모양으로 밀집도 확인하기

# **kind='kde'** 옵션으로 데이터의 밀집도를 보다 부드러운 선으로 확인할 수 있습니ㅏ.

iris = sns.load_dataset('iris')
sns.jointplot("sepal_width", "petal_length", height=8, data=iris, kind="kde", color="g")
plt.show()

