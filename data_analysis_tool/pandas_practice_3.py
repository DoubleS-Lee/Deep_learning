import pandas as pd

df = pd.read_csv('korean-idol.csv')

df.info()

###################################################################
#type 변환
# .astype()
#series 별로 데이터를 변환해준다
df['키'].dtypes

df['키'].astype(int)
#키에는 NaN 값이 2개가 있어서 데이터 변환이 안됨
df['키']=df['키'].fillna(df['키'].mean())

df['키'].astype(int)


###################################################################
# datetime : 날짜 타입을 변환하기
# 판다스에서 날짜 타입을 변환하려면 to_datetime 메소드를 사용한다
df['생년월일']
df['생년월일'] = pd.to_datetime(df['생년월일'])
df['생년월일']

df['생년월일'].dt
df['생년월일'].dt.year
df['생년월일'].dt.month
df['생년월일'].dt.day
df['생년월일'].dt.minute
df['생년월일'].dt.dayofweek
df['생년월일'].dt.weekofyear

df.head()
df['생일_년'] = df['생년월일'].dt.year
df['생일_월'] = df['생년월일'].dt.month
df['생일_일'] = df['생년월일'].dt.day
df

###################################################################
# apply : series나 dataframe에 좀 더 구체적인 로직을 적용하고 싶은 경우 활용한다

df.loc[df['성별'] == '남자', '성별'] = 1
df.loc[df['성별'] == '여자', '성별'] = 0

def male_or_female(x):
    if x =='남자':
        return 1
    else x =='여자':
        return 0

df['성별'].apply(male_or_female)

def cm_to_brand(df):
    value = df['브랜드평판지수'] / df['키']
    return value
df.apply(cm_to_brand, axis=1)

###################################################################
# map : 기존에 존재하는 값을 새롭게 매핑한다
my_map = {'남자':1,'여자':0}

df['성별'].map(my_map)


###################################################################
import numpy as np

df = pd.DataFrame({'통계':[60,70,80,85,75], '미술':[50,55,80,100,95],'체육':[70,65,50,95,100]})
df
df['통계']
type(df['통계'] )
df['통계'] + df['미술']
df['통계'] - df['미술']
df['통계'] * df['미술']
df['통계'] / df['미술']
df['통계'] % df['미술']

df['통계'] + 10
df['통계'] - 10
df['통계'] * 10
df['통계'] / 10
df['통계'] % 10

df
df.mean(axis=0)
df.mean(axis=1)
df.sum(axis=0)
df.sum(axis=1)


df1 = pd.DataFrame({'통계':[60,70,80,85,75], '미술':[50,55,80,100,95],'체육':[70,65,50,95,100]})
df2 = pd.DataFrame({'통계':['good', 'bad', 'ok', 'good', 'ok'],'미술':[50,60,80,100,95],'체육':[70,65,50,70,100]})
df1
df2
df1 + df2
df1 + 10
df2 + 10

###################################################################
# 특정타입의 데이터들을 뽑아낼때 사용
df = pd.read_csv('korean-idol.csv')
df.head()
df.info()

df.select_dtypes(include='object')

df.select_dtypes(exclude='object')

df.select_dtypes(include='object') + 10

df.select_dtypes(exclude='object') + 10

cols = df.select_dtypes(exclude='object')
cols2 = df.select_dtypes(exclude='object').columns
df[cols2]

###################################################################
# 원핫인코딩
# 한개의 요소는 True 나머지 요소는 False로 만들어주는 기법
df.head()

blood_map = {'A':0, 'B':1, 'AB':2, 'O':3}
df['혈액형_code'] = df['혈액형'].map(blood_map)
df

df.head()
df['혈액형_code'].value_counts()

pd.get_dummies(df['혈액형_code'])

pd.get_dummies(df['혈액형_code'], prefix='혈액형')





