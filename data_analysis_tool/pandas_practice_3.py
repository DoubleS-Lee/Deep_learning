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
    if x=='남자':
        return 1
    else x=='여자':
        return 0

df['성별'].apply(male_or_female)

def cm_to_brand(df):
    value = df['브랜드평판지수'] / df['키']
    return value
df.apply(cm_to_brand, axis=1)

###################################################################
# map : 값을 새롭게 매핑한다
my_map = {'남자':1,'여자':0}

df['성별'].map(my_map)


















