import pandas as pd

df=pd.read_csv('korean-idol.csv')


###############################################################
# fillna : 비어있는 값을 채워주는 함수
df.info()

df2 = df

height = df2['키'].mean()
df2['키'] = df2['키'].fillna(height)
df2['키']


###############################################################
# 빈값(NaN)이 있는 행을 제거
df.dropna()
df.dropna(axis=0)
# 빈값이 있는 열을 제거
df.dropna(axis=1)

# NaN값이 한개라도 있으면 드랍
df.dropna(axis=0, how='any')

# 모두 NaN인 경우 드랍
df.dropna(axis=0, how='all')

###############################################################
# 중복값 제거
# .drop_duplicates() : 첫번째 나오는 값은 유지시키고 중복되는 2번째 값부터 제거시킨다
df['키'].drop_duplicates()
df['키'].drop_duplicates(keep='first')
df['키'].drop_duplicates(keep='last')

# 특정열에 대하여 중복된 데이터를 찾는다
df.drop_duplicates('그룹')
df.drop_duplicates('성별')

###############################################################
# drop : column이나 row를 제거
# column 제거
df
df.drop('그룹', axis=1)
df.drop(['그룹','소속사'],axis=1)

# row 제거
df.drop(3, axis=0)
df.drop([1,5], axis=0)

###############################################################
# DataFrame 힙치기
df2 = pd.read_csv('korean-idol-2.csv')
df
df2

#row 기준 합치기
# concat : row나 column 기준으로 단순하게 이어붙이기
df_copy = df.copy()
df_copy
df_concat = pd.concat([df,df_copy], sort=False)
# Index 정렬
df_concat.reset_index(drop=True)

#column 기준 합치기
df
df2
pd.concat([df,df2],axis=1)


# 병합하기 merge : 특정 고유한 키(unique id)값을 기준으로 병합하기
df
df2
df_right = df2.drop([1,3,5,7])
df_right = df_right.reset_index(drop=True)

#이름이라는 column이 겹치니까 이름을 기준으로 이 두 DataFrame을 병합할 수 있다
pd.concat([df, df_right], axis=1)
# 속성 how에서 left를 쓰면 left의 DataFrame에 있는 키값을 기준으로 병합하게 됨
# 여기서 right로 병합하면 df_right에 없는 키값들에 대한 df의 정보는 사라지게 된다
pd.merge(df, df_right, on='이름', how='left')
pd.merge(df, df_right, on='이름', how='right')

#inner와 outer 방식
#inner : 교집합, 두 DataFrame에 모두 키 값이 존재하는 경우에만 병합한다
#outer : 합집합, 하나의 DataFrame에 키 값이 존재하는 경우 모두 병합
pd.merge(df, df_right, on='이름', how='inner')
pd.merge(df, df_right, on='이름', how='outer')

#기준이 되는 column의 index 명이 다른경우
df_right.columns = ['성함','연봉','가족수']
pd.merge(df,df_right, left_on='이름', right_on='성함', how='outer')









