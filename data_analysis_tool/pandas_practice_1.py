import pandas as pd

###############################################################
#Series
#DataFrame

pd.Series([1,2,3,4])

a=[1,2,3,4]

pd.Series(a)

mystyle = [1,2,3,4,5]

pd.Series(mystyle)

company1 = [["삼성",1000,"휴대폰"],
           ["현대",500,"자동차"],
           ["네이버",700,"사이트"]]
df1 = pd.DataFrame(company1)
df1.columns = ["기업명","매출액","업종"]



company2 = {"기업명":['삼성','현대','네이버'],
            "매출액":[1000,500,700],
            "업종":["휴대폰","자동차","포털"]}
df2 = pd.DataFrame(company2)

df1.index = df1['기업명']
df1.index = df1['업종']

df1['매출액']
type(df1['매출액'])


###############################################################
#csv 파일 읽어오기

df = pd.read_csv("korean-idol.csv")


###############################################################
#excel 파일 읽어오기
pd.read_excel("korean-idol.xlsx")

###############################################################
# 열 출력
df.columns

new_col = ['name', '그룹', '소속사', '성별', '생년월일', '키', '혈액형', '브랜드평판지수']

df.columns = new_col
df.columns

###############################################################
# 행 출력
df.index

###############################################################
df.info()
df.describe()
df.shape

df.head(3)
df.tail(3)

df.sort_index()
df.sort_index(ascending=False)

df.sort_values(by='키')
df.sort_values(by='키', ascending=False)

df.sort_values(by=['키','혈액형'])

###############################################################
# 특정 column 데이터만 선택 방법
df['그룹']
df["그룹"]
df.그룹
type(df['그룹'])

###############################################################
# 범위선택
######################## 중요
df[4:7]

#loc : 인덱스로 범위 선택, 조건을 걸고 그 다음에 추출을 원하는 column을 설정
#loc[행,열]   : 파이썬 리스트나 numpy와 다르게 [:]에서 이상과 이하의 값을 가져온다
df.loc[:,'name']
df.loc[4:7,'name']
df.loc[4:7,['name','생년월일']]
df.loc[4:7,'name':'생년월일']

#iloc : 번호로 범위 선택
#iloc[행,열]   : loc와 다르게 [:]에서 numpy와 리스트의 규칙을 따른다
df.iloc[:,[1,3,5]]
df.iloc[:,1:5]
df.iloc[1:5,1:5]

###############################################################
# boolean indexing : 조건에 맞는 값을 True로 반환한다. 이를 이용해서 조건에 맞는 값을 가져올 수 있
df['키'] > 180
df[df['키'] > 180]

df[df['키'] > 180]['name']
df[df['키'] > 180][['name','키']]

#############위의 방법 쓰지말고 loc를 활용하자##############
df.loc[df['키']> 180, 'name']
df.loc[df['키']> 180, ['name','키']]
df.loc[df['키']>180, 'name':'키']

###############################################################
# isin으로 index
# 리스트내의 값을 내가 설정한 조건으로 불러온다
my_condition = ['플레디스','SM']
df['소속사'].isin(my_condition)
df.loc[df['소속사'].isin(my_condition)]
df.loc[df['소속사'].isin(my_condition), ['name','소속사']]
df.loc[df['소속사'].isin(my_condition), 'name':'소속사']

################################################################
# 결측값(Null)
# pandas에서는 NaN 으로 표현된다
# info()로 어느 colomn에 NaN 데이터가 몇개 있는지 알아낼수있
df.info()

#isna(). isnull() 은 NaN을 찾는것
#notnull()은 NaN이 아닌 것을 찾는

# .isna() : 전체 데이터에 NaN이 있는 곳을 찾아라
df.isna()
df['그룹'].isna()

df.isnull()
df['그룹'].isnull()

df.loc[df['그룹'].isnull()]
df.loc[df['그룹'].isnull(), ['그룹']]
df.loc[df['그룹'].isnull(), '그룹':'키']

# NaN이 아닌 값에 대하여 Boolean 인덱싱
df['그룹'].notnull()

# NaN이 아닌 값만 색출해내기
df.loc[df['그룹'].notnull(), ['그룹']]

################################################################
# Copy  :원본 데이터를 유지시키고, 새로운 변수에 복사하기 위함
# copy()

df.head()
# 이런식으로 데이터를 복사하면 같은 메모리 주소를 복사하기 때문에 원래 본체인 df의 name 값도 다 0으로 바뀐다
new_df = df
new_df['name']=0
new_df
df

# 따라서 copy를 써줘서 새로운 변수를 만들어줘야한다
df = pd.read_csv("korean-idol.csv")
new_col = ['name', '그룹', '소속사', '성별', '생년월일', '키', '혈액형', '브랜드평판지수']
df.columns = new_col

copy_df = df.copy()
copy_df['name']=0
copy_df
df

################################################################
# row의 추가 : 반드시 ignore_index=True 옵션을 같이 추가해줘야한다
df.head()
df = df.append({'name':'테디', '그룹':'테디그룹', '소속사':'끝내주는 소속사', '성별':'남자', '생년월일':'1988-05-20', '키':180.5, '혈액형':'A', '브랜드평판지수':5468765}, ignore_index=True)

df.tail()

################################################################
# column 추가 : 단순히 df['원하는 column명'] 을 적어서 생성해주면 된다
df.head()
df['국적'] = '대한민국'
df.head()

df.loc[df['name'] == '지드래곤', ['국적']] = 'korea'
df.head()


################################################################
# 통계값 다루기
df.info()
df.describe()
df['키'].max()
df['키'].min()
df['키'].sum()
df['키'].mean()

#분산과 표준편차

import numpy as np

data_01 = np.array([1,3,5,7,9])
data_02 = np.array([3,4,5,6,7])

data_01.mean()
data_02.mean()

data_01.var(), data_02.var()
data_01.std(), data_02.std()

df['키'].var()
df['키'].std()
# 중앙값
df['키'].median()
# 최빈값 : 제일 많이 출현하는 값을 출력한다
df['키'].mode()



################################################################
# 피벗테이블
# 데이터 열 중에서 두 개의 열을 각각 행 인덱스, 열 인덱스로 사용하여 데이터를 조회하여 펼쳐놓은 것을 의미함
# 내가 최종적으로 관심있고 활용하고 싶은 값은 values에 넣는다
# 데이터가 겹치는 경우 해당 데이터들의 평균값이 나오게 된다 aggfunc을 쓰는 경우 합계, 평균등 원하는 값으로 나타나게 할수 있다
df.head()
df
pd.pivot_table(df, index='소속사', columns='혈액형', values='키')

pd.pivot_table(df, index='그룹', columns='혈액형', values='브랜드평판지수', aggfunc=np.sum)
pd.pivot_table(df, index='그룹', columns='혈액형', values='브랜드평판지수', aggfunc=np.mean)


################################################################
# GroupBy
# 데이터를 그룹으로 묶어서 분석할때 활용
df.head()

df.groupby('소속사').count()
df.groupby('그룹').mean()
df.groupby('성별').sum()

df.groupby('혈액형').mean()
df.groupby('혈액형')['키'].mean()

################################################################
# Multi-Index
# 행 인덱스를 복합적으로 구성하고 싶은 경우에 사용
df2 = df.groupby(['혈액형','성별']).mean()
df2.unstack('혈액형')
df2.unstack('성별')
df2.reset_index()























