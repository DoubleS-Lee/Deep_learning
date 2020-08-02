import pandas as pd

df = pd.read_csv('seoul_house_price.csv')

df.head()

df = df.rename(columns={'분양가격(㎡)':'분양가격'})

df

# 빈값과 DataType 확인
df.info()


# 통계값확인
df.describe()

######### 분양가격 column을 int로 변환
# 분양가격 안에 NaN 값이 있는지 확인
df['분양가격'].isnull()

# 공백 '  ' 없애기
df['분양가격'] = df['분양가격'].str.strip()

# 공백 ''에 0을 집어넣기
df.loc[df['분양가격'] == '', '분양가격'] = 0

# NaN값에 다 0으로 값을 집어넣기
df['분양가격'] = df['분양가격'].fillna(0)

df.info()

# 아직도 에러가 뜬다
df['분양가격'] = df['분양가격'].astype(int)

# 에러가 어디서 발생했는지 찾는다
df.loc[df['분양가격'] == '6,657']

# ,를 없애준다 에러를 수정한다
df['분양가격'] = df['분양가격'].str.replace(',', '')

df['분양가격'] = df['분양가격'].astype(int)

# NaN제거
df['분양가격'] = df['분양가격'].fillna(0)

df['분양가격'] = df['분양가격'].astype(int)

# -제거
df['분양가격'] = df['분양가격'].str.replace('-','')

df['분양가격'] = df['분양가격'].astype(int)


df['분양가격'] = df['분양가격'].fillna(0)

df['분양가격'] = df['분양가격'].astype(int)

df.loc[df['분양가격']=='']

df.loc[df['분양가격']=='','분양가격'] = 0

df['분양가격'] = df['분양가격'].astype(int)

df.info()

df.groupby('지역명')['분양가격'].mean()




























