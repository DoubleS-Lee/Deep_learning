import pandas as pd

df=pd.read_csv('korean-idol.csv')
df

df.info()

df.isnull()

df.dropna(axis=0)

df['그룹'] = df['그룹'].fillna('없음')

df

df['키'].mean()

df['키'] = df['키'].fillna(df['키'].mean())
