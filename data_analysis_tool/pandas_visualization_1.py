import pandas as pd

#################################################################
#!apt -qq -y install fonts-nanum > /dev/null

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

fontpath = 'c:/Windows/Fonts/맑은 고딕.ttf'
font = fm.FontProperties(fname=fontpath, size=10)
fm._rebuild()

#%config InlineBackend.figure_format = 'retina'

plt.rc('font', family = 'NanumBarunGothic')


########################################################################
df = pd.read_csv('house_price_clean.csv')


##############################################################################
# plot의 kind 옵션
# line, bar, bath, hist, hexbin, box, area, pie, scatter

# line
df['분양가'].plot(kind='line')
df['분양가'].plot(kind='line')

df
df_seoul = df.loc[df['지역'] == '서울']
df_seoul

df_seoul_year = df_seoul.groupby('연도').mean()
df_seoul_year

df_seoul_year['분양가'].plot(kind='line')

# bar
df.groupby('지역')['분양가'].mean()
df.groupby('지역')['분양가'].mean().plot(kind='bar')
df.groupby('지역')['분양가'].mean().plot(kind='barh')

# hist
df['분양가'].plot(kind='hist')

# 커널밀도 그래프
df['분양가'].plot(kind='kde')

# Hexbin
xx = df.loc[(df['분양가'] > 1000) & (df['분양가'] <2000)]
df.plot(kind='hexbin', x='분양가', y='연도', gridsize=20)

# 박스 플롯
df_seoul = df.loc[df['지역'] == '서울']
df_seoul['분양가'].plot(kind='box')

# area plot
df.groupby('월')['분양가'].count().plot(kind='line')
df.groupby('월')['분양가'].count().plot(kind='area')

# pie plot
df.groupby('월')['분양가'].count().plot(kind='pie')

# scatter plot
df.plot(x='월', y='분양가', kind='scatter')















