import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#%%
#한글 폰트 깨짐 현상 해결방법
# matplotlib 폰트설정
plt.rc('font', family='NanumGothic') # For Windows
print(plt.rcParams['font.family'])
#%matplotlib inline
# 브라우저에서 바로 이미지를 그린다.

df = pd.read_csv('house_price_clean.csv')

df.plot()
#%%
# 단일 그래프
data = np.arange(1,100)
data

plt.plot(data)

# 다중 그래프
#밑에 3줄을 한꺼번에 실행시켜야함
data = np.arange(1,51)
data2 = np.arange(51,101)
plt.plot(data)
plt.plot(data2)
plt.plot(data2 + 50)

#%%
#figure()는 새로운 그래프를 생성한다
data = np.arange(100,201)
data2 = np.arange(200,301)
plt.plot(data)
plt.figure()
plt.plot(data2)

#%%
# 그래프를 행렬화하여 출력
#subplot
#subplot(row,column,index)
data = np.arange(100,201)
plt.subplot(2,1,1)
plt.plot(data)
data2 = np.arange(200,301)
plt.subplot(2,1,2)
plt.plot(data2)

#subplot(row,column,index)
data = np.arange(100,201)
plt.subplot(1,2,1)
plt.plot(data)
data2 = np.arange(200,301)
plt.subplot(1,2,2)
plt.plot(data2)

#%%
# 여러개의 plot을 그리는 방법
# subplots
#plt.subplots(행의 갯수, 열의 갯수)
data = np.arange(1,51)
fig, axes = plt.subplots(2,3)

axes[0, 0].plot(data)
axes[0, 1].plot(data * data)
axes[0, 2].plot(data ** 3)
axes[1, 0].plot(data % 10)
axes[1, 1].plot(-data)
axes[1, 2].plot(data // 20)

plt.tight_layout()

#%%
# 주요 스타일 옵션
from IPython.display import Image
Image('https://matplotlib.org/_images/anatomy.png')

#%%
# 타이틀
plt.plot([1,2,3],[3,6,9])
plt.plot([1,2,3],[2,4,9])

#타이틀 & font 설정
plt.title('Label 설정 예제입니다', fontsize = 20)

#x축 & y축 Label 설정
plt.xlabel('x축', fontsize = 20)
plt.ylabel('y축', fontsize = 20)

# x tick, y tick 설정
plt.xticks(rotation=-30)
plt.yticks(rotation=30)

# 범례(Legend) 설정
plt.legend(['10 * 2', '10 ** 2'], fontsize=15)

# x와 y의 한계점(limit) 설정
# xlim(), ylim()
plt.xlim(0,5)
plt.ylim(0.5, 10)

#%%
# 스타일 세부설정 - 마커, 라인, 컬러
# https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot

# marker의 종류

# '.' point marker
# ',' pixel marker
# 'o' circle marker
# 'v' triangle_down marker
# '^' triangle_up marker
# '<' triangle_left marker
# '>' triangle_right marker
# '1' tri_down marker
# '2' tri_up marker
# '3' tri_left marker
# '4' tri_right marker
# 's ' square marker
# 'p' pentagon marker
# '*' star marker
# 'h' hexagon1 marker
# 'H' hexagon2 marker
# '+' plus marker
# 'x' x marker
# 'D' diamond marker
# 'd' thin_diamond marker
# '|' vline marker
# '_' hline marker

plt.plot(np.arange(10), np.arange(10)*2, marker='o', markersize=5)
plt.plot(np.arange(10), np.arange(10)*2 - 10, marker='v', markersize=10)
plt.plot(np.arange(10), np.arange(10)*2 - 20, marker='+', markersize=15)
plt.plot(np.arange(10), np.arange(10)*2 - 30, marker='*', markersize=20)

# 타이틀 & font 설정
plt.title('마커 설정 예제', fontsize=20)

# X축 & Y축 Label 설정
plt.xlabel('X축', fontsize=20)
plt.ylabel('Y축', fontsize=20)

# X tick, Y tick 설정
plt.xticks(rotation=90)
plt.yticks(rotation=30)

#%%
# line의 종류

# '-' solid line style
# '--' dashed line style
# '-.' dash-dot line style
# ':' dotted line style

plt.plot(np.arange(10), np.arange(10)*2, marker='o', linestyle='')
plt.plot(np.arange(10), np.arange(10)*2 - 10, marker='o', linestyle='-')
plt.plot(np.arange(10), np.arange(10)*2 - 20, marker='v', linestyle='--')
plt.plot(np.arange(10), np.arange(10)*2 - 30, marker='+', linestyle='-.')
plt.plot(np.arange(10), np.arange(10)*2 - 40, marker='*', linestyle=':')

# 타이틀 & font 설정
plt.title('다양한 선의 종류 예제', fontsize=20)

# X축 & Y축 Label 설정
plt.xlabel('X축', fontsize=20)
plt.ylabel('Y축', fontsize=20)

# X tick, Y tick 설정
plt.xticks(rotation=90)
plt.yticks(rotation=30)

#%%
# color의 종류

# 'b' blue
# 'g' green
# 'r' red
# 'c' cyan
# 'm' magenta
# 'y' yellow
# 'k' black
# 'w' white

plt.plot(np.arange(10), np.arange(10)*2, marker='o', linestyle='-', color='b')
plt.plot(np.arange(10), np.arange(10)*2 - 10, marker='v', linestyle='--', color='c')
plt.plot(np.arange(10), np.arange(10)*2 - 20, marker='+', linestyle='-.', color='y')
plt.plot(np.arange(10), np.arange(10)*2 - 30, marker='*', linestyle=':', color='r')

# 타이틀 & font 설정
plt.title('색상 설정 예제', fontsize=20)

# X축 & Y축 Label 설정
plt.xlabel('X축', fontsize=20)
plt.ylabel('Y축', fontsize=20)

# X tick, Y tick 설정
plt.xticks(rotation=90)
plt.yticks(rotation=30)

#######################################
#%%
plt.plot(np.arange(10), np.arange(10)*2, color='b', alpha=0.1)
plt.plot(np.arange(10), np.arange(10)*2 - 10, color='b', alpha=0.3)
plt.plot(np.arange(10), np.arange(10)*2 - 20, color='b', alpha=0.6)
plt.plot(np.arange(10), np.arange(10)*2 - 30, color='b', alpha=1.0)

# 타이틀 & font 설정
plt.title('투명도 (alpha) 설정 예제', fontsize=20)

# X축 & Y축 Label 설정
plt.xlabel('X축', fontsize=20)
plt.ylabel('Y축', fontsize=20)

# X tick, Y tick 설정
plt.xticks(rotation=90)
plt.yticks(rotation=30)

# 그리드 설정
plt.plot(np.arange(10), np.arange(10)*2, marker='o', linestyle='-', color='b')
plt.plot(np.arange(10), np.arange(10)*2 - 10, marker='v', linestyle='--', color='c')
plt.plot(np.arange(10), np.arange(10)*2 - 20, marker='+', linestyle='-.', color='y')
plt.plot(np.arange(10), np.arange(10)*2 - 30, marker='*', linestyle=':', color='r')

# 타이틀 & font 설정
plt.title('그리드 설정 예제', fontsize=20)

# X축 & Y축 Label 설정
plt.xlabel('X축', fontsize=20)
plt.ylabel('Y축', fontsize=20)

# X tick, Y tick 설정
plt.xticks(rotation=90)
plt.yticks(rotation=30)

# grid 옵션 추가
plt.grid()


# 이미지 저장
plt.savefig('my_graph.png', dpi=300)






