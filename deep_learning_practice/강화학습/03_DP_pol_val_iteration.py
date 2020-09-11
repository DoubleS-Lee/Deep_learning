# Dynamic Programming
# DP를 푸는 2가지 방법 : 정책 반복, 가치 반복
# 전체 flow
## 방법 1) 정책 반복
# 정책반복 : (정책 평가 + 정책 개선)을 정책이 수렴할때까지 반복!
# 자세한 내부 알고리즘은 02_DP_policy_eval_impro.py 를 참고
## 방법 2) 가치 반복
# 정책 평가와 정책 개선을 하나로 합친 알고리즘
# 정책을 쓰지 않고 상태가치함수를 가지고 업데이트를 한다
# V{k+1} = max{모든a}(R{a} + (gamma)*P{a}*V{k+1})


import sys; sys.path.append('C:/Users/DoubleS/Documents/Deep_learning/deep_learning_practice/강화학습') # add project root to the python path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.part2.tensorized_dp import TensorDP
from src.common.gridworld import GridworldEnv
from src.common.grid_visualization import visualize_value_function, visualize_policy

np.random.seed(0)

## `GridWorld` 초기화하기
# 가로로 `nx` 개, 세로로 `ny` 개의 칸을 가진 `GridworldEnv`를 만듭니다!
nx = 5
ny = 5
env = GridworldEnv([ny, nx])

## 동적계획법 '에이전트' 초기화하기
#동적 게획법은 원래 `에이전트`라는 개념을 사용하지 않지만, 일반적으로 사용되는 강화학습의 구현 템플릿에 대한 이해를 돕기 위해 `에이전트`라는 개념을 적용하였습니다.

dp_agent = TensorDP()
dp_agent.set_env(env)

## 방법 1) 정책 반복
#정책반복 : (정책 평가 + 정책 개선)을 정책이 수렴할때까지 반복!
dp_agent.reset_policy()
# policy_iteration() 안에 policy_evaluation(), policy_improvement() 계산이 들어있음 
info_pi = dp_agent.policy_iteration()

figsize_mul = 10
steps = info_pi['converge']
fig, ax = plt.subplots(nrows=steps, ncols=2, figsize=(steps * figsize_mul, figsize_mul * 2))
for i in range(steps):
    visualize_value_function(ax[i][0],info_pi['v'][i], nx, ny)
    visualize_policy(ax[i][1], info_pi['pi'][i], nx, ny)    

## 방법 2) 가치 반복
# 정책 평가와 정책 개선을 하나로 합친 알고리즘
# 정책을 쓰지 않고 상태가치함수를 가지고 업데이트를 한다
# V{k+1} = max{모든a}(R{a} + (gamma)*P{a}*V{k+1})
dp_agent.reset_policy()
info_vi = dp_agent.value_iteration(compute_pi=True)

figsize_mul = 10
steps = info_vi['converge']

fig, ax = plt.subplots(nrows=steps,ncols=2, figsize=(steps * figsize_mul * 0.5, figsize_mul* 3))
for i in range(steps):
    visualize_value_function(ax[i][0],info_vi['v'][i], nx, ny)
    visualize_policy(ax[i][1], info_vi['pi'][i], nx, ny)    