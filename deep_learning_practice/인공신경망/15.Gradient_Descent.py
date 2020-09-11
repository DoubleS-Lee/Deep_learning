# Gradient Descent 최적화 구현

import numpy as np

## 손실 함수 정의 (Analytic)

def f(x):
    return 0.1*x**4 - 1.5*x**3 + 0.6*x**2 + 1.0*x + 20.0

## 손실 함수 미분 정의

def df_dx(x):
    return 0.4*x**3 - 4.5*x**2 + 1.2*x + 1.0

## 하이퍼파라미터 정의

x = 5
eps = 1e-5
lr = 0.01
max_epoch = 1000

## Gradient Descent 알고리즘 구현

min_x = x
min_y = f(min_x)
for _ in range(max_epoch):
    grad = df_dx(x)
    new_x = x - lr * grad
    y = f(new_x)
    
    if min_y > y:
        min_x = new_x
        min_y = y
    
    if np.abs(x - new_x) < eps:
        break
    
    x = new_x

print(min_x, min_y)