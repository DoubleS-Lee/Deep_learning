# Dynamic Programming
# 정책반복 : 정책 평가와 정책 개선을 반복하는 시스템 (여기서는 정책 평가와 정책 개선 알고리즘을 보고 다음 파이썬 파일에 실제 적용해봄)
# 전체 flow
## 1. 환경(env) Gridworld를 설정
## 2. DP 계산을 위해서 agent를 설정해준다 (DP 계산을 위해서 이미 만들어놓은 클래스로 객체 생성)
## 3. 1) 정책 평가 계산(policy evaluation)
###    1-1) T{pi}(V) <- R{pi} + (gamma)*P{pi}*V 계산이 목표임 이를 위해 R{pi}, P{pi}를 계산한다. 
###         V의 경우 처음에 모를 경우 zero 행렬로 초기화한다
###    1-2) R{pi} 계산 =  P * R을 해준다 (get_r_pi())
###    1-3) P{pi} 계산 =  pi * P를 해준다(아인슈타인 계산을 해야한다 2차원행렬*3차원텐서 = 2차원행렬) (get_p_pi())
###    1-4) 구해 놓은거로 T{pi}(V))를 업데이트 한다 (에러가 정해준 값보다 작거나 같을때까지 반복)
###    1-5) 이때의 output이 V{pi} 이다
## 3. 2) 정책 개선 계산(policy improvement) (greedy 방법으로 진행)
###    2-1) Q{pi}(s,a) = R{a,s} + (gamma)*P{a,ss'}*V{pi}(s') 를 계산하고 
###    2-2) 정책 개선(pi') : 가장 높은 Q{pi}(s,a)값을 주는 a를 1로 설정하는 것이 목표
###    2-3) R{a,s} 계산 : P * R을 해준다 (get_r_pi())
###    2-4) Q{pi}(s,a) 계산: 3.1)에서 계산한 V{pi}, gamma, 상태천이함수 P를 이용해서 계산
###    2-5) 2-2)를 수행하여 가장 높은 Q{pi}(s,a)값을 주는 a를 1로 설정
## 4. 개선된 최종 정책을 확인하고, 이에 따르는 가치함수도 확인해본다


import sys; sys.path.append('C:/Users/DoubleS/Documents/Deep_learning/deep_learning_practice/강화학습') # add project root to the python path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.part2.tensorized_dp import TensorDP
from src.common.gridworld import GridworldEnv
from src.common.grid_visualization import visualize_value_function, visualize_policy

np.random.seed(0)

## `GridWorld` 초기화하기
'''
가로로 `nx` 개, 세로로 `ny` 개의 칸을 가진 `GridworldEnv`를 만듭니다!
'''
nx = 5
ny = 5
env = GridworldEnv([ny, nx])

## 동적계획법 '에이전트' 초기화하기
'''
동적 게획법은 원래 `에이전트`라는 개념을 사용하지 않지만, 
일반적으로 사용되는 강화학습의 구현 템플릿에 대한 이해를 돕기 위해 `에이전트`라는 개념을 적용하였습니다.

`TensorDP` 클래스는 2개의 파라미터를 받습니다. 
감소율 `gamma` 와 가치평가/반복 알고리즘에서 수렴조건을 확인할 때 사용하는 수치적 에러의 허용치인 `error_tol` 입니다. 
이 예제에서는 각각 `1.0` 과 `1e-5`로 설정하겠습니다.
'''
dp_agent = TensorDP()

### DP agent에게 환경을 설정봅시다!

dp_agent.set_env(env)

## `Numpy` 에서 텐서 연산
'''
## Tensorized 정책 평가 (Policy evaluation)

Tensorized syncrhonous Policy evaluation 에 대해서 알아봅시다.

정책 평가 알고리즘은 Bellman expectation backup operator $T$ 가 수렴할때까지 반복하여, 
현재 정책함수 $\pi$ 에 대한 주어진 MDP의 가치함수인 $V^{\pi}$ 를 찾는 알고리즘입니다.
Bellamn expectation backup operator $T$ 는 다음과 같이 정의됩니다.
T{pi}(V) <- R{pi} + (gamma)*P{pi}*V
$\gamma$ 는 감가율, $R^{\pi}$ 는 정책 $\pi$ 에 대한 보상함수, $P^{\pi}$ 는 정책 $\pi$ 에 대한 상태천이 행렬입니다. <br>

### $R^{\pi}$ 효율적으로 계산하기
MDP 강의에서 이야기했던대로, $R^{\pi}$ 는 다음과 같이 정의 됩니다.

$$R^{\pi}_s = \sum_{a \in \cal{A}} \pi(a|s) R_s^a $$

각 $R^{\pi}$의 각 원소 $R^{\pi}_s$ 위의 수식으로 정의되고 $R^{\pi}$ 은 모든 상태 $s$ 의 $R^{\pi}_s$를 열 벡터로 표현한 형태가 됩니다. $R^{\pi} \in \mathbb{R}^{|\cal{S}|}$. 

> 예제에서는 구현의 편의를 위해 열 벡터 (Rank1 텐서)에 하나의 축을 더하여 $R^{\pi} \in \mathbb{R}^{|\cal{S}|\times 1}$ 으로 표현하였습니다.

#### Numpy 를 활용해 위의 수식을 구현
'''

# R{pi}와 P{pi}를 계산해보자
# 밑에 있는건 뒤에 괄호가 없으니까 함수가 아니고 클래스로 만든 객체 안에 있는 변수들을 불러온거임
# (state 수, action 수)행렬(25,4)
policy = dp_agent.policy # [num. states x num. actions]
R = dp_agent.R # [num. states x num. actions]

print("Policy")
print(policy)

print("Reward function")
print(R)

### * 오퍼레이터
'''
`*` 오퍼레이터는 두개의 `Numpy` array 를 각 원소별로 곱합니다.
'''
# R{pi} 효율적으로 계산하기
# weighted_R, averaged_R은 나중에 쓰지도 않는데 왜 한거지?
# R{pi}를 효율적으로 계산하기 위한 방법 <- 정책*R을 각 행 별로 sum하는것 (여기서는 나중에 def get_r_pi()에서 계산하게 된다)
weigthed_R = policy * R # [num. states x num. actions]
print(weigthed_R)

averaged_R = weigthed_R.sum(axis=-1)
print(averaged_R)

### P{pi} 효율적으로 계산하기
### 랜덤정책 $\pi$ 에 대한 $P^{\pi}$ 확인해보기
'''
모든 상황에서 각 방향으로 움직일 확률이 0.25 인 정책 pi 의 P{pi}는?
'''
# 현재 P{pi} : 현재 상태에서 임의의 방향(상,하,좌,우)으로 움직일 확률
df = pd.DataFrame(dp_agent.get_p_pi(dp_agent.policy))
df

## 드디어 정책 평가 알고리즘!
## 랜덤 정책함수로 `dp_agent`의 정책 초기화하기
'''
랜덤 정책함수로 `dp_agent` 의 정책함수를 초기화 하였습니다. 한번 확인해볼까요?
'''
policy_state_dim = dp_agent.policy.shape[0]
policy_action_dim = dp_agent.policy.shape[1]
print("===== 정책함수 스펙 =====")
print("state dimension: {}".format(policy_state_dim))
print("action dimension: {} \n".format(policy_action_dim))

print("===== 정책함수 =====")
print(dp_agent.policy)

## 랜덤 정책함수 평가하기 (가치 함수를 찾는 과정)
'''
앞서 정의한 `policy_evaluation()` 를 활용해 현재 정책인 랜덤 정책에 대한 가치 함수를 추산합니다.
perform bellman expectation backup operator인 T{pi}(V)가 수렴할때까지 policy_evaluation()가 계산하고 T{pi}(V)를 뱉는다
이 v_new는 T{pi}(V)이며 이게 수렴할때까지 계산을 반복함
'''
v_pi = dp_agent.policy_evaluation()
fig, ax = plt.subplots(1,2, figsize=(12,6))
visualize_value_function(ax[0], v_pi, nx, ny)
_ = ax[0].set_title("Value pi")
visualize_policy(ax[1], dp_agent.policy, nx, ny)
_ = ax[1].set_title("Policy")

v_old = v_pi # 정책 개선 정리에 대해 설명할때 사용

#정책 개선 (정책을 찾는 과정)

## "내 정책은 조금 전의 정책보다 개선된다!" 
'''
`policy_improvement()` 를 활용해 greedy policy improvement 를 수행합니다.

> Greedy 정책개선
> 1. $V^{\pi}(s)$ 와 $P$, $R$ 를 이용해 $Q^{\pi}(s,a)$ 를 계산한다. <br>
$$Q^\pi(s,a) = R_s^{a} + \gamma \Sigma_{s' \in \cal{S}}P_{ss'}^aV^{\pi}(s')$$

> 2. 개선된 정책 $\pi'(a|s)$ 을 가장 높은 $Q^{\pi}(s,a)$ 값을 주는 $a$ 에 대해서 1로 설정.
나머지는 0.0

파이썬 구현체를 한번 살펴보죠
```python
    def policy_improvement(self, policy=None, v_pi=None):
        if policy is None:
            policy = self.policy

        if v_pi is None:
            v_pi = self.policy_evaluation(policy)

        # (1) Compute Q_pi(s,a) from V_pi(s)
        r_pi = self.get_r_pi(policy)
        q_pi = r_pi + self.P.dot(v_pi)

        # (2) Greedy improvement
        policy_improved = np.zeros_like(policy)
        policy_improved[np.arange(q_pi.shape[1]), q_pi.argmax(axis=0)] = 1
        return policy_improved
```

> `policy_improved = np.zeros_like(policy)`  <br>
개선된 정책 $\pi'$ 는 선택될 action (특정 $s$ 에 대해 가장 큰 $Q(s,a)$를 만족하는 $a$) 이외에는 값이 0 이기 때문에 0으로 초기화 합니다.

> `policy_improved[np.arange(q_pi.shape[1]), q_pi.argmax(axis=0)] = 1` <br>
특정 $s$에 대해 가장 큰 $Q(s,a)$를 만족하는 $a$ 만을 1.0 으로 설정합니다.
'''
### Numpy 의 `argmax` 와 `advance indexing`
'''
정책 개선을 구현하기 위해 조금 헷갈릴수 있는 트릭을 활용하였습니다. 본격적인 설명으로 넘어가기 전에
어떤 일이 벌어졌는지 예시를 들어 확인해보겠습니다. 
> 상태의 종류가 2개이고 가능한 행동이 3개인 간단한 MDP를 생각해봅시다.
'''
Q = np.array([[1,2,3], [3,2,1]]) # [2 x 3] Q values.
print(Q)
'''
우리가 원하는것은 각 상태 $s$ (row) 에 대해 최댓값을 주는 행동 $a$의 index 입니다.
'''
# axis 이해가 안되면 여기가서 보고와라 http://machinelearningkorea.com/2019/05/18/%ED%8C%8C%EC%9D%B4%EC%8D%AC-axis-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0-%EC%B9%98%ED%8A%B8%EC%BD%94%EB%93%9C/
a_max = Q.argmax(axis=-1) # 마지막 축 (axis) 에 대해 argmax 를 구하면?
a_max # 파이썬은 숫자를 0부터 셉니다!

policy_improved = np.zeros(shape=(2, 3))
print("Initialized policy")
print(policy_improved)

policy_improved[0, a_max[0]] = 1
print("set policy for the first state")
print(policy_improved)

policy_improved[1, a_max[1]] = 1
print("set policy for the second state")
print(policy_improved)
'''
하지만 `numpy`의 indexing 을 활용하면 위의 코드를 한줄로 표현할수 있습니다.
'''
policy_improved = np.zeros(shape=(2, 3))
print("Initialized policy")
print(policy_improved)

policy_improved[(0,1),(a_max[0],a_max[1])] = 1
print("Policy improvement")
print(policy_improved)
'''
`policy_improved[np.arange(q_pi.shape[1]), q_pi.argmax(axis=0)] = 1`를 실습 해보세요.
'''

## Mini HW
p_new = dp_agent.policy_improvement()
dp_agent.set_policy(p_new) # DP agent 의 정책을 개선된 정책 `p_new`로 설정

### 개선된 정책 확인하기
policy_state_dim = dp_agent.policy.shape[0]
policy_action_dim = dp_agent.policy.shape[1]
print("===== 정책함수 스펙 =====")
print("state dimension: {}".format(policy_state_dim))
print("action dimension: {} \n".format(policy_action_dim))

# greedy improvement를 했으니까 특정 행동 1개는 1의 값을 갖고 나머지는 0을 갖는다
print("===== 정책함수 =====")
print(dp_agent.policy)

v_pi = dp_agent.policy_evaluation()
fig, ax = plt.subplots(1,2, figsize=(12,6))
visualize_value_function(ax[0], v_pi, nx, ny)
_ = ax[0].set_title("Value pi")
visualize_policy(ax[1], dp_agent.policy, nx, ny)
_ = ax[1].set_title("Policy")

v_new = v_pi # 개선된 정책에 대한 가치함수

## 정책개선 정리 결과 확인해보기
'''
__정책개선 정리__ : 정책 개선 정리를 활용해 구해진 $\pi'$과 개선전 정책 $\pi$는 다음의 관계를 만족한다.
pi(개선) >= pi(개선전)
$$\pi' \geq \pi \leftrightarrow V_{\pi'}(s) \geq V_{\pi}(s) \forall s \in S$$
'''
delta_v = v_new - v_old

delta_v
