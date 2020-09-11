import sys; sys.path.append('C:/Users/DoubleS/Documents/Deep_learning/deep_learning_practice/강화학습') # add project root to the python path
import numpy as np
# 마르코프 의사결정 과정 MDP
from src.common.gridworld import GridworldEnv # Gridworld Environment

num_y, num_x = 4, 4
env = GridworldEnv(shape=[num_y, num_x])

# 전체 state
observation_space = env.observation_space
# 전체 action
action_space = env.action_space
# state의 수
print("Number of states: {}".format(observation_space))
# action의 수
print("Number of actions: {}".format(action_space))

# 상태천이 행렬 (3차원 텐서임)
## 상태천이 매트릭스 (텐서) $P$ (P 3차원)
# UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3
### 코드에서 상태천이 텐서  $P$ 불러오기 
# P의 shape = (행동의 수nA, 현재상태nS, 다음상태nS)
P = env.P_tensor
print("P shape : {}".format(P.shape))

### `Numpy` 배열의 장점, 직관적인(?) 편리한 인덱싱
# 행동 '상'(=인덱스0)을 선택했을때 현재상태(row)에서 다음상태(colunm)로 전이하는 확률을 나타냄.
# P의 shape = (행동의 수nA, 현재상태nS, 다음상태nS)
action_up_prob = P[0, :, :]
print(action_up_prob)

# 1) 모든 요소가 0보다 크거나 같은지 확인
is_greater_than_0 = action_up_prob >= 0
is_all_greater_than_0 = is_greater_than_0.sum() == is_greater_than_0.size
is_all_greater_than_0

### (2) 가로합은 1.0 일까?
action_up_prob.sum(axis=1) # 2 번째 축으로 합을 계산합니다. 파이썬은 숫자를 0부터 셉니다.

### 소소한 코딩팁. `Numpy` 의 auto inferencing 기능

action_up_prob.sum(axis=-1) # '마지막' 축으로 합을 계산합니다.

## 보상함수 $R$ 
# 2차원 텐서이며 세로 축은 상태, 가로축은 행동을 의미
# 종결점에 도달하기 전까지는 매 이동마다 -1을 받음
R = env.R_tensor
print(R)

## MDP 의 Episode
'''
MDP의 Episode는 $<(s_0, a_0, r_0, s_1), (s_1, a_1, r_1, s_2), ..., (s_{t}, a_{t}, r_{t}, s_{t+1}),..., (s_{T-1}, a_{T-1}, r_{T-1}, s_{T})>$ 으로 구성됩니다. 
각 튜플은 매 시점 t의 상태 $s_t$, 선택한 행동 $a_t$, 보상 $r_t$, 그 다음 상태 $s_{t+1}$ 를 포함합니다. $T$는 종결 시점을 나타냅니다. 
하지만 일반적으로 RL 알고리즘을 구현할때는 구현의 편의를 위해 특정 시점이 종결 상태인지 아닌지를 확인하는 `done` 이라는 정보를 
episode의 각 튜플에 추가해서 관리합니다.
`GridworldEnv`에서 에피소드를 한번 시뮬레이션 해보도록 하죠.
### MDP를 시뮬레이션하기 위해서는 정책 $\pi$ 가 필요합니다!
일단은 각 상태에서 모든 행동을 0.25의 확률로 고르는 정책함수로 `Gridworld`를 시뮬레이션 해봅시다.
'''
# 값이 필요없는 경우라서 _로 이름을 만들어 줌
# Gridworld 를 초기화합니다.
# 현재 상태 위치가 초기화 됨
_ = env.reset() 
print("Current position index : {}".format(env.s))

# 좀 더 직관적인 가시화를 위해서 action 인덱스를 방향으로 바꿔줍니다.
action_mapper = {
    0: 'UP',
    1: 'RIGHT',
    2: 'DOWN',
    3: 'LEFT'
}

step_counter = 0
# MDP의 횟수가 랜덤이라서 while문을 사용한다
while True:
    # [0,4) 에서 정수값 1개를 임의로 선택합니다. 기본적인 설정이 high 값은 포함하지 않는다는 것에 유의!        
    print("At t = {}".format(step_counter))
    env._render()
    
    cur_state = env.s
    action = np.random.randint(low=0, high=4)
    next_state, reward, done, info = env.step(action)    
    
    print("state : {}".format(cur_state))
    print("aciton : {}".format(action_mapper[action]))
    print("reward : {}".format(reward))
    print("next state : {} \n".format(next_state))
    step_counter += 1
    if done:
        break

## 여러 에피소드를 시뮬레이션 해보기
'''
`GridworldEnv` 의 상태천이 행렬 $P$ 가 결정적이지만, 
정책함수 $\pi$ 가 추계적이므로 같은 시작 상태에서 에피소드를 시작하더라도 
각 에피소드에서 방문한 상태 및 에피소드의 길이가 다를 수 있습니다. 그 또한 한번 시뮬레이션 해보도록하죠
'''
# 여기서 env는 위에서 정의해 준 환경, s0는 초기 위치를 말한다
def run_episode(env, s0):
    _ = env.reset() # Gridworld 를 초기화합니다.(=현재 상태 위치 리셋)
    env.s = s0
    
    step_counter = 0
    while True:
        action = np.random.randint(low=0, high=4)
        next_state, reward, done, info = env.step(action)

        step_counter += 1
        if done:
            break
    return step_counter

n_episodes = 10
s0 = 6

for i in range(n_episodes):
    len_ep = run_episode(env, s0)
    print("Episode {} | Length of episode : {}".format(i, len_ep))