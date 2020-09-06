import sys; sys.path.append('C:/Users/Seohee/Documents/01.강의자료/모델성능개선으로 익히는 강화학습/ReinforcementLearningAtoZ-master') # add project root to the python path

# 마르코프 의사결정 과정 MDP
'''
MDP는 강화 학습문제를 정의하는 수학적 장치입니다. MDP는 일반적으로 $<\cal{S}, \cal{A}, P, R, \gamma>$ 인 튜플로 정의됩니다.
이번 실습에서는 MDP의 각 요소를 확인해봅시다!

당분간 MDP의 예제로 `GridworldEnv`를 사용할 것입니다. `GridworldEnv`는 openai `gym` 의 `discrete.DiscreteEnv` 를 상속받아서 만들어졌습니다. openai `gym`은 강화학습 환경으로 사용하기에 적합한 표준화된 인터페이스를 제공하기 때문에, 많은 강화학습 환경들의 베이스 클래스로 사용됩니다.
'''

from src.common.gridworld import GridworldEnv # Gridworld Environment

num_y, num_x = 4, 4
env = GridworldEnv(shape=[num_y, num_x])
'''
`4 X 4 그리드월드` 

우리의 MDP 환경을 시각화하면 아래의 그림과 같습니다.

```
===========
T  x  o  o
o  o  o  o
o  o  o  o
o  o  o  T
===========
```

T: 도착점 (종결상태, Terminal state) <br>
x: 현재 위치 $s_t$<br>
o: 다른 환경의 점
'''
## 상태공간 $\cal{S}$ 와 행동공간 $\cal{A}$
'''
그리드 월드에서 상태 공간 $\cal{S}$은 가로 4개 세로 4개의 격자로 전체 16개의 상태가 존재합니다. <br>
그리드 월드에서 행동 공간 $\cal{A}$은 모든 상태 $s$ 에서 

>`위로 움직이기`, `오른쪽으로 움직이기`, `아래로 움직이기`, `왼쪽으로 움직이기` 

로 4가지입니다. <br>

만약 가장자리에서 바깥으로 나가는 `행동`을 하게되면, 에이전트의 위치는 바뀌지 않습니다. <br> 
예를들어, 우상단에서 `위로 움직이기` 혹은 `오른쪽으로 움직이기`를 선택해서 행동하게 되면 에이전트는 움직이지 않게됩니다.
'''

observation_space = env.observation_space
action_space = env.action_space
print("Number of states: {}".format(observation_space))
print("Number of actions: {}".format(action_space))

## 상태천이 행렬 $P$, 과 보상함수 $R$
'''
강의에서 설명했듯, MDP의 상태천이 행렬 $P$는 사실은 행렬이 아닙니다. 명확하게 이야기하면 상태천이 `텐서` 라고 지칭하는게 명확하지만 편의를 위해 앞으로는 상태천이 행렬 (Transition matrix) 혹은 상태천이 모델 (Transition matrix)등으로 지칭하겠습니다.
'''
### 텐서 Tensor
'''
<img src="./images/tensor.png" width="100%" height="50%" title="px(픽셀) 크기 설정" alt="Tensor"></img>

앞으로 우리는 `n`차원 자료형을 Rank `n` 텐서라고 혹은 `n`차원 텐서라고 부를 것 입니다. `n` 을 직관적으로 이해하면, '텐서내에 특정 자료에 접근하기
위해서는 몇개의 인덱스가 필요한가' 라고 생각해보세요. 
'''
import numpy as np # 파이썬에서 계산과학 및 수학계산을 위해서 많이 사용되는 라이브러리인 `numpy`를 불러옵니다.

## `import numpy as np` ?
'''
```python
import numpy
import numpy as npy 
```

또한 모두 동일하게 작동합니다. 하지만 `numpy` 를 `np` 라고 임포트하는 것은
많은 사람들 사이에 관용적 표현입니다. <br>

따라서 아래의 경우도 __모두 잘 작동합니다__.

```python
import numpy as pd
import torch as np
import pandas as th
```

라고 임포트해도 사용상에는 아무 문제가 없지만 누군가가 여러분들의 코드를 보면
아마도 여러분께 화를 낼지도 모릅니다.
'''
num_row = 2
num_col = 2 

# [num_row x now_col] 행렬을 만듭니다. 이때 행렬의 각 원소는 임의의 값으로 채워집니다.
rank2_tensor = np.random.random(size=(num_row,num_col)) 

### 텐서와 친해지기

print(rank2_tensor)
'''
`numpy_array.shape()`은 각 차원별로 몇개의 인덱스가 있는지를 반환해줍니다.
'''
tesor_shape = rank2_tensor.shape
tensor_rank = len(tesor_shape)
print("Tensor shape : {}".format(tesor_shape))
print("Tensor rank : {}".format(tensor_rank))

## 상태천이 매트릭스 (텐서) $P$ 
'''
상태천이 매트릭스 $P$는 랭크 3 텐서 입니다. 첫번째 축은 행동 $a$에 대한 축이며, 둘째 축은 현재상태 $s$, 셋째 축은 다음상태 $s'$를 나타냅니다.


행동 $a$의 인덱스를 0,1,2,3로 주었으며, 각각 현재 위치에서 [상,우,하,좌] 로 움직이는 행동 입니다. <br>
(왜 인덱스는 0부터시작? 파이썬은 인덱싱을 0부터 시작합니다.)

상태 $s$의 종류 4x4=16개 입니다. 이에 따라 인덱스를 0,1,.., 15까지 주었습니다. 0은 좌측 상단 위치, 1은 좌상단에서 오른쪽으로 한칸, ... , 15는 우측 하단의 위치를 표현합니다. <br>
'''


### 코드에서 상태천이 텐서  $P$ 불러오기 

P = env.P_tensor
print("P shape : {}".format(P.shape))

### `Numpy` 배열의 장점, 직관적인(?) 편리한 인덱싱

# 행동 '상'(인덱스0)을 선택했을때 현재상태에서 다음상태로 전이하는 확률을 나타냄.
action_up_prob = P[0, :, :]

print(action_up_prob) 
'''
`GridworldEnv` 는 행동에 대한 상태천이가 결정적(deteministic) 하게 디자인했습니다.


>즉, 상태 천이 매트릭스 $P$ 의 각 열(row)의 원소가 하나를 제외하고 모두 0.0 입니다.


이 경우 `상` 이라는 행동을 하면, 무조건 위로 움직이게 됩니다. 하지만 일반적인 MDP에서는
에이전트의 행동이 결정적 (deterministic) 으로 환경에 영향을 미치지 않고 추계적으로(stochastic) 하게 영향을 미칩니다. 예를 들자면, `상` 이라는 행동을 했지만, 모종의 이유로 위로 가지않고 다른 방향으로 가게될 수도 있다는 것이죠.

>추계적 (stochastic) = '랜덤적인 혹은 확률적인 요소를 가지고 있다' 는 의미

하지만 결정적인 `GridworldEnv` 에서는 최적정책 및 최적가치를 직관적으로 이해할 수 있으니 당분간은
결정적인 환경을 고려해보도록 합시다.

## 상태천이 행렬의 특징을 기억하세요?

상태천이 행렬의 각 행(row)는 특정 상태 $s$에서 다음 상태 $s'$ 으로 이동할 확률을 나타냅니다.
따라서 (1) 상태천이 행렬의 모든 원소의 값은 항상 0보다 크거나 같고 (2) 각 열의 원소의 합은 1이 됩니다.

### (1) 그럼 모든 원소가 0보다 클까?

action_up_prob >= 0

### 눈으로 확인도 좋지만, 코딩으로도 확인해보자

우리는 매트릭스의 전체크기를 알고 있으므로 모든 원소가 양수인지를 코딩으로도 확인해볼 수 있습니다.

#### 방어적 프로그래밍

__운전에서도 방어운전이 제일이듯, 코딩도 방어적 프로그래밍이 제일입니다__ <br>

학습 알고리즘이 학습 할때는 우리가 디자인하지 않은 상황이 발생해도 그냥 학습이 진행되는 경우가 빈번합니다. 가끔은 연산중에 의도치 않은 오류로 `Nan`=__N__ot __a__ __N__umber 이 발생해도 학습이 진행됩니다. 그리고 나서 최악의 경우에 며칠후에 학습이 끝난 모델을 검증할 때서야, 

> 오... 1일때 부터 심각한 문제가 있었구나!

라는것을 알고 나서 울면서 다시 모델을 학습하는 경우들이 있습니다. <br>

가능하다면 코딩을 하실때도 각 요소가 어떻게 행동할지 알고 있다면, 코드내에서 요소들이 의도한대로 작동하는지 꼭 확인해보시길 바랍니다.
'''
is_greater_than_0 = action_up_prob >= 0
is_all_greater_than_0 = is_greater_than_0.sum() == is_greater_than_0.size

is_all_greater_than_0

### (2) 가로합은 1.0 일까?

action_up_prob.sum(axis=1) # 2 번째 축으로 합을 계산합니다. 파이썬은 숫자를 0부터 셉니다.

### 소소한 코딩팁. `Numpy` 의 auto inferencing 기능

action_up_prob.sum(axis=-1) # '마지막' 축으로 합을 계산합니다.

## 보상함수 $R$ 
'''
$R$ 는 Rank2 텐서, 즉 매트릭스입니다. 세로축은 상태를 가로축은 행동을 의미합니다.

우리의 `GridworldEnv`는 종결점에 도달하기 전까지는 어떤 상태에서 어떤 행동을 하든 매 이동마다 `-1` 의 보상을 받습니다.
'''
R = env.R_tensor
print(R)

## MDP 의 Episode
'''
MDP의 Episode는 $<(s_0, a_0, r_0, s_1), (s_1, a_1, r_1, s_2), ..., (s_{t}, a_{t}, r_{t}, s_{t+1}),..., (s_{T-1}, a_{T-1}, r_{T-1}, s_{T})>$ 으로 구성됩니다. 각 튜플은 매 시점 t의 상태 $s_t$, 선택한 행동 $a_t$, 보상 $r_t$, 그 다음 상태 $s_{t+1}$ 를 포함합니다. $T$는 종결 시점을 나타냅니다. 

하지만 일반적으로 RL 알고리즘을 구현할때는 구현의 편의를 위해 특정 시점이 종결 상태인지 아닌지를 확인하는 `done` 이라는 정보를 episode의 각 튜플에 추가해서 관리합니다.

`GridworldEnv`에서 에피소드를 한번 시뮬레이션 해보도록 하죠.

### MDP를 시뮬레이션하기 위해서는 정책 $\pi$ 가 필요합니다!

일단은 각 상태에서 모든 행동을 0.25의 확률로 고르는 정책함수로 `Gridworld`를 시뮬레이션 해봅시다.
'''
_ = env.reset() # Gridworld 를 초기화합니다.
print("Current position index : {}".format(env.s))

# 좀 더 직관적인 가시화를 위해서 action 인덱스를 방향으로 바꿔줍니다.
action_mapper = {
    0: 'UP',
    1: 'RIGHT',
    2: 'DOWN',
    3: 'LEFT'
}

step_counter = 0
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
`GridworldEnv` 의 상태천이 행렬 $P$ 가 결정적이지만, 정책함수 $\pi$ 가 추계적이므로 같은 시작 상태에서 에피소드를 시작하더라도 각 에피소드에서 방문한 상태 및 에피소드의 길이가 다를 수 있습니다. 그또한 한번 시뮬레이션 해보도록하죠
'''
def run_episode(env, s0):
    _ = env.reset() # Gridworld 를 초기화합니다.
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