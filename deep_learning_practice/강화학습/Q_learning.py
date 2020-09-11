import sys; sys.path.append('C:/Users/DoubleS/Documents/[패캠]딥러닝-인공지능-강의자료/강화학습/ReinforcementLearningAtoZ-master') # add project root to the python path

import numpy as np
import matplotlib.pyplot as plt

from src.part2.temporal_difference import QLearner
from src.common.gridworld import GridworldEnv
from src.common.grid_visualization import visualize_value_function, visualize_policy

np.random.seed(0)

## `GridWorld` 초기화하기
'''
가로로 `nx` 개, 세로로 `ny` 개의 칸을 가진 `GridworldEnv`를 만듭니다!
'''
nx, ny = 4, 4
env = GridworldEnv([ny, nx])

## Q-Learning 에이전트 초기화하기
'''
`QLearner`는 이전에 배웠던 `TDAgent`를 상속받는 형태로 구현되었습니다. `QLearner`의 생성자는 다음과 같습니다.

```python
class QLeaner(TDAgent):

    def __init__(self,
                 gamma: float,
                 num_states: int,
                 num_actions: int,
                 epsilon: float,
                 lr: float):
        super(QLeaner, self).__init__(gamma=gamma,
                                      num_states=num_states,
                                      num_actions=num_actions,
                                      epsilon=epsilon,
                                      lr=lr,
                                      n_step=1)

```

각 인자의 의미는 다음과 같습니다.
1. `gamma` : 감가율
2. `num_states` : 상태공간의 크기 (서로 다른 상태의 갯수)
3. `num_actions` : 행동공간의 크기 (서로 다른 행동의 갯수)
4. `epsilon`: $\epsilon$-탐욕적 정책의 파라미터
5. `lr` : 학습률
'''
qlearning_agent = QLearner(gamma=1.0,
                           lr=1e-1,
                           num_states=env.nS,
                           num_actions=env.nA,
                           epsilon=1.0)

## Q-learning 업데이트
'''
Q-learning 업데이트는 다음의 의사코드와 같이 진행됩니다.

<img src="./images/q_learning.png" width="60%" height="100%" title="px(픽셀) 크기 설정" alt="qlearning_update"></img> 

알고리즘 의사코드를 파이썬으로 구현하면 다음과 같습니다.

```python
def update_sample(self, state, action, reward, next_state, done):
    s, a, r, ns = state, action, reward, next_state
    # Q-Learning target
    td_target = r + self.gamma * self.q[ns, :].max() * (1 - done)
    self.q[s, a] += self.lr * (td_target - self.q[s, a])
```
'''
num_eps = 10000
report_every = 1000
qlearning_qs = []
iter_idx = []
qlearning_rewards = []


for i in range(num_eps):
    
    reward_sum = 0
    env.reset()    
    while True:
        state = env.s
        action = qlearning_agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        
        qlearning_agent.update_sample(state=state,
                                      action=action,
                                      reward=reward,
                                      next_state=next_state,
                                      done=done)
        reward_sum += reward
        if done:
            break
    
    qlearning_rewards.append(reward_sum)
    
    if i % report_every == 0:
        print("Running {} th episode".format(i))
        print("Reward sum : {}".format(reward_sum))
        qlearning_qs.append(qlearning_agent.q.copy())
        iter_idx.append(i)

num_plots = len(qlearning_qs)
fig, ax = plt.subplots(num_plots, figsize=(num_plots*5*5, num_plots*5))
for i, (q, viz_i) in enumerate(zip(qlearning_qs, iter_idx)):
    visualize_policy(ax[i], q, env.shape[0], env.shape[1])
    _ = ax[i].set_title("Greedy policy at {} th episode".format(viz_i))