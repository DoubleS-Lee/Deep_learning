# Monte-Carlo
# 모델은 모르고 샘플링을 통해 추정한는 방법
# 에피소드가 끝나가 Return(G)과 가치함수(Q)가 업데이트 되는 단점이 있음
# 1 MC policy evaluation
# 1-1) N(s) <- G(s)/N(s)
# 1-2) G(s) : 상태 s의 리턴의 합, N(s) : 상태 s의 방문 횟수
# 1-3) 작동 순서 : 현재상태 관측env.observe() -> 정책함수로 현재 상태에 대한 행동 결정mc_agent.get_action(cur_state) -> 행동 시행env.step(action)
# 1-4) ExactMCAgent를 상속받아 적용

# 2 Incremental MC policy evaluation
# 2-1) V(s) <- V(s) + (G_t - V(s))/N(s)

# 3 2)에서 학습률 도입
# 3-1) V(s) <- V(s) + alpha*(G_t - V(s))
# 3-2) MCAgent를 상속받아 적용


'''
반복:
    에피소드 시작
    반복:
        현재 상태 <- 환경으로 부터 현재 상태 관측
        현재 행동 <- 에이전트의 정책함수(현재 상태)
        다음 상태, 보상 <- 환경에 '현재 행동'을 가함
        if 다음 상태 == 종결 상태
            반복문 탈출
    에이전트의 가치함수 평가 및 정책함수 개선
'''

import sys; sys.path.append('C:/Users/Seohee/Documents/Deep_learning/deep_learning_practice/강화학습') # add project root to the python path

import numpy as np
import matplotlib.pyplot as plt

from src.part2.monte_carlo import ExactMCAgent, MCAgent
from src.common.gridworld import GridworldEnv
from src.common.grid_visualization import visualize_value_function, visualize_policy

np.random.seed(0)

## `GridWorld` 초기화하기
'''
가로로 `nx` 개, 세로로 `ny` 개의 칸을 가진 `GridworldEnv`를 만듭니다!
'''
nx, ny = 4, 4
env = GridworldEnv([ny, nx])


## Monte-carlo '에이전트' 초기화하기
'''
또한, 우리가 평가하려는 정책은 행동 가치함수 Q(s,a) 에 대한 'epsilon-탐욕적 정책' 이라고 생각해보겠습니다. 이제 한번 파이썬 구현체를 살펴보도록 할까요?
'''
# epsilon=1.0? -> 모든 행동을 같은 확률로 하는 정책
mc_agent = ExactMCAgent(gamma=1.0,num_states=nx * ny, num_actions=4, epsilon=1.0) 

# 좀 더 직관적인 가시화를 위해서 action 인덱스를 방향으로 바꿔줍니다.
action_mapper = {
    0: 'UP',
    1: 'RIGHT',
    2: 'DOWN',
    3: 'LEFT'
}

## My first '에이전트-환경' interaction
'''
강화학습을 구현할 때 전형적인 형태의 `'에이전트-환경' interaction` 은 다음과 같습니다.

```
반복:
    에피소드 시작
    반복:
        현재 상태 <- 환경으로 부터 현재 상태 관측
        현재 행동 계산 <- 에이전트의 정책함수(현재 상태)
        다음 상태, 보상 획득 <- 환경에 '현재 행동'을 가함
        if 다음 상태 == 종결 상태
            반복문 탈출
    에이전트의 가치함수 평가 및 정책함수 개선
```

파이썬을 활용한 구현체를 확인해볼까요?

1. 에피소드 (재) 시작하기 
우리가 사용할 `GridworldEnv`에서는, `env.reset()`을 활용해서 주어진 환경을 재시작합니다.

2. 환경에서 현재 상태 관측하기
우리가 사용할 `GridworldEnv`에서는, `env.observe()` 를 활용해서 현재 상태를 관측합니다.

3. 현재 상태로 부터 정책함수로 행동 결정하기
`action = mc_agent.get_action(cur_state)` 을 활용해서 정책함수로 현재 상황에 대한 행동을 구할 수 있습니다.

4. 현재 행동을 환경에 가하기
`next_state, reward, done, info = env.step(action)` 을 활용해서 현재 상태에서 주어진 행동을 가한 후(!) 의 상태 `next_state` , 
그에 따른 보상 `reward`, 다음 상태가 종결상태인지 여부 `done` 및 환경에 대한 정보 `info`를 확인 할 수 있습니다.
'''
### Note
'''
여러분들이 사용할 모든 환경이 `env.reset()`, `'env.step()'` 과 같이 표준화된 인터페이스를 제공하지 않을수도 있습니다.
 다행히도, `gym` 환경을 상속받아 만들어진 환경들은 앞서 설명드린 표준화된 인터페이스를 갖추는 것을 권장하고 있습니다. 
 차후에 `gym` 의 환경을 상속 받은 환경을 사용하시게 된다면 표준화된 인터페이스가 구현되어있는 지 확인해보시는 것도 좋을것같네요. 
 또 여러분들께서 직접 환경을 구축하게 된다면, 해당 인터페이스를 구현하시는게 좋겠죠?
'''
env.reset()
step_counter = 0
while True:
    print("At t = {}".format(step_counter))
    env._render()
    
    cur_state = env.observe()
    action = mc_agent.get_action(cur_state)
    next_state, reward, done, info = env.step(action)
    
    print("state : {}".format(cur_state))
    print("aciton : {}".format(action_mapper[action]))
    print("reward : {}".format(reward))
    print("next state : {} \n".format(next_state))
    step_counter += 1
    if done:
        break
#%%
# 1 MC policy evaluation
############################################################################################################################################
## Monte-calro 정책 평가
'''
이제 Vanilla version의 Monte-carlo Policy evaluation을 수행해보도록 할까요?
'''
def run_episode(env, agent):
    env.reset()
    states = []
    actions = []
    rewards = []
    
    while True:
        state = env.observe()
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        
        if done:
            break
    
    episode = (states, actions, rewards)
    agent.update(episode)

### `agent.update(episode)`가 하는 일
'''
우리 실습에서는 every-visit Monte-carlo 정책 평가를 활용해서 정책에 해당하는 가치함수를 추산할 것입니다. 

Monte-carlo 정책평가는 하나의 온전한 에피스드가 필요했던거 다들 기억하시죠? 
따라서 하나의 에피소드가 끝난 후에 `agent.update(episode)`를 수행하게 됩니다. 그러면 `agent.update(episode)` 에서는 무슨일이 일어날까요?

'''
mc_agent.reset_statistics() # agent.n_v, agent.n_q, agent.s_v, agent.s_q 을 0으로 초기화 합니다.
for _ in range(10):  
    run_episode(env, mc_agent)

## agent.compute_values()
# value func 계산
'''
앞서 추산한 리턴의 추산치와 각 상태 s 및 상태-행동 (s,a) 방문 횟수를 활용해서 상태 가치함수 V 와 행동 가치함수 Q 를 계산합니다.
'''
# V와 Q를 계산한다
mc_agent.compute_values()

mc_agent.v
mc_agent.q

## Monte-carlo 방식으로 추산한 V(s) 이 정말 맞을까요?
'''
우리는 이 `GridworldEnv` 에 대해서 정답을 알고 있죠? 
바로 `동적 계획법`을 통해서 계산한 $V(s)$ 입니다. 
여기서는 `Monte-carlo` 로 추산한 가치함수와 동적 계획법으로 계산한 가치함수의 값을 비교해볼까요?
'''
from src.part2.tensorized_dp import TensorDP

dp_agent = TensorDP()
dp_agent.set_env(env)


v_pi = dp_agent.policy_evaluation()

fig, ax = plt.subplots(1,2, figsize=(12,6))
visualize_value_function(ax[0], mc_agent.v, nx, ny)
_ = ax[0].set_title("Monte-carlo Policy evaluation")

visualize_value_function(ax[1], v_pi, nx, ny)
_ = ax[1].set_title("Dynamic programming Policy evaluation")

## Monte-carlo 기법에 실망하셨나요? 
'''
`dp_agent` 와 `mc_agent`에게 비슷한 시간을 주고 가치 함수를 평가해봤었는데
`mc_agent` 의 결과가 영 시원치 않죠? 바로 `MDP` 환경모델을 활용 여부에 따른 차이입니다.

`dp_agent`는 환경에 대해 훤히 알고 있으니, 짧은 시간 (혹은 계산) 만에 원하는 답을 알아내는 것은
어쩌면 당연하겠죠. `mc_agent`에게 조금 더 시간을 줘 보는게 어떨까요?
'''
total_eps = 2000
log_every = 500

def run_episodes(env, agent, total_eps, log_every):
    mc_values = []
    log_iters = []

    agent.reset_statistics()
    for i in range(total_eps+1):  
        run_episode(env, agent)

        if i % log_every == 0:
            agent.compute_values()
            mc_values.append(agent.v.copy())
            log_iters.append(i)
    
    info = dict()
    info['values'] = mc_values
    info['iters'] = log_iters
    return info

info = run_episodes(env, mc_agent, total_eps, log_every)

log_iters = info['iters']
mc_values = info['values']

n_rows = len(log_iters)
figsize_multiplier = 10


fig, ax = plt.subplots(n_rows, 2, figsize=(n_rows*figsize_multiplier*0.5, 
                                           3*figsize_multiplier))

for viz_i, i in enumerate(log_iters):
    visualize_value_function(ax[viz_i, 0], mc_values[viz_i], nx, ny,
                            plot_cbar=False)
    _ = ax[viz_i, 0].set_title("MC-PE after {} episodes".format(i), size=20)

    visualize_value_function(ax[viz_i, 1], v_pi, nx, ny,
                             plot_cbar=False)
    _ = ax[viz_i, 1].set_title("DP-PE", size=20)

fig.tight_layout()

## 혹시 눈치 채셨나요? 매 실행마다 결과값이 달라진다는 것을?
'''
Monte-carlo Policy evaluation 에서는 매 실행마다, 가치함수 추산값이 달라지는것을 확인하셨나요?
그러면 한번 매 실행마다 얼마나 결과값이 다른지, 즉, 가치함수 `추산치의 분산`이 얼마나 되는지 확인해볼까요?
'''
reps = 10
values_over_runs = []
total_eps = 3000
log_every = 30

for i in range(reps):
    print("start to run {} th experiment ... ".format(i))
    info = run_episodes(env, mc_agent, total_eps, log_every)
    values_over_runs.append(info['values'])
    
values_over_runs = np.stack(values_over_runs)

values_over_runs.shape

v_pi_expanded = np.expand_dims(v_pi, axis=(0,1))

errors = np.linalg.norm(values_over_runs - v_pi_expanded, axis=-1)
error_mean = np.mean(errors, axis=0)
error_std = np.std(errors, axis=0)

np.save('mc_errors.npy', errors)

fig, ax = plt.subplots(1,1, figsize=(10,5))
ax.grid()
ax.fill_between(x=info['iters'],
                y1=error_mean + error_std,
                y2=error_mean - error_std,
                alpha=0.3)
ax.plot(info['iters'], error_mean, label='Evaluation error')
ax.legend()
_ = ax.set_xlabel('episodes')
_ = ax.set_ylabel('Errors')

#%%
## Incremental Monte-carlo 정책 평가
# 3 Incremental MC policy evaluation에 학습률 적용
# V(s) <- V(s) + alpha*(G_t - V(s))

'''
앞선 예제들에서는 Vanilla version의 MC 정책 평가에 대해서 살펴보았습니다. 
이 방식은 가치함수 추산에 리턴들의 합과 각 상상태 s 및 상태-행동 (s,a) 방문 횟수를 따로 기록하여 두 통계치를 활용해서 가치함수들을 추산하였습니다. 
이번에는 `적당히 작은` 학습률 alpha(learning rate; lr)을 활용하는 방식을 이용해서 정책 평가를 수행해보도록 할까요?

V(s) <- V(s) + alpha*(G_t - V(s))

`MCAgent` 는 기존의 `ExacatMCAgent`와 유사하나 추가적으로 학습률 alpha 인자를 하나 더 받습니다.
'''
mc_agent = MCAgent(gamma=1.0, lr=1e-3, num_states=nx * ny, num_actions=4, epsilon=1.0)

## `MCAgent.update()`
'''
새로운 `MCAgent`의 학습방식이 학습률 alpha을 활용하니, 그에 따라 `update()` 함수도 조금 수정이 필요하겠죠? 
수정된 `update()`함수를 살펴볼까요?
update() 함수는 MCAgent 클래스 안에 있다

```python
def update(self, episode):
    states, actions, rewards = episode

    # reversing the inputs!
    # for efficient computation of returns
    states = reversed(states)
    actions = reversed(actions)
    rewards = reversed(rewards)

    iter = zip(states, actions, rewards)
    cum_r = 0
    for s, a, r in iter:
        cum_r *= self.gamma
        cum_r += r

        self.v[s] += self.lr * (cum_r - self.v[s])
        self.q[s, a] += self.lr * (cum_r - self.q[s, a])
```
'''
for _ in range(5000):
    run_episode(env, mc_agent)

fig, ax = plt.subplots(1,2, figsize=(12,6))
visualize_value_function(ax[0], mc_agent.v, nx, ny)
_ = ax[0].set_title("Monte-carlo Policy evaluation")

visualize_value_function(ax[1], v_pi, nx, ny)
_ = ax[1].set_title("Dynamic programming Policy evaluation")

## MCAgent, 다른 학습률 $\alpha$에 대해선 어떨까?
'''
MCAgent의 학습률 $\alpha$ 는 분명히 상태 (행동) 가치함수 추산에 영향을 미칠텐데, 어떻게 영향을 미치는지 알아보도록 할까요?
# learning rate 값이 학습에 많은 영향을 미치게 된다
'''
def run_mc_agent_with_lr(agent, env, lr):
    agent.reset()
    agent.lr = lr
    
    for _ in range(5000):
        run_episode(env, agent)
    
    fig, ax = plt.subplots(1,2, figsize=(12,6))
    visualize_value_function(ax[0], mc_agent.v, nx, ny)
    _ = ax[0].set_title("Monte-carlo Policy evaluation")

    visualize_value_function(ax[1], v_pi, nx, ny)
    _ = ax[1].set_title("Dynamic programming Policy evaluation")

run_mc_agent_with_lr(agent=mc_agent,
                     env=env,
                     lr=1.0)

run_mc_agent_with_lr(agent=mc_agent,
                     env=env,
                     lr=1e-1)

run_mc_agent_with_lr(agent=mc_agent,
                     env=env,
                     lr=1e-2)