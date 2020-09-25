import sys; sys.path.append('C:/Users/DoubleS/Documents/Deep_learning/deep_learning_practice/강화학습') # add project root to the python path

from os.path import join
import gym
import torch

from src.part3.MLP import MultiLayerPerceptron as MLP
from src.part4.PolicyGradient import REINFORCE
from src.common.train_utils import EMAMeter, to_tensor
from src.common.memory.episodic_memory import EpisodicMemory

env = gym.make('CartPole-v1')
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.n

## Batch REINFORCE 알고리즘 실습
'''
이전 실습에서 배웠던 REINFORCE 알고리즘은 새로운 Episode가 주어지면 매 step에 대해 업데이트를 수행했었던 것 기억하시죠?
 하지만 많은 경우, REINFORCE 알고리즘 및 PG 알고리즘을 구현할 때는 계산상의 효율성을 위해서, 
 Episode (혹은 여러개의 Episode들) 단위로 업데이트하는 경우가 좀더 흔합니다. 
 이번에는 Episode들을 배칭해서 업데이트하는, Batch update 버젼의 REINFORCE 알고리즘들을 구현해봅시다. 
'''
### 수업에서 배운 수식을 잠깐 복습해볼까요?
'''
>REINFORCE (1992):
theta leftarrow theta + alpha nabla_{theta}ln pi_{theta}(A_t|S_t)G_t

>Episodic update REINFORCE
theta leftarrow theta + alphafrac{1}{T}biggr(sum_{t=1}^{T}nabla_{theta}ln pi_{theta}(A_t|S_t)G_tbiggr)

>Batch episodic update REINFORCE:
theta leftarrow theta + alphafrac{1}{sum_{i=1}^{N} T^{i}} biggr(sum_{i=1}^{N}sum_{t=1}^{T^i}nabla_{theta}ln pi_{theta}(A_t^i|S_t^i)G_t^ibiggr)

i 는 에피소드 인덱스, T^i는 에피소드 i 의 길이, A_t^i, S_t^i, G_t^i는 에피소드 i 의 t 시점의 행동, 상태, 리턴을 의미합니다.
'''
net = MLP(s_dim, a_dim, [128])
agent = REINFORCE(net)
memory = EpisodicMemory(max_size=100, gamma=1.0)
ema = EMAMeter()

### New kids on the block `EpisodicMemory`
'''
`EpisodicMemory` 라는 이전까지 보지 못한 새로운 녀석이 나타났네요. `EpisodicMemory`는 sample (s_t, a_t, r_t, s_{t+1}, text{done}) 을 저장해주고 리턴을 계산해주는 장치입니다. 한번 살펴보도록 할까요?

> 전체 코드는 `src.common.memory.episodic_memory.py` 를 참조하세요. <br>
> 참고) `EpisodicMemory`는 PG 구현에 필수적인것은 아닙니다. 사용하게 되면 구현할 때 신경쓸 부분이 적어져서 좋습니다.

```python
import torch
from collections import deque
from src.common.memory.trajectory import Trajectory


class EpisodicMemory:

    def __init__(self, max_size: int, gamma: float):
        self.max_size = max_size  # maximum number of trajectories
        self.gamma = gamma
        self.trajectories = deque(maxlen=max_size)
        self._trajectory = Trajectory(gamma=gamma)

    def push(self, state, action, reward, next_state, done):
        self._trajectory.push(state, action, reward, next_state, done)
        if done:
            self.trajectories.append(self._trajectory)
            self._trajectory = Trajectory(gamma=self.gamma)

    def reset(self):
        self.trajectories.clear()
        self._trajectory = Trajectory(gamma=self.gamma)

    def get_samples(self):
        # May require some modification depending on the environment.
        
        states, actions, rewards, next_states, dones, returns = [], [], [], [], [], []
        while self.trajectories:
            traj = self.trajectories.pop()
            s, a, r, ns, done, g = traj.get_samples()
            states.append(torch.cat(s, dim=0))
            actions.append(torch.cat(a, dim=0))
            rewards.append(torch.cat(r, dim=0))
            next_states.append(torch.cat(ns, dim=0))
            dones.append(torch.cat(done, dim=0))
            returns.append(torch.cat(g, dim=0))

        states = torch.cat(states, dim=0)
        actions = torch.cat(actions, dim=0)
        rewards = torch.cat(rewards, dim=0)
        next_states = torch.cat(next_states, dim=0)
        dones = torch.cat(dones, dim=0)
        returns = torch.cat(returns, dim=0)

        return states, actions, rewards, next_states, dones, returns
    
```
'''
'''
`Trajectory` 전체 코드는 `src.common.memory.trajectory.py`를 참조해주세요.
```python
class Trajectory:

    def __init__(self, gamma: float):
        self.gamma = gamma
        self.states = list()
        self.actions = list()
        self.rewards = list()
        self.next_states = list()
        self.dones = list()

        self.length = 0
        self.returns = None
        self._discounted = False

    def push(self, state, action, reward, next_state, done):
        if done and self._discounted:
            raise RuntimeError("done is given at least two times!")

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.length += 1

        if done and not self._discounted:
            # compute returns
            self.compute_return()

    def compute_return(self):
        rewards = self.rewards
        returns = list()

        g = 0
        # iterating returns in reverse order
        for r in rewards[::-1]:
            g = r + self.gamma * g
            returns.insert(0, g)
        self.returns = returns
        self._discounted = True

    def get_samples(self):
        return self.states, self.actions, self.rewards, self.next_states, self.dones, self.returns

```
'''

n_eps = 1000
update_every = 2 # Update every `update_every` episodes
print_every = 50

for ep in range(n_eps):
    s = env.reset()
    cum_r = 0

    states = []
    actions = []
    rewards = []

    while True:
        s = to_tensor(s, size=(1, 4))
        a = agent.get_action(s)
        ns, r, done, info = env.step(a.item())
        
        # preprocess data
        r = torch.ones(1,1) * r
        done = torch.ones(1,1) * done
        
        memory.push(s,a,r,torch.tensor(ns),done)
                
        s = ns
        cum_r += r
        if done:
            break

    ema.update(cum_r)
    if ep % print_every == 0:
        print("Episode {} || EMA: {} ".format(ep, ema.s))
    
    if ep % update_every == 0:
        s,a, _, _, done, g = memory.get_samples()
        agent.update_episodes(s, a, g, use_norm=True)
        memory.reset()

## `wandb` 를 이용한 Happy logging 및 저장
'''
> 저는 `wandb`와 아무런 이해관계도 없습니다. 제가 여러가지를 써보니까 이게 가장 편하고 강력하더라고요.

REINFORCE 및 다른 강화학습 알고리즘을 할 때, 수동으로 결과를 저장/출력 (logging)하고 여러번의 반복실험을 거쳐서 결과를 저장했던 지난날들이 있었죠? 

1. 이런 방식은 새로운 모델을 만들고 훈련할때마다 로깅하는 방식이 달라질수 있기 때문에, 매번 새롭게 로깅하는 코드를 작성해야 했고
2. 모델을 저장할 때도 저장된 모델이, 매번 어떤 hyperparameter 였는지 혹은 어떤 알고리즘이었는지 추가적으로 따로 관리해야하는 단점이 있습니다.

사소해 보이는 단점이긴 하지만, 연구나 대규모의 실험을 진행하다보면 이 과정이 매우 번거롭습니다. 
이번 기회에 `wandb` (Weight AND Bias)로 앞서 이야기한 과정을 자동화하고, 
추가적으로 모델의 학습과정 및 저장된 모델을 웹 기반으로 관리할 수 있는 방법에 대해서 이야기 해보면 좋겠네요.
'''
## wandb 설정하기
'''
한번 `wandb`를 설정해보도록 할까요?

#### 1. wandb 설치하기
```bash
pip install wandb
```

#### 2. wandb 계정 만들기 <br>
wandb [홈페이지](https://www.wandb.com/) 에 접속해서 계정을 만들어주세요. 
학교에 재학하는 중이시라면 학교 계정 (ex. nonexist@kaist.ac.kr) 으로 계정을 만들면 
wandb의 pro 기능을 무료로 사용할 수 있으니 학교 계정으로 가입하는것도 좋은 선택입니다.

#### 3. 로컬에서 wandb login 하기
```bash
wandb login 'your-api-key'
```

`'your-api-key'`는 https://app.wandb.ai/settings 페이지에서 `API keys` 라는 항목에서 찾을 수 있습니다.

더 많은 wandb 설명 문서는 [여기](https://docs.wandb.com/)를 참조해주세요.
'''
## wandb로 logging 하기

import wandb
import json

net = MLP(s_dim, a_dim, [128])
agent = REINFORCE(net)
memory = EpisodicMemory(max_size=100, gamma=1.0)

n_eps = 1000
update_every = 2 # Update every `update_every` episodes

### `wandb.config`로 모델의 Configuration 을 기록해보기
'''
일단 하나의 실험의 configuration을 기록해보도록 할까요?
'''
config = dict()
config['n_eps'] = n_eps
config['update_every'] = update_every

wandb.init(project='my-first-wandb-project', config=config)

### `wandb.log()`로 logging 하기
'''
`wandb`의 정말 큰 장점중 하나는 원래 code에 단 몇줄의 수정으로 원하는 값들을 logging 할수 있다는 것입니다.
'''
for ep in range(n_eps):
    s = env.reset()
    cum_r = 0

    states = []
    actions = []
    rewards = []

    while True:
        s = to_tensor(s, size=(1, 4))
        a = agent.get_action(s)
        ns, r, done, info = env.step(a.item())
        
        # preprocess data
        r = torch.ones(1,1) * r
        done = torch.ones(1,1) * done
        
        memory.push(s,a,r,torch.tensor(ns),done)
                
        s = ns
        cum_r += r
        if done:
            break    
    
    if ep % update_every == 0:
        s,a, _, _, done, g = memory.get_samples()
        agent.update_episodes(s, a, g, use_norm=True)
    
    log_dict = dict()
    log_dict['cum_return'] = cum_r
    wandb.log(log_dict)


# Save model and experiment configuration
json_val = json.dumps(config)
with open(join(wandb.run.dir, 'config.json'), 'w') as f:
    json.dump(json_val, f)
    
torch.save(agent.state_dict(), join(wandb.run.dir, 'model.pt'))

# close wandb session
wandb.join()

## 원격 저장소에서 저장된 파일 불러오기
'''
`wandb`를 활용하면 손쉽게 logging을 할 수 있는것 외에도 장점이 많습니다. 
`wandb`의 장점은 Hyperparameter를 최적화해주는 hyperparmeter sweeping, 수려한 plotting 등이 있지만, 
그 외에도 이번 실습에서 확인해볼 기능은 원격저장소에서 저장된 파일을 내려받는 기능입니다.

이 기능을 활용해서, 특정 `wandb` run 에 해당하는 모델의 파라미터를 불러와볼까요?
'''
# wandb 사이트에 가서 방금 했던 프로젝트 고르고 overview에 가면 Run path가 있는데 그걸 복사해서 붙여넣는다
# wandb 사이트 리모트 저장소에서 config.json 파일을 불러온다
wandb_run_path = 'doubles/my-first-wandb-project/2k9q8sji'
model_config_path = wandb.restore('config.json', wandb_run_path, replace=True)

with open(model_config_path.name, "r") as f:
    config_str = json.load(f)
    config_loaded = json.loads(config_str)

print("Config")
print(config)

print("loaded config")
print(config_loaded)

model_path = wandb.restore('model.pt', wandb_run_path, replace=True)

net2 = MLP(s_dim, a_dim, [128])
agent2 = REINFORCE(net2)

agent2.load_state_dict(torch.load(model_path.name))

agent.state_dict()

agent2.state_dict()

### Wandb 를 활용해서 결과를 확인해봅시다.
'''
여러번의 run을 돌려서 batch update REINFORCE 실험들을 확인해볼까요?
결과는 [여기](https://app.wandb.ai/junyoung-park/reinforce_exp?workspace=user-junyoung-park)에서 확인해볼수 있습니다.
'''