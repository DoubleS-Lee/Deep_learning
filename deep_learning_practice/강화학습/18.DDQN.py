# DDPG (Deep Deterministic Policy Gradient)
# 연속적인 행동공간 a 에서 정책함수를 모델링하기 위해서 사용


import sys; sys.path.append('C:/Users/DoubleS/Documents/Deep_learning/deep_learning_practice/강화학습') # add project root to the python path

import torch
import gym
import numpy as np

from src.part3.MLP import MultiLayerPerceptron as MLP

from src.part5.DQN import prepare_training_inputs
from src.part5.DDPG import DDPG, Actor, Critic
from src.part5.DDPG import OrnsteinUhlenbeckProcess as OUProcess
from src.common.memory.memory import ReplayMemory
from src.common.train_utils import to_tensor
from src.common.target_update import soft_update

# Pendulum Env
'''
DDPG는 연속적인 행동공간 mathcal{A}에서 정책함수를 모델링하기 위해서 사용되었던 강화학습 알고리즘이었습니다. 
DDPG의 특징을 제대로 살펴보기 위해서는 우리도 그에 적합한 환경을 찾아보는게 좋겠죠? 우리가 이번 실습에 활용할 환경은 `Pendulum-v0`라는 환경입니다.

`Pendulum-v0` 환경의 상태 s 는 Pendulum 의 각도 theta의 코사인 값 cos(theta), 사인 값 sin(theta), 그리고 각속도 dot theta 로 구성되어 있습니다. 
환경의 행동 a는 Pendulum의 끝에 좌/우 방향으로 최대 2.0 의 토크값을 줄수 있습니다. 
즉, 우리의 행동공간 mathcal{A} = [-2.0, 2.0] 입니다. 보상 r 은 theta, dot theta, a 가 0에 가까워 질수록 높은 보상을 받습니다. 

이렇게 설정된 MDP에서 우리의 목적은 Pendulum 을 최대한 곧게 위의 방향으로 세우는 것입니다.
'''
from IPython.display import HTML
HTML('<img src="images/pendulum.gif">')

env = gym.make('Pendulum-v0')

# 저장되어 있는 파라미터(weight)들을 사용할거면 False라고 써놓는다
FROM_SCRATCH = False
# 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

## Deep deterministic Policy Gradient (DDPG)
'''
DDPG의 의사코드는 다음과 같습니다.

python 으로 DDPG를 한번 구현해볼까요?
```python
class DDPG(nn.Module):

    def __init__(self,
                 critic: nn.Module,
                 critic_target: nn.Module,
                 actor: nn.Module,
                 actor_target: nn.Module,
                 lr_critic: float = 0.0005,
                 lr_actor: float = 0.001,
                 gamma: float = 0.99):

        super(DDPG, self).__init__()
        self.critic = critic
        self.actor = actor
        self.lr_critic = lr_critic
        self.lr_actor = lr_actor
        self.gamma = gamma

        # setup optimizers
        self.critic_opt = torch.optim.Adam(params=self.critic.parameters(),
                                           lr=lr_critic)

        self.actor_opt = torch.optim.Adam(params=self.actor.parameters(),
                                          lr=lr_actor)

        # setup target networks
        critic_target.load_state_dict(critic.state_dict())
        self.critic_target = critic_target
        actor_target.load_state_dict(actor.state_dict())
        self.actor_target = actor_target

        self.criteria = nn.SmoothL1Loss()

    def get_action(self, state):
        with torch.no_grad():
            a = self.actor(state)
        return a

    def update(self, state, action, reward, next_state, done):
        s, a, r, ns = state, action, reward, next_state

        # compute critic loss and update the critic parameters
        with torch.no_grad():
            critic_target = r + self.gamma * self.critic_target(ns, self.actor_target(ns)) * (1 - done)
        critic_loss = self.criteria(self.critic(s, a), critic_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # compute actor loss and update the actor parameters
        actor_loss = -self.critic(s, self.actor(s)).mean()  # !!!! Impressively simple
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
```
'''

## Soft-target update
'''
Soft-target update는 moving target issue를 줄여 주기위해서 사용되는 trick 입니다. `pytorch` 에서는 간단하게
target network의 파라미터 update 대상이 되는 network의 파라미터를 복사하는 형태로, soft-target update를 진행할 수 있습니다. 
코드로 살펴볼까요?

```python
def soft_update(net, net_target, tau):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

```
'''
## Actor 와 Critic 의 구조
'''
`Pendulum-v0` 환경에 적합한 Actor 와 Critic Network를 디자인해볼까요? 
앞서 이야기한대로 행동공간 mathcal{A} = [-2.0, 2.0] 입니다. 기
존에 우리가 다루었던 모델들은 모델의 출력치의 범위가 제한되어있지 않았었는데, 모델의 출력치를 제한하려면 어떤방식을 사용할 수 있을까요? 
정답은 마지막 레이어의 Activation 함수를 잘 조절해주는 것입니다.

```python
class Actor(nn.Module):

    def __init__(self):
        super(Actor, self).__init__()
        self.mlp = MLP(3, 1,
                       num_neurons=[128, 64],
                       hidden_act='ReLU',
                       out_act='Identity')

    def forward(self, state):
        # Action space of Pendulum v0 is [-2.0, 2.0]
        return self.mlp(state).clamp(-2.0, 2.0)
```
'''

'''
`Critic`은 Q(s,a|theta^Q) 는 서로 다른 종류의 입력인 s, a를 받습니다. 
이런 경우에는 많은 경우에, 뉴럴 네트워크의 인풋수준에서 두개의 텐서를 하나로 합치는 `concatentation` 오퍼레이터를 많이 사용합니다. 
예를 들어 sa = [s ; a]. 그 후 하나로 합쳐진 인풋을 Q(s,a|theta^Q) 의 입력으로 넘겨주게 됩니다. 
하지만 이번 실습에서는 조금 다른 형태를 사용해보려고 합니다. 
각각의 서로다른 종류의 입력, 즉 정보, 를 각각의 sub-network를 통과시킨후 한번 가공된 정보인 hidden vector를 하나로 합친후, 
합쳐진 hidden vector에서 Q(s,a)를 추산하는 모델을 만들어 보겠습니다.

```python
class Critic(nn.Module):

    def __init__(self):
        super(Critic, self).__init__()
        self.state_encoder = MLP(3, 64,
                                 num_neurons=[],
                                 out_act='ReLU')  # single layer model
        self.action_encoder = MLP(1, 64,
                                  num_neurons=[],
                                  out_act='ReLU')  # single layer model
        self.q_estimator = MLP(128, 1,
                               num_neurons=[32],
                               hidden_act='ReLU',
                               out_act='Identity')

    def forward(self, x, a):
        emb = torch.cat([self.state_encoder(x), self.action_encoder(a)], dim=-1)
        return self.q_estimator(emb)
```
'''

lr_actor = 0.005
lr_critic = 0.001
gamma = 0.99
batch_size = 256
memory_size = 50000
tau = 0.001 # polyak parameter for soft target update
sampling_only_until = 2000

actor, actor_target = Actor(), Actor()
critic, critic_target = Critic(), Critic()

agent = DDPG(critic=critic,
             critic_target=critic_target,
             actor=actor,
             actor_target=actor_target).to(DEVICE)

memory = ReplayMemory(memory_size)

total_eps = 200
print_every = 10

env = gym.make('Pendulum-v0')

if FROM_SCRATCH:
    for n_epi in range(total_eps):
        ou_noise = OUProcess(mu=np.zeros(1))
        s = env.reset()
        cum_r = 0

        while True:
            s = to_tensor(s, size=(1, 3)).to(DEVICE)
            a = agent.get_action(s).cpu().numpy() + ou_noise()[0]
            ns, r, done, info = env.step(a)

            experience = (s,
                          torch.tensor(a).view(1, 1),
                          torch.tensor(r).view(1, 1),
                          torch.tensor(ns).view(1, 3),
                          torch.tensor(done).view(1, 1))
            memory.push(experience)

            s = ns
            cum_r += r

            if len(memory) >= sampling_only_until:
                # train agent
                sampled_exps = memory.sample(batch_size)
                sampled_exps = prepare_training_inputs(sampled_exps, device=DEVICE)
                agent.update(*sampled_exps)
                # update target networks
                soft_update(agent.actor, agent.actor_target, tau)
                soft_update(agent.critic, agent.critic_target, tau)        

            if done:
                break

        if n_epi % print_every == 0:
            msg = (n_epi, cum_r) # ~ -100 cumulative reward = "solved"
            print("Episode : {} | Cumulative Reward : {} |".format(*msg))

    torch.save(agent.state_dict(), 'ddpg_cartpole_user_trained.ptb')
else:
    agent.load_state_dict(torch.load('ddpg_cartpole.ptb'))

env = gym.make('Pendulum-v0')

s = env.reset()
cum_r = 0

while True:
    s = to_tensor(s, size=(1, 3)).to(DEVICE)
    a = agent.get_action(s).to('cpu').numpy()
    ns, r, done, info = env.step(a)
    s = ns
    env.render()
    if done:
        break
    
env.close()











