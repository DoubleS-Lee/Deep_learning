
# REINFORCE는 episode by episode로 업데이트가 이루어짐
# TD-actor-critic은 sample by sample로 업데이트가 이루어짐 


import sys; sys.path.append('C:/Users/DoubleS/Documents/Deep_learning/deep_learning_practice/강화학습') # add project root to the python path

import gym
import torch

from src.part3.MLP import MultiLayerPerceptron as MLP
from src.part4.ActorCritic import TDActorCritic
from src.common.train_utils import EMAMeter, to_tensor

env = gym.make('CartPole-v1')
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.n

# TD Actor-critic
'''
이번 실습에는 Vanilla version 의 TD Actor-critic을 만들어볼까요? 
> TD actor-critic은 Advantage function A(s,a)을 V_{psi}(s) 활용해서 추산하고 
그 값을 리턴대신 활용해서 Policy gradient 를 계산하는 기법인거 잊지 않으셨죠?

> A(s,a) ~~ delta_psi(s,a) = r+gamma*V_psi(s')-V(s)

`TD Actor-critic`의 의사 코드는 다음과 같습니다.
초기화 : 정책 pi_{theta}(a|s)의 매개변수 theta, 행동-가치함수 V_{phi}(s)
반복(에피소드):
    초기상태 s 관측
    반복(에피소드 내에서):
        행동 a를 pi_{theta}(a|s)로 결정 (샘플링)
        행동 a를 환경에 가한후 r, s'를 관측
        delta = r + gamma*V_{phi}(s') - V_{phi}(s)         Advantage function 계산
        phi <- phi - 학습률*ddelta/dphi                    delta를 phi에 대해 미분하여 업데이트 이를 이용하여 행동-가치함수 V_{phi}(s)를 구한다
        theta <- theta + alpha*delta*편미분_{theta}*ln(pi_{theta}(A_t|S_t))
        a <- a', s <- s'


파이썬으로 구현한 `TD Actor-critic` 은 어떻게 생겼을까요?

```python
class TDActorCritic(nn.Module):

    def __init__(self,
                 policy_net,
                 value_net,
                 gamma: float = 1.0,
                 lr: float = 0.0002):
        super(TDActorCritic, self).__init__()
        self.policy_net = policy_net
        self.value_net = value_net
        self.gamma = gamma
        self.lr = lr

        # use shared optimizer
        total_param = list(policy_net.parameters()) + list(value_net.parameters())
        self.optimizer = torch.optim.Adam(params=total_param, lr=lr)

        self._eps = 1e-25
        self._mse = torch.nn.MSELoss()
        
    def get_action(self, state):
        with torch.no_grad():
            logits = self.policy(state)
            dist = Categorical(logits=logits)
            a = dist.sample()  # sample action from softmax policy
        return a
```

'''

'''
가장 중요한 업데이트는 어떻게 생겼을까요?

```python
    def update(self, state, action, reward, next_state, done):
        # compute targets
        with torch.no_grad():
            td_target = reward + self.gamma * self.value_net(next_state) * (1-done)
            td_error = td_target - self.value_net(state)

        # compute log probabilities
        dist = Categorical(logits=self.policy_net(state))
        prob = dist.probs.gather(1, action)

        # compute the values of current states
        v = self.value_net(state)

        loss = -torch.log(prob + self._eps) * td_error + self._mse(v, td_target)
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```
'''
# 정책 pi_{theta}(a|s)
policy_net = MLP(s_dim, a_dim, [128])
# 행동-가치함수 V_{phi}(s)
value_net = MLP(s_dim, 1, [128])

agent = TDActorCritic(policy_net, value_net)
ema = EMAMeter()

## TD 알고리즘의 장점; sample-by-sample update
'''
TD Actor-critic 의 경우 Critic의 학습이 Temporal-difference 기법을 활용해서 진행되므로, 
하나의 에피소드가 끝나기 전에 각각의 transition sample 만으로도 학습을 진행할 수 있습니다. 
이번 예제에서는 sample-by-sample update를 수행하는 TD Actor-critic을 훈련해볼까요?
'''
n_eps = 10000
print_every = 500

for ep in range(n_eps):
    s = env.reset()
    cum_r = 0

    while True:
        s = to_tensor(s, size=(1, 4))
        a = agent.get_action(s)
        ns, r, done, info = env.step(a.item())
        
        ns = to_tensor(ns, size=(1,4))
        agent.update(s, a.view(-1,1), r, ns, done)
        
        s = ns.numpy()
        cum_r += r
        if done:
            break

    ema.update(cum_r)
    if ep % print_every == 0:
        print("Episode {} || EMA: {} ".format(ep, ema.s))

## REINFORCE와 TD Actor-critic 비교하기
'''
REINFORCE와 TD Actor-critic 알고리즘의 성능을 비교해볼까요? 
실험결과는 [여기](https://app.wandb.ai/junyoung-park/cartpole_exp?workspace=user-junyoung-park)에서 확인할 수 있습니다.
'''
### `TDActorCritic`의 성능이 생각보다 좋지않네요? 왜 그럴까요?
'''
(개인적인 의견으로는, 일단 '서로 다른 두개의 알고리즘을 비교하는 방식이 잘못됐다'고 말씀드리고 싶습니다) <br>

다양한 이유가 있겠지만, 최소한 세가지 요인이 있습니다.

1. 당연하게도, 하이퍼파라미터 튜닝의 필요성
2. Actor-critic 알고리즘 학습의 불안정성
3. 심층신경망을 활용하는 RL기법들의 불안정성

위 문제들은 <파트5 심층강화학습>에서 본격적으로 다루어 봅시다.
'''