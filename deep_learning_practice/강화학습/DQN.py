import sys; sys.path.append('..') # add project root to the python path

import torch
import gym

from src.part3.MLP import MultiLayerPerceptron as MLP
from src.part5.DQN import DQN, prepare_training_inputs
from src.common.memory.memory import ReplayMemory
from src.common.train_utils import to_tensor

## Deep Q-network (DQN)
'''
DQN의 전체적인 트레이닝 과정을 도식화 하면 다음과 같습니다.

<img src="./images/dqn_overview.png" width="40%" height="40%" title="DQN" alt="DQN"></img>

DQN의 Loss 함수는 다음과 같습니다.

$$\theta \leftarrow \theta + \eta \frac{\partial \frac{1}{m}\sum_{i=1}^{m}\mathcal{L}(s_i, a_i, r_i, s_i^{'})}{\partial \theta}$$

$$ \mathcal{L}(s_i, a_i, r_i, s_i^{'}) = |r_i+\gamma \max_{a'} Q_\theta(s_i^{'},a')-Q_\theta(s_i, a_i)|_2$$

$$(s_i, a_i, r_i, s_i^{'}) \sim \mathcal{D}$$

자 그럼 python 으로는 DQN을 어떻게 구현할 수 있을까요?
```python
class DQN(nn.Module):

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 qnet: nn.Module,
                 qnet_target: nn.Module,
                 lr: float,
                 gamma: float,
                 epsilon: float):
        """
        :param state_dim: input state dimension
        :param action_dim: action dimension
        :param qnet: main q network
        :param qnet_target: target q network
        :param lr: learning rate
        :param gamma: discount factor of MDP
        :param epsilon: E-greedy factor
        """

        super(DQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.qnet = qnet
        self.lr = lr
        self.gamma = gamma
        self.opt = torch.optim.Adam(params=self.qnet.parameters(), lr=lr)
        self.register_buffer('epsilon', torch.ones(1) * epsilon)

        # target network related
        self.qnet_target = qnet_target
        self.criteria = nn.SmoothL1Loss()

    def get_action(self, state):
        qs = self.qnet(state)
        prob = np.random.uniform(0.0, 1.0, 1)
        if torch.from_numpy(prob).float() <= self.epsilon:  # random
            action = np.random.choice(range(self.action_dim))
        else:  # greedy
            action = qs.argmax(dim=-1)
        return int(action)

    def update(self, state, action, reward, next_state, done):
        s, a, r, ns = state, action, reward, next_state

        # compute Q-Learning target with 'target network'
        with torch.no_grad():
            q_max, _ = self.qnet_target(ns).max(dim=-1, keepdims=True)
            q_target = r + self.gamma * q_max * (1 - done)

        q_val = self.qnet(s).gather(1, a)
        loss = self.criteria(q_val, q_target)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
```
'''

## `nn.SmoothL1Loss()` 가 뭐지?
'''
Mean-squared Error (MSE) Loss 단점 중 하나는 데이터의 outlier에 매우 취약하다는 것입니다.
모종의 이유로 타겟하는 레이블 y (우리의 경우는 td-target/q-learning target 이죠)이 noisy 할때를 가정하면, 잘못된 y 값을 맞추기 위해 파라미터들이 너무 sensitive 하게 움직이게 됩니다.

이런 현상은 q-learning 의 학습초기에 매우 빈번해 나타날 것으로 예상할수 있습니다. 이러한 문제를 조금이라도 완화하기 위해서 outlier에 덜 민감한 loss 함수를 사용했습니다.
'''
### SmoothL1Loss (aka Huber loss)
'''
$$loss(x,y) = \frac{1}{n}\sum_i z_i$$
$|x_i - y_i| <1$ 일때,
$$z_i = 0.5(x_i - y_i)^2$$
$|x_i - y_i| \geq1$ 일때,
$$z_i = |x_i - y_i|-0.5$$

더욱 자세한 설명은 [여기](https://pytorch.org/docs/master/generated/torch.nn.SmoothL1Loss.html)를 참조해주세요.

그럼 DQN논문에서 제안한 두 가지의 중요한 기법을 구현하는 방법에 대해서 알아봅시다. 그 두가지 기법은 다음과 같습니다.

> 1. Target network 를 활용한 'Moving Target problem' 완화 <br>
> 2. Sample 간의 시간적 연관관계를 줄이고, 한번에 더 많은 샘플을 학습할 수 있게 만든 Exeperience Replay 
'''
## Target network 구현
'''
Target network는 main network와 동일한 구조 및 파라미터를 가지는 네트워트 였죠? `pytorch`에서는 과연 그럼 어떻게 target network를 구현할까요? <br>
답은 간단합니다. Main network의 `state_dict`를 target network의 `state_dict`에 덮어쓰면 되겠죠?

```python
qnet_target.load_state_dict(qnet.state_dict())
```

## Experience Replay 구현

Experience Replay는 간단하게 생각하면
> (1) 기존의 transition sample 들을 저장하고 <br>
> (2) 필요할 때, 저장된 샘플중에서 일부를 sampling 해서 돌려주는 장치입니다.

이를 파이썬으로 구현하면 다음과 같이 구현할 수 있습니다.

```python
from random import sample


class ReplayMemory:
    def __init__(self, max_size):
        # deque object that we've used for 'episodic_memory' is not suitable for random sampling
        # here, we instead use a fix-size array to implement 'buffer'
        self.buffer = [None] * max_size
        self.max_size = max_size
        self.index = 0
        self.size = 0

    def push(self, obj):
        self.buffer[self.index] = obj
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size):
        indices = sample(range(self.size), batch_size)
        return [self.buffer[index] for index in indices]

    def __len__(self):
        return self.size
```
'''
## DQN 으로 `Cartpole-v1` 에서 학습하기
'''
앞에서 설명한 것들을 하나로 묶어서 DQN의 학습 과정을 구현해봅시다.
'''
### Hyperparameter
'''
(심층) 강화학습 알고리즘에서 성능에 지대한 영향을 미치는 하이퍼파라미터들입니다.
이 실습에 쓰인 하이퍼 파라미터는 https://github.com/seungeunrho/minimalRL/blob/master/dqn.py 에서 제안된 값들을 사용하였습니다.
'''
lr = 1e-4 * 5
batch_size = 256
gamma = 1.0
memory_size = 50000
total_eps = 3000
eps_max = 0.08
eps_min = 0.01
sampling_only_until = 2000
target_update_interval = 10

qnet = MLP(4, 2, num_neurons=[128])
qnet_target = MLP(4, 2, num_neurons=[128])

# initialize target network same as the main network.
qnet_target.load_state_dict(qnet.state_dict())
agent = DQN(4, 1, qnet=qnet, qnet_target=qnet_target, lr=lr, gamma=gamma, epsilon=1.0)
env = gym.make('CartPole-v1')
memory = ReplayMemory(memory_size)

print_every = 100

for n_epi in range(total_eps):
    # epsilon scheduling
    # slowly decaying_epsilon
    epsilon = max(eps_min, eps_max - eps_min * (n_epi / 200))
    agent.epsilon = torch.tensor(epsilon)
    s = env.reset()
    cum_r = 0

    while True:
        s = to_tensor(s, size=(1, 4))
        a = agent.get_action(s)
        ns, r, done, info = env.step(a)

        experience = (s,
                      torch.tensor(a).view(1, 1),
                      torch.tensor(r / 100.0).view(1, 1),
                      torch.tensor(ns).view(1, 4),
                      torch.tensor(done).view(1, 1))
        memory.push(experience)

        s = ns
        cum_r += r
        if done:
            break

    if len(memory) >= sampling_only_until:
        # train agent
        sampled_exps = memory.sample(batch_size)
        sampled_exps = prepare_training_inputs(sampled_exps)
        agent.update(*sampled_exps)

    if n_epi % target_update_interval == 0:
        qnet_target.load_state_dict(qnet.state_dict())
    
    if n_epi % print_every == 0:
        msg = (n_epi, cum_r, epsilon)
        print("Episode : {:4.0f} | Cumulative Reward : {:4.0f} | Epsilon : {:.3f}".format(*msg))

## Target update interval 에 따른 효과를 비교해봅시다.
'''
이전과 같이 wandb에서 결과를 확인해볼까요? 이 [링크](https://app.wandb.ai/junyoung-park/DQN?workspace=user-junyoung-park) 를 참조해주세요.
'''