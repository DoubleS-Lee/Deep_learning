import numpy as np


class ExactMCAgent:
    """
    The exact Monte-Carlo agent.
    This agents performs value update as follows:
    V(s) <- s(s) / n(s)
    Q(s,a) <- s(s,a) / n(s,a)
    
    1. `gamma` : 감가율
    2. `num_states` : 상태공간의 크기 (서로 다른 상태의 갯수)
    3. `num_actions` : 행동공간의 크기 (서로 다른 행동의 갯수)
    4. `epsilon`: $\epsilon$-탐욕적 정책의 파라미터
    """

    def __init__(self,
                 gamma: float,
                 num_states: int,
                 num_actions: int,
                 epsilon: float):
        self.gamma = gamma
        self.num_states = num_states
        self.num_actions = num_actions
        self.epsilon = epsilon
        
        # e-greedy에 사용되는 값은 아니고 G(s)를 N(s)로 나눠줄때 N(s)가 0에 가까울 경우 계산상의 오류를 방지하기 위해 더해주는 값
        # _는 보통 외부에서 사용되지 않고 내부에서만 사용되는 변수에 대해 붙여주는 경향이 있다
        self._eps = 1e-10  # for stable computation of V and Q. NOT the one for e-greedy !

        # Initialize statistics
        self.n_v = None
        self.s_v = None
        self.n_q = None
        self.s_q = None
        self.reset_statistics()

        # Initialize state value function V and action value function Q
        self.v = None
        self.q = None
        self.reset_values()

        # Initialize "policy Q"
        # "policy Q" is the one used for policy generation.
        self._policy_q = None
        self.reset_policy()

    def reset_statistics(self):
        self.n_v = np.zeros(shape=self.num_states)
        self.s_v = np.zeros(shape=self.num_states)

        self.n_q = np.zeros(shape=(self.num_states, self.num_actions))
        self.s_q = np.zeros(shape=(self.num_states, self.num_actions))

    def reset_values(self):
        self.v = np.zeros(shape=self.num_states)
        self.q = np.zeros(shape=(self.num_states, self.num_actions))

    def reset_policy(self):
        self._policy_q = np.zeros(shape=(self.num_states, self.num_actions))

    def reset(self):
        self.reset_statistics()
        self.reset_values()
        self.reset_policy()

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

            self.n_v[s] += 1
            self.n_q[s, a] += 1
            
            # 리턴(G)을 s라고 표기함
            self.s_v[s] += cum_r
            self.s_q[s, a] += cum_r

    def compute_values(self):
        # self.n_v와 self.n_q가 0에 수렴할 수도 있기때문에 아주 작은 값인 _eps를 더해준다
        self.v = self.s_v / (self.n_v + self._eps)
        self.q = self.s_q / (self.n_q + self._eps)

    def get_action(self, state):
        prob = np.random.uniform(0.0, 1.0, 1)
        # e-greedy policy over Q
        if prob <= self.epsilon:  # random
            action = np.random.choice(range(self.num_actions))
        else:  # greedy
            action = self._policy_q[state, :].argmax()
        return action

    def improve_policy(self):
        self._policy_q = self.q.copy()
        #self.reset_statistics()

    def decaying_epsilon(self, factor):
        self.epsilon *= factor


class MCAgent(ExactMCAgent):
    """
    The 'learning-rate' Monte-Carlo agent.
    This agents performs value update as follows:
    V(s) <- V(s) + lr * (Gt - V(s))
    Q(s,a) <- Q(s,a) + lr * (Gt - Q(s,a))
    """

    def __init__(self,
                 gamma: float,
                 num_states: int,
                 num_actions: int,
                 epsilon: float,
                 lr: float):
        super(MCAgent, self).__init__(gamma=gamma,
                                      num_states=num_states,
                                      num_actions=num_actions,
                                      epsilon=epsilon)
        self.lr = lr

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


def run_episode(env, agent, timeout=1000):
    env.reset()
    states = []
    actions = []
    rewards = []

    i = 0
    timeouted = False
    while True:
        state = env.s
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)

        if done:
            break
        else:
            i += 1
            if i >= timeout:
                timeouted = True
                break

    if not timeouted:
        episode = (states, actions, rewards)
        agent.update(episode)


if __name__ == '__main__':
    from envs.gridworld import GridworldEnv

    nx, ny = 5, 5
    env = GridworldEnv([ny, nx])

    mc_agent = MCAgent(gamma=1.0,
                       lr=1e-3,
                       num_states=nx * ny,
                       num_actions=4,
                       epsilon=1.0)

    run_episode(env, mc_agent)
