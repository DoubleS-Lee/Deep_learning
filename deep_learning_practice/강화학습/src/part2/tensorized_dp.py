import numpy as np


class TensorDP:

    def __init__(self,
                 gamma=1.0,
                 error_tol=1e-5):
        self.gamma = gamma
        self.error_tol = error_tol

        # Following attributes will be set after call "set_env()"

        self.env = None  # environment
        self.policy = None  # policy
        self.ns = None  # Num. states
        self.na = None  # Num. actions
        self.P = None  # Transition tensor
        self.R = None  # Reward tensor

    def set_env(self, env, policy=None):
        self.env = env
        # 초기 policy 값들 계산
        if policy is None:
            self.policy = np.ones([env.nS, env.nA]) / env.nA

        self.ns = env.nS
        self.na = env.nA
        self.P = env.P_tensor  # Rank 3 tensor [num. actions x num. states x num. states]
        self.R = env.R_tensor  # Rank 2 tensor [num. actions x num. states]

        print("Tensor DP agent initialized")
        print("Environment spec:  Num. state = {} | Num. actions = {} ".format(env.nS, env.nA))

    def reset_policy(self):
        self.policy = np.ones([self.ns, self.na]) / self.na

    def set_policy(self, policy):
        assert self.policy.shape == policy.shape
        self.policy = policy
        
    # R{pi}
    def get_r_pi(self, policy):
        r_pi = (policy * self.R).sum(axis=-1)  # [num. states x 1]
        return r_pi
    
    # P{pi}
    def get_p_pi(self, policy):
        # einsum(아인슈타인 표기법) einsum("첨자1,첨자2,첨자3->첨자", 텐서1,텐서2,텐서3)
        # 2차원행렬(policy)와 3차원텐서(self.P)를 아인슈타인으로 계산하여 차원을 2차원 행렬로 바꿔줌
        # ex) na,anm->nm [(25x4,4x25x25)->(25x25)]
        p_pi = np.einsum("na,anm->nm", policy, self.P)  # [num. states x num. states]
        return p_pi

    def policy_evaluation(self, policy=None, v_init=None):
        """
        :param policy: policy to evaluate (optional)
        :param v_init: initial value 'guesstimation' (optional)
        :param steps: steps of bellman expectation backup (optional)
        if none, repeat the backup until converge.

        :return: v_pi: value function of the input policy
        """
        if policy is None:
            policy = self.policy

        r_pi = self.get_r_pi(policy)  # [num. states x 1]
        p_pi = self.get_p_pi(policy)  # [num. states x num. states]

        if v_init is None:
            v_old = np.zeros(self.ns)
        else:
            v_old = v_init

        while True:
            # perform bellman expectation back
            # 이 v_new는 T{pi}(V)이며 이게 수렴할때까지 계산을 반복함
            v_new = r_pi + self.gamma * np.matmul(p_pi, v_old)

            # 수렴 여부 확인
            bellman_error = np.linalg.norm(v_new - v_old)
            if bellman_error <= self.error_tol:
                break
            else:
                v_old = v_new

        return v_new

    def policy_improvement(self, policy=None, v_pi=None):
        if policy is None:
            policy = self.policy

        if v_pi is None:
            v_pi = self.policy_evaluation(policy)

        # (1) Compute Q_pi(s,a) from V_pi(s)
        r_pi = self.get_r_pi(policy)
        q_pi = r_pi + self.P.dot(v_pi)  # q_pi = [num.action x num states]

        # (2) Greedy improvement
        ## 개선될 정채 pi'은 특정 s에 대해 가장 큰 Q(s,a)를 만족하는 a 이외에는 값이 0이니까 0으로 초기화
        policy_improved = np.zeros_like(policy)
        ## 특정 s에 대해 가장 큰 Q(s,a)를 만족하는 a 만을 1.0으로 설정
        policy_improved[np.arange(q_pi.shape[1]), q_pi.argmax(axis=0)] = 1
        return policy_improved

    def policy_iteration(self, policy=None):
        if policy is None:
            pi_old = self.policy
        else:
            pi_old = policy

        info = dict()
        info['v'] = list()
        info['pi'] = list()
        info['converge'] = None

        steps = 0
        converged = False
        while True:
            v_old = self.policy_evaluation(pi_old)
            pi_improved = self.policy_improvement(pi_old, v_old)
            steps += 1

            info['v'].append(v_old)
            info['pi'].append(pi_old)

            # check convergence
            # 두 정책 사이의 거리를 측정
            policy_gap = np.linalg.norm(pi_improved - pi_old)
            
            # 두 정책의 gap이 우리가 설정한 값보다 작으면 탈출, 크면 다시 while문 반복
            if policy_gap <= self.error_tol:
                if not converged:  # record the first moment of within error tolerance.
                    info['converge'] = steps
                break
            else:
                pi_old = pi_improved
        return info

    def value_iteration(self, v_init=None, compute_pi=False):
        # 여기는 policy를 input으로 받지 않는다
        # 여기서 compute_pi를 True로 하면 밑에 if 문에 의해 우리가 알고있는 V값으로 Q를 계산하고 이 Q로 policy를 추산하는 계산도 추가로 한다
        """
        :param v_init: (np.array) initial value 'guesstimation' (optional)
        :param compute_pi: (bool) compute policy during VI
        :return: v_opt: the optimal value function
        """

        if v_init is not None:
            v_old = v_init
        else:
            v_old = np.zeros(self.ns)

        info = dict()
        info['v'] = list()
        info['pi'] = list()
        info['converge'] = None

        steps = 0
        converged = False

        while True:
            # Bellman optimality backup
            # self.R.T : 원래 R함수는 (state,action) 행렬로 되어 있는데 여기서는 계산의 편의를 위해서 Transpose를 시켰다
            # V{k+1} = max{모든a}(R{a} + (gamma)*P{a}*V{k+1})
            v_improved = (self.R.T + self.gamma * self.P.dot(v_old)).max(axis=0)
            info['v'].append(v_improved)

            if compute_pi:
                # compute policy from v
                # 1) Compute v -> q
                q_pi = (self.R.T + self.gamma * self.P.dot(v_improved))

                # 2) Construct greedy policy
                pi = np.zeros_like(self.policy)
                pi[np.arange(q_pi.shape[1]), q_pi.argmax(axis=0)] = 1
                info['pi'].append(pi)

            steps += 1

            # check convergence
            policy_gap = np.linalg.norm(v_improved - v_old)

            if policy_gap <= self.error_tol:
                if not converged:  # record the first moment of within error tolerance.
                    info['converge'] = steps
                break
            else:
                v_old = v_improved
        return info
