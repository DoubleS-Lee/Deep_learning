GAN
Generative Adversarial Network



generator : 생성자
discriminator : 감별사

생성자가 아무것도 모르고 데이터 distribution를 만들면 감별사가 틀린 경우 계속 틀렸다고 회신을 주면서
생성자가 맞을때까지 학습을 시킨다

DC GAN

설계도면 생성을 GAN으로 할 수 있지 않을까?

휠 디자인을 GAN으로 만들었음

distribution을 만드는 것






#%%
강화학습 (최적화) 
비슷한거로 GA가 있다 GA는 Sequential 하지 않아도 된다
강화학습은 Sequential 해야한다
Reinforcement Learning
에이전트가 주어진 환경(state)에 대해 어떤 행동(action)을 취하고 이로부터 어떤 보상(reward)을 얻으면서 학습을 하는 방법

환경이 좋은 쪽으로 갈지, 보상이 좋은쪽으로 갈지 결정해야함

배우지 않았지만 직접 시도하면서 행동과 그 결과로 나타나는 보상 사이의 상관관계를 학습하는 것
 - 보상을 많이 받는 행동의 확률을 높이기

policy(정책)
 - 우리가 강화학습을 통해서 근본적으로 얻을 수 있는 것

Optimal Policy(최적 정책)
 - 우리가 찾고자하는 최고의 정책

연속적인 행동(sequential)

에어서스펜션의 성능을 시시각각 다르게 적용해야하는 경우

이 행동이 좋은지 안 좋은지 그 결과는 맨 끝에 가봐야 알 수 있다는 단점이 있다

마코프 프로세스 가정을 사용
- 현재 오늘 날씨가 맑을때 30일 뒤의 날씨가 어떨지

강화학습에서는 Reward가 무조건 처음부터 정해져 있어야 학습을 할 수 있다


에이전트 : 학습을 통해서 어떤 행동을 결정하는 것(액션을 취한다)
환경 : 에이전트를 제외한 나머지 (에이전트에게 피드백을 준다)

강화학습이 풀어야하는 문제 : 시간적인 문제, 지속적인 결정을 연속으로 해야하는 것. 실시간으로 무엇을 할지 결정해야한다 (Sequential Decision Problem)

* 마르코프 결정과정(Markov Decision Process)
문제에 대한 수학적 정의 : Markov Decision Process (가장 reward가 큰 것을 골라주겠다)
 Markov Property : 시스템의 t+1의 출력은 t의 출력에만 영향을 받는다(?)
 오늘의 사건을 기반으로 미래의 일을 예측할 수 있게 해주는 것
 내 현재의 행동이 미래에 어떻게 진행될지 예측할 수 있다
ex)감염병 모델
 Markov reward
 
몬테 카를로(Monte-Carlo)기법
 - 기댓값을 근사적으로 계산하는 방법
 - 전체 확률을 모르더라도 충분히 많은 시행을 한 결과를 가지고 있으면 기댓값을 구할 수 있다.

조건부 기대값(conditional expectation)
 - 에이전트의 '가치함수' 를 정의할때 사용

상태 가치함수 : 상태가 입력으로 들어오면 그 상태에서 앞으로 받을 보상의 합을 출력하는 함수 (벨만 기대값)
행동 가치함수 : 어떤 상태에서 각 행동에 대해 따로 가치함수를 만들어서 어떤 행동이 얼마나 좋은지 알려주는 함수 (Q Function)

DP는 가치함수를 계산 (Dynamic Programming)
 - policy iteration(정책 이터레이션) : 다이나믹 프로그래밍으로 벨만 기대 방정식을 이용하여 순차적인 행동 결정 문제를 푸는 것
 - value iteration(가치 이터레이션) : 다이나믹 프로그래밍으로 벨만 최적 방정식을 이용하여 순차적인 행동 결정 문제를 푸는 것
RL은 가치함수를 계산하지 않고 sampling을 통한 approximation (Reinforcement Learning)

정책을 고려한 벨만 기대방정식 : 현재 State에서 할 수 있는 Action의 점수를 매겨서 가장 점수가 높은 Action을 고르자

sparse reward
delayed reward

1. Dynamic Programming

2. Monte Carlo Prediction
 - 실시간 예측이 불가능. 시나리오가 최종 끝까지 가야 가중치가 업데이트 가능하다

3. Temporal difference Prediction
 - 매 타임스텝마다 가치함수를 업데이트

SALSA

Q-Learning
 - SALSA의 문제점을 해결하기 위해 나옴
 - 




























