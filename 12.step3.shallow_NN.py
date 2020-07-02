import numpy as np
import matplotlib.pyplot as plt

# sigmoid 함수 구현
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# softmax 함수 구현
def softmax(x):
    e_x = np.exp(x)
    return e_x / np.sum(e_x)


# 네트워크 구조 정의 (변수(행렬구조) 생성)
# Input 인자는 각각 input, hidden, output의 뉴런의 개수이다
class ShallowNN:
    # 각 변수들을 __init__ 함수에 정의해준다
    # 데이터를 담을 행렬을 정의한다
    def __init__(self, num_input, num_hidden, num_output):
        self.W_h = np.zeros((num_hidden, num_input), dtype=np.float32)
        self.b_h = np.zeros((num_hidden,), dtype=np.float32)
        self.W_o = np.zeros((num_output, num_hidden), dtype=np.float32)
        self.b_o = np.zeros((num_output,), dtype=np.float32)
    
    # 뉴럴네트워크의 연산 함수작성 y=wx*b
    def __call__(self, x):
        h = sigmoid(np.matmul(self.W_h, x) + self.b_h)
        return softmax(np.matmul(self.W_o, h) + self.b_o)


# 데이터셋 가져오고 정리하기
dataset = np.load('ch2_dataset.npz')
#입력
inputs = dataset['inputs']
#타겟
labels = dataset['labels']

# 모델 만들기
# 준비된 데이터의 입력이 2종류, 타겟이 10종류 이다. 가운데는 히든레이어 수 임
model = ShallowNN(2, 128, 10)

# 사전에 학습된 파라미터 불러오기
# 사전에 학습된 모델이 준비되어 있음
weights = np.load('ch2_parameters.npz')
model.W_h = weights['W_h']
model.b_h = weights['b_h']
model.W_o = weights['W_o']
model.b_o = weights['b_o']


#모델 구동 및 결과 프린트
# output을 받을 리스트 선언
outputs = list()
# inputs는 pt로, labels는 label로 하나씩 for문을 돌면서 들어간다
for pt, label in zip(inputs, labels):
    # pt에 대하여 class ShallowNN의 def __call__(self, x): 를 수행한다
    # 그 결과값은 softmax 계산이 된 형태로 출력이 됨
    output = model(pt)
    # argmax는 결과값중에서 확률이 제일 큰 항에 index를 집어넣는 함수이다
    outputs.append(np.argmax(output))
    #출력과 타겟을 비교해본다
    print(np.argmax(output), label)
# stack을 실행
outputs = np.stack(outputs, axis=0)

# 타겟 데이터 스캐터 플랏
plt.figure()
#총 10 종류의 타겟을 갖고 있음
for idx in range(10):
    mask = labels == idx
    plt.scatter(inputs[mask, 0], inputs[mask, 1])
plt.title('true_label')
plt.show()

# 출력 데이터 스캐터 플랏
plt.figure()
for idx in range(10):
    mask = outputs == idx
    plt.scatter(inputs[mask, 0], inputs[mask, 1])
plt.title('model_output')
plt.show()