#$수치 미분을 이용한 심층 신경망 학습
#학습을 할떄 얼마나 걸리는지 측정하기 위해서 time을 import 한다
import time
import numpy as np

#1. 유틸리티 함수
epsilon = 0.0001

#행렬 Transpose
def _t(x):
    return np.transpose(x)

#행렬 곱
def _m(A, B):
    return np.matmul(A, B)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mean_squared_error(h, y):
    return 1 / 2 * np.mean(np.square(h - y))


#2. 뉴런 구현
class Neuron:
    def __init__(self, W, b, a):
        # Model Parameter를 저장하기 위한 변수 선언
        self.W = W
        self.b = b
        self.a = a
        
        # Gradients를 저장하기 위한 변수 선언
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def __call__(self, x):
        return self.a(_m(_t(self.W), x) + self.b) # activation((W^T)x + b)


#3. 심층신경망 구현
class DNN:
    #hidden_depth=히든레이어 개수, num_neuron=히든레이어 하나당 뉴런의 개수, num_input=입력수, num_output=출력수, activation=sigmoid
    def __init__(self, hidden_depth, num_neuron, num_input, num_output, activation=sigmoid):
        #i = input의 개수, o = output의 개수
        #((0.0),0.01) = 0이 평균이고 0.01이 표준편차인 랜덤 숫자 행렬을 제작
        #np.zeros((o,)) 갯수가 o개인 bias를 0으로 생성
        def init_var(i, o):
            return np.random.normal(0.0, 0.01, (i, o)), np.zeros((o,))

        self.sequence = list()
        # First hidden layer
        W, b = init_var(num_input, num_neuron)
        self.sequence.append(Neuron(W, b, activation))
        
        # Hidden layers
        # 히든레이어는 여러개이기 때문에 for문을 돌린다
        # 첫번째 히든레이어는 위에 있으니까 1개를 빼준다
        for _ in range(hidden_depth - 1):
            W, b = init_var(num_neuron, num_neuron)
            self.sequence.append(Neuron(W, b, activation))

        # Output layer
        W, b = init_var(num_neuron, num_output)
        self.sequence.append(Neuron(W, b, activation))

    #순전파(학습) 구현
    def __call__(self, x):
        for layer in self.sequence:
            x = layer(x)
        return x

    #그라디언트 구현
    def calc_gradient(self, x, y, loss_func):
        #특정 레이어의 뉴런을 교체해주는 기능
        #특정 특정 w나 b를 교체하여 다시 넣어준다
        def get_new_sequence(layer_index, new_neuron):
            new_sequence = list()
            for i, layer in enumerate(self.sequence):
                if i == layer_index:
                    new_sequence.append(new_neuron)
                else:
                    new_sequence.append(layer)
            return new_sequence
        
        # 새 sequence를 평가하는 코드, 다시 순전파(학습) 구현
        def eval_sequence(x, sequence):
            for layer in sequence:
                x = layer(x)
            return x
        
        # self(x)는 self.__init__(x)과 동일하다
        loss = loss_func(self(x), y)
        
        for layer_id, layer in enumerate(self.sequence): # iterate layer
            for w_i, w in enumerate(layer.W): # W의 row 를 iterate
                for w_j, ww in enumerate(w): # W의 col를 iterate
                    W = np.copy(layer.W)
                    W[w_i][w_j] = ww + epsilon
                    
                    new_neuron = Neuron(W, layer.b, layer.a)
                    new_seq = get_new_sequence(layer_id, new_neuron)
                    h = eval_sequence(x, new_seq)
                    
                    num_grad = (loss_func(h, y) - loss) / epsilon  # (f(x+eps) - f(x)) / epsilon
                    layer.dW[w_i][w_j] = num_grad
            
                for b_i, bb in enumerate(layer.b): # iterate bias
                    b = np.copy(layer.b)
                    b[b_i] = bb + epsilon
                    
                    new_neuron = Neuron(layer.W, b, layer.a)
                    new_seq = get_new_sequence(layer_id, new_neuron)
                    h = eval_sequence(x, new_seq)
                    
                    num_grad = (loss_func(h, y) - loss) / epsilon  # (f(x+eps) - f(x)) / epsilon
                    layer.db[b_i] = num_grad
        return loss


#4. 경사하강 학습법 여기에 이거 말고 adam, RMSprop등을 적용해도 된다
def gradient_descent(network, x, y, loss_obj, alpha=0.01):
    loss = network.calc_gradient(x, y, loss_obj)
    for layer in network.sequence:
        layer.W += -alpha * layer.dW
        layer.b += -alpha * layer.db
    return loss



#5. 동작테스트
x = np.random.normal(0.0, 1.0, (10,))
y = np.random.normal(0.0, 1.0, (2,))

dnn = DNN(hidden_depth=5, num_neuron=32, num_input=10, num_output=2, activation=sigmoid)

t = time.time()
for epoch in range(100):
    loss = gradient_descent(dnn, x, y, mean_squared_error, 0.01)
    print('Epoch {}: Test loss {}'.format(epoch, loss))
print('{} seconds elapsed.'.format(time.time() - t))


