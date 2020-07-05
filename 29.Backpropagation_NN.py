#$ 역전파 학습법을 이용한 심층 신경망 학습
#학습을 할떄 얼마나 걸리는지 측정하기 위해서 time을 import 한다
import time
import numpy as np

#1. 유틸리티 함수
def _t(x):
    return np.transpose(x)

def _m(A, B):
    return np.matmul(A, B)


#2. Sigmoid 구현
class Sigmoid:
    def __init__(self):
        #last_0 = 마지막 출력
        self.last_o = 1

    def __call__(self, x):
        #시그모이드 함수를 입력한다
        self.last_o = 1 / (1.0 + np.exp(-x))
        return self.last_o

    #시그모이드 함수의 gradient 구현
    def grad(self): # sigmoid(x)(1-sigmoid(x))
        #시그모이드 함수의 미분값을 입력한다
        return self.last_o * (1 - self.last_o)


#3. Mean Squared Error 구현
class MeanSquaredError:
    def __init__(self):
        # gradient
        self.dh = 1
        self.last_diff = 1        

    def __call__(self, h, y):
        # MSE 구현 = 1/2 * mean ((h - y)^2)
        self.last_diff = h - y
        return 1 / 2 * np.mean(np.square(h - y))

    #MSE의 gradient 구현
    def grad(self):
        #MSE의 미분값 구현 = h - y
        return self.last_diff



#4. 뉴런 구현
class Neuron:
    def __init__(self, W, b, a_obj):
        # Model parameters 생성
        self.W = W
        self.b = b
        self.a = a_obj()
        
        # gradient 생성
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.dh = np.zeros_like(_t(self.W))
        
        #마지막 x와 h를 저장해주기 위해서 last_x, last_y 생성
        self.last_x = np.zeros((self.W.shape[0]))
        self.last_h = np.zeros((self.W.shape[1]))

    #순전파 구현
    def __call__(self, x):
        self.last_x = x
        self.last_h = _m(_t(self.W), x) + self.b
        return self.a(self.last_h)

    #gradient 계산 구현
    def grad(self):
        # dy/dh = W
        return self.W * self.a.grad()

    #w의 gradient 구현
    #dh = 지금까지 gradient에서 누적되어 계산되어 오는 값
    def grad_W(self, dh):
        grad = np.ones_like(self.W)
        grad_a = self.a.grad()
        for j in range(grad.shape[1]):
            # y = w^T*x + b
            # dy/dw = x
            grad[:, j] = dh[j] * grad_a[j] * self.last_x
        return grad
    
    #bias의 gradient 구현
    def grad_b(self, dh):
        # dy/dh = 1
        return dh * self.a.grad()*1



#5. 심층신경망 구현
class DNN:
    def __init__(self, hidden_depth, num_neuron, input, output, activation=Sigmoid):
        def init_var(i, o):
            return np.random.normal(0.0, 0.01, (i, o)), np.zeros((o,))

        self.sequence = list()
        # First hidden layer
        W, b = init_var(input, num_neuron)
        self.sequence.append(Neuron(W, b, activation))

        # Hidden layers
        # 히든레이어는 여러개이기 때문에 for문을 돌린다
        for index in range(hidden_depth):
            W, b = init_var(num_neuron, num_neuron)
            self.sequence.append(Neuron(W, b, activation))

        # Output Layer
        W, b = init_var(num_neuron, output)
        self.sequence.append(Neuron(W, b, activation))

    def __call__(self, x):
        for layer in self.sequence:
            x = layer(x)
        return x

    #전체 미분을 통합하는 과정
    def calc_gradient(self, loss_obj):
        loss_obj.dh = loss_obj.grad()
        self.sequence.append(loss_obj)
        
        # back-prop loop
        for i in range(len(self.sequence) - 1, 0, -1):
            l1 = self.sequence[i]
            l0 = self.sequence[i - 1]
            
            l0.dh = _m(l0.grad(), l1.dh)
            l0.dW = l0.grad_W(l1.dh)
            l0.db = l0.grad_b(l1.dh)
        
        self.sequence.remove(loss_obj)




#6. 경사하강 학습법
def gradient_descent(network, x, y, loss_obj, alpha=0.01):
    loss = loss_obj(network(x), y)  # Forward inference
    network.calc_gradient(loss_obj)  # Back-propagation
    for layer in network.sequence:
        layer.W += -alpha * layer.dW
        layer.b += -alpha * layer.db
    return loss


#7. 동작 테스트
x = np.random.normal(0.0, 1.0, (10,))
y = np.random.normal(0.0, 1.0, (2,))

t = time.time()
dnn = DNN(hidden_depth=5, num_neuron=32, input=10, output=2, activation=Sigmoid)
loss_obj = MeanSquaredError()
for epoch in range(100):
    loss = gradient_descent(dnn, x, y, loss_obj, alpha=0.01)
    print('Epoch {}: Test loss {}'.format(epoch, loss))
print('{} seconds elapsed.'.format(time.time() - t))









