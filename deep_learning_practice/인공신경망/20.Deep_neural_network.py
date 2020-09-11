## 수치 미분을 이용한 심층 신경망 학습

## Import modules

import time
import numpy as np

## 유틸리티 함수

epsilon = 0.0001

def _t(x):
    return np.transpose(x)

def _m(A, B):
    return np.matmul(A, B)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mean_squared_error(h, y):
    return 1 / 2 * np.mean(np.square(h - y))

## Dense Layer 구현

class Dense:
    def __init__(self, W, b, a):
        self.W = W
        self.b = b
        self.a = a
        
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def __call__(self, x):
        return self.a(_m(_t(self.W), x) + self.b)   # matmul((ixo)T,ix1) + ox1

## 심층신경망 구현

class DNN:
    def __init__(self, hidden_depth, num_neuron, num_input, num_output, activation=sigmoid):
        def init_var(i, o):
            return np.random.normal(0.0, 0.01, (i, o)), np.zeros((o,))

        self.sequence = list()
        # First hidden layer
        W, b = init_var(num_input, num_neuron)
        self.sequence.append(Dense(W, b, activation))
        
        # Hidden layers
        for _ in range(hidden_depth - 1):
            W, b = init_var(num_neuron, num_neuron)
            self.sequence.append(Dense(W, b, activation))

        # Output layer
        W, b = init_var(num_neuron, num_output)
        self.sequence.append(Dense(W, b, activation))

    def __call__(self, x):
        for layer in self.sequence:
            x = layer(x)
        return x

    def calc_gradient(self, x, y, loss_func):
        def get_new_sequence(layer_index, new_layer):
            new_sequence = list()
            for i, layer in enumerate(self.sequence):
                if i == layer_index:
                    new_sequence.append(new_layer)
                else:
                    new_sequence.append(layer)
            return new_sequence
        
        def eval_sequence(x, sequence):
            for layer in sequence:
                x = layer(x)
            return x
        
        loss = loss_func(self(x), y)
        
        for layer_id, layer in enumerate(self.sequence):
            for w_i, w in enumerate(layer.W):
                for w_j, ww in enumerate(w):
                    W = np.copy(layer.W)
                    W[w_i][w_j] = ww + epsilon
                    
                    new_layer = Dense(W, layer.b, layer.a)
                    new_seq = get_new_sequence(layer_id, new_layer)
                    h = eval_sequence(x, new_seq)
                    
                    num_grad = (loss_func(h, y) - loss) / epsilon # (f(x+eps) - f(x)) / eps
                    layer.dW[w_i][w_j] = num_grad
                    
            for b_i, bb in enumerate(layer.b):
                b = np.copy(layer.b)
                b[b_i] = bb + epsilon

                new_layer = Dense(layer.W, b, layer.a)
                new_seq = get_new_sequence(layer_id, new_layer)
                h = eval_sequence(x, new_seq)

                num_grad = (loss_func(h, y) - loss) / epsilon # (f(x+eps) - f(x)) / eps
                layer.db[b_i] = num_grad
        
        return loss


## 경사하강 학습법

def gradient_descent(network, x, y, loss_obj, alpha=0.01):
    loss = network.calc_gradient(x, y, loss_obj)
    for layer in network.sequence:
        layer.W += -alpha * layer.dW
        layer.b += -alpha * layer.db
    return loss

## 동작 테스트

x = np.random.normal(0.0, 1.0, (10,))
y = np.random.normal(0.0, 1.0, (2,))

dnn = DNN(hidden_depth=5, num_neuron=32, num_input=10, num_output=2, activation=sigmoid)

t = time.time()
for epoch in range(100):
    loss = gradient_descent(dnn, x, y, mean_squared_error, 0.01)
    print('Epoch {}: Test loss {}'.format(epoch, loss))
print('{} seconds elapsed.'.format(time.time() - t))