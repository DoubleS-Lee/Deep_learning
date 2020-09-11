# Perceptron 구현 (OR 함수 구현)

## Import modules

import tensorflow as tf
import numpy as np

## Perceptron 구현

class Perceptron:
    def __init__(self, w, b):
        self.w = tf.Variable(w, dtype=tf.float32)
        self.b = tf.Variable(b, dtype=tf.float32)
    
    def __call__(self, x):
        return tf.sign(tf.reduce_sum(self.w * x) + self.b)

## Utility 함수 구현

def v(*args):
    return np.array(args)

## Perceptron 정의

w = v(1, 1)
b = 0.5

perceptron = Perceptron(w, b)

## Perceptron 동작 확인

p1 = perceptron(v(1, 1)) # T, T
p2 = perceptron(v(-1, 1)) # F, T
p3 = perceptron(v(-1, -1)) # F, F
p4 = perceptron(v(1, -1)) # T, F

print(p2.numpy(), p1.numpy())
print(p3.numpy(), p4.numpy())


#%%

# Perceptron 구현 (XOR 함수 구현)

## Import modules

import tensorflow as tf
import numpy as np

## Perceptron 구현

class Perceptron:  # Implementation of a perceptron with a bias
    def __init__(self, w, b):
        self.w = tf.Variable(w, dtype=tf.float32)
        self.b = tf.Variable(b, dtype=tf.float32)

    def __call__(self, x):
        return tf.sign(tf.reduce_sum(self.w * x) + self.b)

## Utility 함수 구현

def v(*args):  # Enhance code readability
    return np.array(args)

## Perceptron 정의

p_nand = Perceptron(w=v(-1, -1),
                    b=0.5)

p_or = Perceptron(w=v(1, 1),
                    b=0.5)

p_and = Perceptron(w=v(1, 1),
                    b=-0.5)

def xor(x):
    h1 = p_nand(x)
    h2 = p_or(x)
    return p_and(v(h1, h2))

## Perceptron 동작 확인

p1 = xor(v(1, 1)) # T, T
p2 = xor(v(-1, 1)) # F, T
p3 = xor(v(-1, -1)) # F, F
p4 = xor(v(1, -1)) # T, F

print(p2.numpy(), p1.numpy())
print(p3.numpy(), p4.numpy())