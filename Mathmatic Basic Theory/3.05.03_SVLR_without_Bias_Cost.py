# 3.03.01_SVLR_without_Bias_Loss.py와 비교해서 이제 Loss를 넘어서 Cost를 구해서 theta를 업데이트 할거고 그로 인해서
# 이제부터는 벡터 연산을 하게 될것이다
# mini-batch를 전체 데이터셋인 100으로 놓고 계산했을 경우에 대한 코드

# import required modules
import matplotlib.pyplot as plt
import numpy as np

from dataset_generator import dataset_generator
import basic_nodes as basic_nodes

# dataset preparation
dataset_gen = dataset_generator()
dataset_gen.set_coefficient([5,0])
x_data, y_data = dataset_gen.make_dataset()
dataset_gen.dataset_visualizer()

# model implementation
node1 = nodes.mul_node()

# square error loss implementation
node2 = nodes.minus_node()
node3 = nodes.square_node()
node4 = nodes.mean_node()

# hyperparameter setting
epochs = 50 # total epoch setting
lr = 0.05 # learning rate setting

th = -1 # arbitary theta (=weight)
cost_list = []
th_list = []

for epoch in range(epochs):
    X, Y = x_data, y_data
    # X.shape, Y.shape == (100,1) X와 Y의 데이터 개수가 100개이다

    # forward propagation 계산
    Z1 = node1.forward(th, X)
    Z2 = node1.forward(Y, Z1)
    L = node3.forward(Z2)
    J = node4.forward(L)
    # Z1, Z2, Le의 크기는 (100,1)이다
    # J의 크기는 (1)이다
    
    # backward propagation 계산
    dL = node4.backward(1)
    dZ2 = node3.backward(dL)
    dY, dZ1 = node2.backward(dZ2)
    dTh, dX = node1.backward(dZ1)
    # dL, dZ2, dZ1, dTh의 크기는 (100,1)이다

    # theta 업데이트
    th = th - lr*np.sum(dTh)

    th_list.append(th)
    cost_list.append(J)

