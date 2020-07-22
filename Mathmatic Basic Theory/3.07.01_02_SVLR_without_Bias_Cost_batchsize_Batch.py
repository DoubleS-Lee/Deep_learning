# 3.03.01_SVLR_without_Bias_Cost.py와 비교해서 이제 Batch size에 대한 학습을 시킬것이다
# Batch Gradient Descent
# batch_size = data_size
# epochs 수가 곧 iterations 수이다

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

# model part
node1 = nodes.mul_node()

# square error loss/MSE cost part
node2 = nodes.minus_node()
node3 = nodes.square_node()
node4 = nodes.mean_node()

# hyperparameter setting
epochs = 200 # total epoch setting
lr = 0.01 # learning rate setting

th = -1 # arbitary theta (=weight)
cost_list = []
th_list = []

for _ in range(epochs):
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

fig, ax = plt.subplots(2, 1, figsize = (42,20))
ax[0].plot(th_list, linewidth = 5)
ax[1].plot(cost list, linewidth=5)