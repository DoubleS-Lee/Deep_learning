# 3.03.01_SVLR_without_Bias_Cost.py와 비교해서 이제 Batch size에 대한 학습을 시킬것이다
# Mini-batch Gradient Descent 에서 Shuffle을 적용하고 Replacement는 적용하지 않은 모델
# train data를 셔플하여 학습에 적용시킨다
# batch_size = n

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
lr = 0.01 # learning rate setting

th = -1 # arbitary theta (=weight)
cost_list = []
th_list = []

batch_size = 16
n_batch = int(np.ceil(len(x_data)/batch_size))

# target_iteration
t_iteration = 500
# 500번의 iteration을 하도록 일부러 epoch 횟수를 아래와 같이 세팅해놨음
epochs = np.ceil(t_iteration/n_batch).astype(int)

for epoch in range(epochs):
    # random shuffle
    idx_np = np.arrange(len(x_data))
    np.random.shuffle(idx_np)
    x_data = x_data[idx_np]
    y_data = y_data[idx_np]

    # batch 사이즈로 총 데이터 개수를 나눴을때 맨 마지막에 좀 남은 데이터를 처리하는 코드를 만들어줌
    for batch_idx in range(n_batch):
        if batch_idx is n_batch-1:
            X = x_data[batch_idx*batch_size :]
            Y = y_data[batch_idx*batch_size :]
        else:
            X = x_data[batch_idx*batch_size : (batch_idx + 1)*batch_size]
            Y = y_data[batch_idx*batch_size : (batch_idx + 1)*batch_size]

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