import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import basic_nodes as nodes
from dataset_generator import dataset_generator

plt.style.use('seaborn')
np.random.seed(0)

# parameter setting
t_th1, t_th0 = 5,5
th1, th0 = 1,1
lr=0.01
epochs = 20
batch_size = 8

dataset_gen = dataset_generator()

dataset_gen.set_coefficient([t_th1, t_th0])
x_data, y_data = dataset_gen.make_dataset()
data = np.hstack((x_data, y_data))
n_batch = int(data.shape[0]/batch_size)

# model implementation
node1 = nodes.mul_node()
node2 = nodes.plus_node()

# square loss/MSE cost implementation
node3 = nodes.minus_node()
node4 = nodes.square_node()
node5 = nodes.mean_node()

th1_list, th0_list = [], []
cost_list = []

for epoch in range(epochs):
    np.random.shuffle(data)
    for batch_idx in range(n_batch):
        
        # batch가 벡터 값이므로 이제부터 계산하는 forward, backward는 다 벡터 계산이다
        batch = data[batch_idx*batch_size : (batch_idx+1)*batch_size]

        Z1 = node1.forward(th1,batch[:,0])
        Z2 = node2.forward(Z1, th0)
        Z3 = node3.forward(batch[:,0], Z2)
        L = node4.forward(Z3)
        J = node5.forward(L)

        dL = node5.backward(1)
        dZ3 = node4.backward(dL)
        dy, dZ2 = node3.backward(dZ3)
        dZ1, dTh0 = node2.backward(dZ2)
        dTh1, dX = node1.backward(dZ1)

        th1 = th1 - lr*np.sum(dTh1)
        th0 = th0 - lr*np.sum(dTh0)

        th1_list.append(th1)
        th0_list.append(th0)
        cost_list.append(J)

fig, ax = plt.subplots(2, 1, figsize = (42,20))
ax[0].plot(th_list, linewidth = 5)
ax[1].plot(cost list, linewidth=5)

