import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import basic_nodes as nodes
from dataset_generator import dataset_generator

plt.style.use('seaborn')
np.random.seed(0)


dataset_gen = dataset_generator()

dataset_gen.set_coefficient([5,2])
x_data, y_data = dataset_gen.make_dataset()
data = np.hstack((x_data, y_data))


# model implementation
node1 = nodes.mul_node()
node2 = nodes.plus_node()

# square loss/MSE cost implementation
node3 = nodes.minus_node()
node4 = nodes.square_node()

th1, th0 = 1,0
lr = 0.01
epochs = 2

th1_list, th0_list = [], []
loss_list = []

for epoch in range(epochs):
    for data_idx, (x,y) in enumerate(data):

        z1 = node1.forward(th1,x)
        z2 = node2.forward(z1, th0)
        z3 = node3.forward(y, z2)
        l = node4.forward(z3)

        dl = node4.backward(1)
        dy, dz2 = node3.backward(dl)
        dz1, dth0 = node2.backward(dz2)
        dth1, dx = node1.backward(dz1)

        th1 = th1 - lr*dth1
        th0 = th0 - lr*dth0

        th1_list.append(th1)
        th0_list.append(th0)
        loss_list.append(l)

fig, ax = plt.subplots(2, 1, figsize = (42,20))
ax[0].plot(th_list, linewidth = 5)
ax[1].plot(cost list, linewidth=5)

