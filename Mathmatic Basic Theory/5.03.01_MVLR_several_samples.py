import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from dataset_generator_v2 import dataset_generator
import basic_nodes as nodes

feature_dim = 2
batch_size = 8
plt.style.use('seaborn')
np.random.seed(0)

def get_data_batch(data, batch_idx):
    if batch_idx is n_batch -1:
        batch = data[batch_idx*batch_size:]
    else:
        batch = data[batch_idx*batch_size : (batch_idx+1)*batch_size]
    return batch

data_gen = dataset_generator(feature_dim = feature_dim)
x_data, y_data = data_gen.make_dataset()

data = np.hstack((x_data, y_data))
n_batch = np.ceil(data.shape[0]/batch_size).astype(int)

Th = np.random.normal(0, 1, size = (feature_dim + 1)).reshape(-1,1)

epochs, lr = 100, 0.001


# model implementation
node1 = [None] + [nodes.mul_node() for _ in range(feature_dim)]
node2 = [None] + [nodes.plus_node() for _ in range(feature_dim)]

# cost implementation
node3 = nodes.minus_node()
node4 = nodes.square_node()
node5 = nodes.mean_node()


for epoch in range(epochs):
    np.random.shuffle(data)
    for batch_idx in range(n_batch):
        batch = get_data_batch(data, batch_idx)
        X, Y = batch[:,:-1], batch[:,-1]
        
        Z1_list = [None]*(feature_dim + 1)
        Z2_list = Z1_list.copy()
        dZ1_list, dZ2_list = Z1_list.copy(), Z1_list.copy()
        dTh_list = dZ1_list.copy()

        for node_idx in range(1, feature_dim + 1):
            Z1_list[node_idx] = node1[node_idx].forward(Th[node_idx], X[:,node_idx])

        Z2_list[1] = node2[1].forward(Th[0], Z1_list[1])

        for node_idx in range(2, feature_dim + 1):
            Z2_list[node_idx] = node2[node_idx].forward(Z2_list[node_idx-1], Z1_list[node_idx])

        Z3 = node3.forward(Y, Z2_list[-1])
        Z4 = node4.forward(Z3)
        J = node5.forward(Z4)


        dZ4 = node5.backward(1)
        dZ3 = node4.backward(dZ4)
        _, dZ2_last = node3.backward(dZ3)
        dZ2_list[-1] = dZ2_last
        for node_idx in reversed(range(1, feature_dim + 1)):
            dZ2, dZ1 = node2[node_idx].backward(dZ2_list[node_idx])
            dZ2_list[node_idx-1] = dZ2
            dZ1_list[node_idx] = dZ1
        dTh_list[0] = dZ2_list[0]
        for node_idx in reversed(range(1, feature_dim + 1)):
            dTh, _ = node1[node_idx].backward(dZ1_list[node_idx])
            dTh_list[node_idx] = dTh
        for th_idx in range(Th.shape[0]):
            Th[th_idx] = Th[th_idx] - lr*np.sum(dTh_list[th_idx])

class Affine_Function:
    def __init__(self):
        self._feature_dim = feature_dim
        self._Th = Th
        self._z1_list = [None]*(self._feature_dim + 1)
        self._z2_list = self._z1_list.copy()
        self._dz1_list, self._dz2_list = self._z1_list.copy(), self._z1_list.copy()
        self._dth_list = self._z1_list.copy()
        self.affine_imp()

    def affine_imp(self):
        self._node1 = [None] + [nodes.mul_node() for _ in range(self._feature_dim)]
        self._node2 = [None] + [nodes.plus_node() for _ in range(self._feature_dim)]
        
    def forward(self, X):
        for node_idx in range(1, self._feature_dim + 1):
            self._Z1_list[node_idx] = self._node1[node_idx].forward(self._Th[node_idx], X[:,node_idx])
        self._Z2_list[1] = self._node2[1].forward(self._Th[0], self._Z1_list[1])
        for node_idx in range(2, feature_dim + 1):
            self._Z2_list[node_idx] = self._node2[node_idx].forward(self._Z2_list[node_idx-1], self._Z1_list[node_idx])
        return self._Z2_list[-1]

    def backward(self, dZ2_last, lr):
        self._dZ2_list[-1] = dZ2_last
        for node_idx in reversed(range(1, self._feature_dim + 1)):
            dZ2, dZ1 = self._node2[node_idx].backward(self._dZ2_list[node_idx])
            self._dZ2_list[node_idx-1] = dZ2
            self._dZ1_list[node_idx] = dZ1

        self._dTh_list[0] = self._dZ2_list[0]
        for node_idx in reversed(range(1, self._feature_dim + 1)):
            dTh, _ = self._node1[node_idx].backward(self._dZ1_list[node_idx])
            self._dTh_list[node_idx] = dTh
        for th_idx in range(self._Th.shape[0]):
            self._Th[th_idx] = self._Th[th_idx] - lr*np.sum(self._dTh_list[th_idx])
        return self._Th

class MSE_Cost:
    def __init__(self):
        self.cost_imp()
    def cost_imp(self):
        self._node3 = nodes.minus_node()
        self._node4 = nodes.square_node()
        self._node5 = nodes.mean_node()
    def forward(self, Y, Pred):
        Z3 = self._node3.forward(Y, Pred)
        Z4 = self._node4.forward(Z3)
        J = self._node5.forward(Z4)
        return J
    def backward(self):
        dZ4 = self._node5.backward(1)
        dZ3 = self._node4.backward(dZ4)
        _, dZ2_last = self._node3.backward(dZ3)
        return dZ2_last

affine = Affine_Function()
cost = MSE_Cost()

for epoch in range(epochs):
    np.random.shuffle(data)
    for batch_idx in range(n_batch):
        batch = get_data_batch(data, batch_idx)
        X, Y = batch[:,:-1], batch[:,-1]
        Pred = affine.forward(X)
        J = cost.forward(Y, Pred)
        dPred = cost.backward()
        affine.backward(dPred, lr)
        th_accum = np.hstack((th_accum, affine._Th))
        cost_list.append(J)




