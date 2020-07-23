import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import basic_nodes as nodes

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import basic_nodes as nodes
from dataset_generator import dataset_generator

class SVLR:
    # 이 클래스 내에서 사용할 변수들을 선언한다
    def __init__(self, th1, th0):
        self.th1, self.th0 = th1, th0

        self.th1_list, self.th0_list = [], []
        self.cost_list=[]

        #
        self.iter_cnt, self.check_cnt = 0, 0

        #밑에 2개의 메서드를 자동으로 실행시키게 만듦
        self.model_imp()
        self.cost_imp()

    def model_imp(self):
        self.node1 = nodes.mul_node()
        self.node2 = nodes.plus_node()
    
    def cost_imp(self):
        self.node3 = nodes.minus_node()
        self.node4 = nodes.square_node()
        self.node5 = nodes.mean_node()

    def forward(self, mini_batch):
        sekf,iter_cbt += 1

        Z1 = self.node1.forward(self.th1, mini_batch[:,0])
        Z2 = self.node2.forward(Z1, self.th0)
        Z3 = self.node3.forward(mini_batch[:,1], Z2)
        L = self.node4.forward(Z3)
        J = self.node5.forward(L)

        # 이터레이션 카운트가 check_freq의 배수일때만 cost_list를 업데이트 해준다
        if self.iter_cnt % check_freq == 0 or self.iter_cnt == 1:
            self.cost_list.append(J)

    def backward(self, lr):
        # 이터레이션 카운트가 check_freq의 배수일때만 th1_list,th0_list를 업데이트 해준다
        if self.iter_cnt % check_freq == 0 or self.iter_cnt == 1:
            self.th1_list.append(self.th1)
            self.th0_list.append(self.th0)
            self.check_cnt += 1
        
        dL = self.node5.backward(1)
        dZ3 = self.node4.backward(dL)
        dY, dZ2 = self.node3.backward(dZ3)
        dZ1, dTh0 = self.node2.backward(dZ2)
        dTh1, dX = self.node1.backward(dZ1)

        self.th1 = self.th1 - lr*np.sum(dTh1)
        self.th0 = self.th0 - lr*np.sum(dTh0)

        self.iter_cnt += 1

    def result_visualization(self):

        fig, ax = plt.subplots(2, 1, figsize = (30.15))
        ax[0].plot(self.th1_list, label = r'$\theta_{1}$')
        ax[0].plot(self.th0_list, label = r'$\theta_{1}$')
        ax[1].plot(self.cost_list)
        ax[0].legend(loc = 'lower right', fontsize = 30)

        fig,subplots_adjust(top = 0.95, bottom = 0.05, left = 0.05, right = 0.95, hspace = 0.03)
        ax[0].axex.get_xaxis().set_visible(False)
        ax[0].axex.get_xaxis().set_visible(False)

        x_ticks = np.linspace(0, self.shck_cnt, 10).astype(int)
        x_ticklabels = x_ticks*check_freq
        y_ticks = np.arange(0, t_th0 + 0.5)

        ax[1].set_xticks(x_ticks)
        ax[1].set_xticklabels(x_ticklabels)
        ax[0].set_yticks(y_ticks)

        ax[0].tick_params(axis = 'both', labelsize = 40)
        ax[1].tick_params(axis = 'both', labelsize = 40)

def get_data_batch(data, batch_idx):
    # 맨 마지막 짜투리 데이터를 이용하는 코드
    if batch_idx is n_batch - 1:
        batch = data[batch_idx * batch_size : ]
    # 마지막 전까지 정상적인 배치 크기의 데이터를 이용하는 코드
    else:
        batch = data[batch_idx * batch_size : (batch_idx+1)*batch_size]
    return batch

plt.style.use('seaborn')
np.random.seed(0)

# parameter setting
t_th1, t_th0 = 5,5
th1, th0 = 1,1

# data params setting
distribution_params = {'feature_0' : {'mean':0, 'std':1}}

# learning params setting
lr = 0.01
epochs = 10
batch_size = 4
check_freq = 3

# dataset preparation
dataset_gen = dataset_generator()

dataset_gen.set_coefficient([t_th1, t_th0])
dataset_gen.set_distribution_params(distribution_params)
x_data, y_data = dataset_gen.make_dataset()
data = np.hstack((x_data, y_data))
n_batch = np.ceil(data.shape[0]/batch_size).astype(int)

# Learning
model = SVLR(th1, th0)

for epoch in range(epochs):
    np.random.shuffle(data)

    for batch_idx in range(n_batch):
        batch = get_data_batch(data, batch_idx)

        model.forward(batch)
        model.backward(lr)

# Visualization
model.result_visualization()

