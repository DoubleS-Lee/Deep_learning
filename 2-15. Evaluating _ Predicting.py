#!/usr/bin/env python
# coding: utf-8

# # PyTorch: Evaluating

# https://github.com/pytorch/examples/tree/master/mnist

# In[1]:


import os
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np


# In[2]:


seed = 1

lr = 0.001
momentum = 0.5

batch_size = 64
test_batch_size = 64

epochs = 1
no_cuda = False
log_interval = 100


# # Model

# In[3]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# # Preprocess

# In[4]:


datasets.MNIST('../dataset', train=True, download=True,
               transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
               ]))


# In[5]:


torch.manual_seed(seed)

use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}



train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)




test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=test_batch_size, shuffle=True, **kwargs)


# # Optimization

# In[6]:


model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)


# In[7]:


params = list(model.parameters())


# In[8]:


for i in range(8):
    print(params[i].size())


# # Training

# In[9]:


for epoch in range(1, epochs + 1):
    # Train Mode
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # backpropagation 계산하기 전에 0으로 기울기 계산
        output = model(data)
        loss = F.nll_loss(output, target)  # https://pytorch.org/docs/stable/nn.html#nll-loss
        loss.backward()  # 계산한 기울기를 
        optimizer.step()  

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    


# # Evaluation

# - 앞에서 model.train() 모드로 변한 것처럼 평가 할 때는 model.eval()로 설정
#     - Batch Normalization이나 Drop Out 같은 Layer들을 잠금

# In[ ]:


# Test mode
model.eval()  # batch norm이나 dropout 등을 train mode 변환


# - autograd engine, 즉 backpropagatin이나 gradient 계산 등을 꺼서 memory usage를 줄이고 속도를 높임

# In[ ]:


# 뒤에서 설명
test_loss = 0
correct = 0  

with torch.no_grad():
    data, target = next(iter(test_loader))
    data, target = data.to(device), target.to(device)
    output = model(data)
    
    # Model에서 나온 결과의 Loss를 계산하여 test_loss에 더함
    test_loss += F.nll_loss(output, target, reduction='sum').item()
    
    correct += pred.eq(target.view_as(pred)).sum().item()  # pred와 target과 같은지 확인


# In[ ]:


# 앞에서 얻은 test_loss의 값을 test datset의 갯수로 나눠 평균값을 얻음
test_loss /= len(test_loader.dataset)

# test_loss, correct의 Log 확인
print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))


# In[10]:


# Test mode
model.eval()  # batch norm이나 dropout 등을 train mode 변환
test_loss = 0
correct = 0
with torch.no_grad():  # autograd engine, 즉 backpropagatin이나 gradient 계산 등을 꺼서 memory usage를 줄이고 속도를 높임
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()  # pred와 target과 같은지 확인

test_loss /= len(test_loader.dataset)

print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

