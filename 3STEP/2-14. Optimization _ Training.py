#!/usr/bin/env python
# coding: utf-8

# # PyTorch: Optimization & Training

# https://github.com/pytorch/examples/tree/master/mnist

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np


# In[2]:


seed = 1

batch_size = 64
test_batch_size = 64

no_cuda = False


# In[3]:


use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


# # Preprocess

# In[6]:


torch.manual_seed(seed)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('dataset', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)


test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('dataset', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=test_batch_size, shuffle=True)


# # Model

# In[5]:


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


# # Optimization

# - Model과 Optimization 설정

# In[7]:


model = Net().to(device)


# In[8]:


optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)


# # Before Training

# - 학습하기 전에 Model이 Train할 수 있도록 Train Mode로 변환
#     - Convolution 또는 Linear 뿐만 아니라, DropOut과 추후에 배우게 될 Batch Normalization과 같이 parameter를 가진 Layer들도 학습하기 위해 준비

# In[11]:


model.train()  # train mode


# - 모델에 넣기 위한 첫 Batch 데이터 추출

# In[12]:


data, target = next(iter(train_loader))


# In[13]:


data.shape, target.shape


# - 추출한 Batch 데이터를 cpu 또는 gpu와 같은 device에 compile

# In[14]:


data, target = data.to(device), target.to(device)


# In[15]:


data.shape, target.shape


# - gradients를 clear해서 새로운 최적화 값을 찾기 위해 준비

# In[16]:


optimizer.zero_grad()


# - 준비한 데이터를 model에 input으로 넣어 output을 얻음

# In[17]:


output = model(data)


# - Model에서 예측한 결과를 Loss Function에 넣음
#     - 여기 예제에서는 Negative Log-Likelihood Loss 라는 Loss Function을 사용

# In[18]:


loss = F.nll_loss(output, target)


# - Back Propagation을 통해 Gradients를 계산

# In[19]:


loss.backward()


# - 계산된 Gradients는 계산된 걸로 끝이 아니라 Parameter에 Update

# In[20]:


optimizer.step()


# # Start Training

# 위의 최적화 과정을 반복하여 학습 시작

# In[21]:


epochs = 1
log_interval = 100


# In[22]:


for epoch in range(1, epochs+1):
    # Train Mode
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100 * batch_idx / len(train_loader), loss.item()
            ))
            
    model.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


# # Evaluation

# - 앞에서 model.train() 모드로 변한 것처럼 평가 할 때는 model.eval()로 설정
#     - Batch Normalization이나 Drop Out 같은 Layer들을 잠금

# In[23]:


model.eval()


# - autograd engine, 즉 backpropagatin이나 gradient 계산 등을 꺼서 memory usage를 줄이고 속도를 높임

# In[26]:


test_loss = 0
correct = 0

with torch.no_grad():
    data, target = next(iter(test_loader))
    data, target = data.to(device), target.to(device)
    output = model(data)
    
    test_loss += F.nll_loss(output, target, reduction='sum').item()
    
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()


# In[33]:


output.shape


# In[32]:


pred.shape


# In[37]:


target.view_as(pred).shape


# In[41]:


pred.eq(target.view_as(pred)).sum().item() / 64


# In[27]:


test_loss


# In[28]:


correct


# In[42]:


test_loss /= len(test_loader.dataset)


# In[43]:


test_loss


# In[ ]:





# In[45]:


model.eval()

test_loss = 0
correct = 0

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)

print('\nTest set: Average Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


# In[ ]:




