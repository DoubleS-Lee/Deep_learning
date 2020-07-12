#!/usr/bin/env python
# coding: utf-8

# # PyTorch Layer 이해하기

# # 예제 불러오기

# In[4]:


import torch
from torchvision import datasets, transforms


# In[2]:


import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('dataset', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=1)


# In[6]:


image, label = next(iter(train_loader))


# In[8]:


image.shape, label.shape


# In[10]:


plt.imshow(image[0, 0, :, :], 'gray')
plt.show()


# # 각 Layer별 설명

# - Network 쌓기 위한 준비

# In[11]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# ### Convolution

# - in_channels: 받게 될 channel의 갯수
# - out_channels: 보내고 싶은 channel의 갯수  
# - kernel_size: 만들고 싶은 kernel(weights)의 사이즈

# In[15]:


nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1)


# In[16]:


layer = nn.Conv2d(1, 20, 5, 1).to(torch.device('cpu'))
layer


# - weight 시각화를 위해 slice하고 numpy화

# In[19]:


weight = layer.weight
weight.shape


# - 여기서 weight는 학습 가능한 상태이기 때문에 바로 numpy로 뽑아낼 수 없음
# - detach() method는 그래프에서 잠깐 빼서 gradient에 영향을 받지 않게 함

# In[21]:


weight = weight.detach().numpy()


# In[24]:


weight.shape


# In[25]:


plt.imshow(weight[0, 0, :, :], 'jet')
plt.colorbar()
plt.show()


# - output 시각화 준비를 위해 numpy화

# In[26]:


output_data = layer(image)


# In[29]:


output_data = output_data.data


# In[30]:


output = output_data.cpu().numpy()


# In[32]:


output.shape


# - Input으로 들어간 이미지 numpy화 

# In[34]:


image_arr = image.numpy()
image_arr.shape


# In[35]:


plt.figure(figsize=(15, 30))
plt.subplot(131)
plt.title('Input')
plt.imshow(np.squeeze(image_arr), 'gray')
plt.subplot(132)
plt.title('Weight')
plt.imshow(weight[0, 0, :, :], 'jet')
plt.subplot(133)
plt.title('Output')
plt.imshow(output[0, 0, :, :], 'gray')
plt.show()


# In[ ]:





# In[ ]:





# ### Pooling

# input을 먼저 앞에 넣고, 뒤에 kernel 사이즈와 stride를 순서대로 넣음

# In[36]:


image.shape


# In[37]:


pool = F.max_pool2d(image, 2, 2)
pool.shape


# - MaxPool Layer는 weight가 없기 때문에 바로 numpy()가 가능

# In[39]:


pool_arr = pool.numpy()
pool_arr.shape


# In[40]:


image_arr.shape


# In[41]:


plt.figure(figsize=(10, 15))
plt.subplot(121)
plt.title("Input")
plt.imshow(np.squeeze(image_arr), 'gray')
plt.subplot(122)
plt.title('Output')
plt.imshow(np.squeeze(pool_arr), 'gray')
plt.show()


# ### Linear

# nn.Linear는 2d가 아닌 1d만 들어가기 때문에 .view() 1D로 펼쳐줘야함

# In[42]:


image.shape


# In[43]:


flatten = image.view(1, 28 * 28)
flatten.shape


# In[44]:


lin = nn.Linear(784, 10)(flatten)
lin.shape


# In[45]:


lin


# In[47]:


plt.imshow(lin.detach().numpy(), 'jet')
plt.show()


# ### Softmax

# 결과를 numpy로 꺼내기 위해선 weight가 담긴 Linear에 weight를 꺼줘야함

# In[48]:


with torch.no_grad():
    flatten = image.view(1, 28 * 28)
    lin = nn.Linear(784, 10)(flatten)
    softmax = F.softmax(lin, dim=1)


# In[49]:


softmax


# In[51]:


np.sum(softmax.numpy())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Layer 쌓기

# 예제 출처: https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#id23

# nn 과 nn.functional의 차이점
# - nn은 학습 파라미터가 담긴 것 
# - nn.functional은 학습 파라미터가 없는 것이라 생각하면 간단

# In[61]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
        
    def forward(self, x):
        # Feature Extraction
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        
        # Fully Connected (Classficiation)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# - Image를 Model에 넣어서 결과 확인

# In[62]:


model = Net()


# In[63]:


result = model.forward(image)


# In[64]:


result


# In[65]:




