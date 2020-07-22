#!/usr/bin/env python
# coding: utf-8

# https://github.com/pytorch/examples/tree/master/mnist

# # PyTorch Data Preprocess

# In[3]:


import torch

from torchvision import datasets, transforms


# ### Data Loader 부르기
# 
# 파이토치는 DataLoader를 불러 model에 넣음

# In[4]:


batch_size = 32
test_batch_size = 32


# In[5]:


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('dataset/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize(mean=(0.5,), std=(0.5,))
                   ])),
    batch_size=batch_size,
    shuffle=True)


# In[7]:


test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('dataset', train=False, 
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5))
                   ])),
    batch_size=test_batch_size,
    shuffle=True)


# In[ ]:





# ### 첫번재 iteration에서 나오는 데이터 확인

# In[8]:


images, labels = next(iter(train_loader))


# In[9]:


images.shape


# In[10]:


labels.shape


# PyTorch는 TensorFlow와 다르게 [Batch Size, Channel, Height, Width] 임을 명시해야함

# ### 데이터 시각화

# In[11]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


images[0].shape


# In[13]:


torch_image = torch.squeeze(images[0])
torch_image.shape


# In[14]:


image = torch_image.numpy()
image.shape


# In[15]:


label = labels[0].numpy()


# In[16]:


label.shape


# In[17]:


label


# In[18]:


plt.title(label)
plt.imshow(image, 'gray')
plt.show()


# In[ ]:




