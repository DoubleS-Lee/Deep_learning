#!/usr/bin/env python
# coding: utf-8

# # PyTorch Basic
# 
# PyTorch 기초 사용법

# In[1]:


import numpy as np

import torch


# In[2]:


nums = torch.arange(9)
nums


# In[3]:


nums.shape


# In[4]:


type(nums)


# In[5]:


nums.numpy()


# In[6]:


nums.reshape(3, 3)


# In[7]:


randoms = torch.rand((3, 3))
randoms


# In[8]:


zeros = torch.zeros((3, 3))
zeros


# In[9]:


ones = torch.ones((3, 3))
ones


# In[10]:


torch.zeros_like(ones)


# In[11]:


zeros.size()


# # Operations

# PyTorch로 수학연산

# In[12]:


nums * 3


# In[13]:


nums = nums.reshape((3, 3))


# In[14]:


nums + nums


# In[15]:


result = torch.add(nums, 10)


# In[16]:


result.numpy()


# ## View

# In[17]:


range_nums = torch.arange(9).reshape(3, 3)


# In[18]:


range_nums


# In[19]:


range_nums.view(-1)


# In[21]:


range_nums.view(1, 9)


# # Slice and Index

# In[22]:


nums


# In[23]:


nums[1]


# In[24]:


nums[1, 1]


# In[25]:


nums[1:, 1:]


# In[26]:


nums[1:]


# ## Compile

# numpy를 torch tensor로 불러오기

# In[27]:


arr = np.array([1, 1, 1])


# In[28]:


arr_torch = torch.from_numpy(arr)


# In[31]:


arr_torch.float()


# In[32]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[33]:


arr_torch.to(device)


# In[34]:


device


# # AutoGrad

# In[36]:


x = torch.ones(2, 2, requires_grad=True)
x


# In[37]:


y = x + 2
y


# In[38]:


print(y.grad_fn)


# In[39]:


z = y * y * 3
out = z.mean()


# In[40]:


print(z, out)


# In[41]:


out.backward()


# In[43]:


print(x.grad)


# In[44]:


print(x.requires_grad)
print((x ** 2).requires_grad)


# In[45]:


with torch.no_grad():
    print((x ** 2).requires_grad)


# In[ ]:




