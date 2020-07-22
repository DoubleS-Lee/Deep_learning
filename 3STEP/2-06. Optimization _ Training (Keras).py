#!/usr/bin/env python
# coding: utf-8

# # Optimization & Training (Beginner)

# - tf와 layers 패키지 불러오기

# In[3]:


import tensorflow as tf
from tensorflow.keras import layers

from tensorflow.keras import datasets


# ## 학습 과정 돌아보기

# ![image.png](attachment:image.png)

# # Prepare MNIST Datset

# In[4]:


(train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()


# ## Build Model

# ![image.png](attachment:image.png)

# In[5]:


inputs = layers.Input((28, 28, 1))
net = layers.Conv2D(32, (3, 3), padding='SAME')(inputs)
net = layers.Activation('relu')(net)
net = layers.Conv2D(32, (3, 3), padding='SAME')(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)
net = layers.Dropout(0.25)(net)

net = layers.Conv2D(64, (3, 3), padding='SAME')(net)
net = layers.Activation('relu')(net)
net = layers.Conv2D(64, (3, 3), padding='SAME')(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)
net = layers.Dropout(0.25)(net)

net = layers.Flatten()(net)
net = layers.Dense(512)(net)
net = layers.Activation('relu')(net)
net = layers.Dropout(0.5)(net)
net = layers.Dense(10)(net)  # num_classes
net = layers.Activation('softmax')(net)

model = tf.keras.Model(inputs=inputs, outputs=net, name='Basic_CNN')


# # Optimization  
# 모델을 학습하기 전 설정 

# - Loss Function  
# - Optimization  
# - Metrics

# # Loss Function  
# Loss Function 방법 확인

# ### Categorical vs Binary

# In[ ]:


loss = 'binary_crossentropy'
loss = 'categorical_crossentropy'


# ### sparse_categorical_crossentropy vs categorical_crossentropy

# In[13]:


loss_fun = tf.keras.losses.sparse_categorical_crossentropy


# In[7]:


tf.keras.losses.categorical_crossentropy


# In[8]:


tf.keras.losses.binary_crossentropy


# # Metrics  
# 
# 모델을 평가하는 방법

# accuracy를 이름으로 넣는 방법

# In[10]:


metrics = ['accuracy']


# tf.keras.metrics.

# In[12]:


metrics = [tf.keras.metrics.Accuracy()]


# ## Compile  
# Optimizer 적용

# - 'sgd'
# - 'rmsprop'
# - 'adam'

# In[11]:


optm = tf.keras.optimizers.Adam()


# - tf.keras.optimizers.SGD()  
# - tf.keras.optimizers.RMSprop()    
# - tf.keras.optimizers.Adam()  

# In[15]:


model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss='sparse_categorical_crossentropy', 
              metrics=[tf.keras.metrics.Accuracy()])


# In[ ]:





# # Prepare Dataset  
# 학습에 사용할 데이터셋 준비

# shape 확인

# In[16]:


train_x.shape, train_y.shape


# In[17]:


test_x.shape, test_y.shape


# 차원 수 늘리기

# In[18]:


import numpy as np


# In[20]:


np.expand_dims(train_x, -1).shape


# In[21]:


tf.expand_dims(train_x, -1).shape


# In[22]:


train_x = train_x[..., tf.newaxis]
test_x = test_x[..., tf.newaxis]


# 차원 수 잘 늘었는지 확인

# In[23]:


train_x.shape


# Rescaling

# In[24]:


np.min(train_x), np.max(train_x)


# In[25]:


train_x = train_x / 255.
test_x = test_x / 255.


# In[26]:


np.min(train_x), np.max(train_x)


# # Training  
# 본격적으로 학습 들어가기

# 학습용 Hyperparameter 설정
# 
# - num_epochs
# - batch_size

# In[27]:


num_epochs = 1
batch_size = 32


# - model.fit

# In[28]:


model.fit(train_x, train_y, 
          batch_size=batch_size, 
          shuffle=True, 
          epochs=num_epochs) 


# # Check History  
# 학습 과정(History) 결과 확인

# In[29]:


hist = model.fit(train_x, train_y, 
                 batch_size=batch_size, 
                 shuffle=True, 
                 epochs=num_epochs) 


# In[ ]:




