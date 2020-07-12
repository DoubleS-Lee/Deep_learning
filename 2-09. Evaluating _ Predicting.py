#!/usr/bin/env python
# coding: utf-8

# # TensorFlow: Evaluating & Prediction

# In[1]:


import tensorflow as tf
from tensorflow.keras import layers

from tensorflow.keras import datasets 


# # Build Model

# In[2]:


input_shape = (28, 28, 1)
num_classes = 10

learning_rate = 0.001


# In[3]:


inputs = layers.Input(input_shape)
net = layers.Conv2D(32, (3, 3), padding='SAME')(inputs)
net = layers.Activation('relu')(net)
net = layers.Conv2D(32, (3, 3), padding='SAME')(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)
net = layers.Dropout(0.5)(net)

net = layers.Conv2D(64, (3, 3), padding='SAME')(net)
net = layers.Activation('relu')(net)
net = layers.Conv2D(64, (3, 3), padding='SAME')(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)
net = layers.Dropout(0.5)(net)

net = layers.Flatten()(net)
net = layers.Dense(512)(net)
net = layers.Activation('relu')(net)
net = layers.Dropout(0.5)(net)
net = layers.Dense(num_classes)(net)
net = layers.Activation('softmax')(net)

model = tf.keras.Model(inputs=inputs, outputs=net, name='Basic_CNN')


# In[4]:


# Model is the full model w/o custom layers
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# # Preprocess

# 데이터셋 불러오기 

# In[5]:


(train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()


# In[6]:


train_x = train_x[..., tf.newaxis]
test_x = test_x[..., tf.newaxis]

train_x = train_x / 255.
test_x = test_x / 255.


# # Training

# In[7]:


num_epochs = 1
batch_size = 64


# In[8]:


hist = model.fit(train_x, train_y, 
                 batch_size=batch_size, 
                 shuffle=True)


# In[25]:


hist.history


# # Evaluating  
# - 학습한 모델 확인

# In[26]:


model.evaluate(test_x, test_y, batch_size=batch_size)


# ### 결과 확인

# Input으로 들어갈 이미지 데이터 확인

# In[27]:


import matplotlib.pyplot as plt

import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


# In[28]:


test_image = test_x[0, :, :, 0]
test_image.shape


# In[31]:


plt.title(test_y[0])
plt.imshow(test_image, 'gray')
plt.show()


# - 모델에 Input Data로 확인 할 이미지 데이터 넣기

# In[33]:


test_image.shape


# In[34]:


pred = model.predict(test_image.reshape(1, 28, 28, 1))


# In[35]:


pred.shape


# In[ ]:


0 0 0 0 0 0 0 0 0 0 


# In[38]:


pred


# - np.argmax

# In[37]:


np.argmax(pred)


# ## Test Batch

# Batch로 Test Dataset 넣기

# In[39]:


test_batch = test_x[:32]
test_batch.shape


# Batch Test Dataset 모델에 넣기

# In[43]:


preds = model.predict(test_batch)
preds.shape


# - 결과 확인

# In[47]:


np.argmax(preds, -1)


# In[50]:


plt.imshow(test_batch[2, :, :, 0], 'gray')
plt.show()

