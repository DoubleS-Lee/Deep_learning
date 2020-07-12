import tensorflow as tf

# ## Input Image  
# Input으로 들어갈 DataSet을 들여다보면서 시각화까지 한다

# 패키지 로드  
# - os
# - glob
# - matplotlib

import os
import matplotlib.pyplot as plt

#1. mnist 데이터셋 불러오기
from tensorflow.keras import datasets
(train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()

#2. 데이터 확인해보고 전처리하는 과정
# 첫번째 이미지를 image로 불러온다
image = train_x[0]
#print(image)
# 이미지 shape 확인
image.shape
#print(image.shape)
# 차원 수 높이기
# shape의 구성요소 [Batch Size, Height, Width, Channel] 를 맞춰주기 위해서 맨 앞과 끝에 채널 추가
image = image[tf.newaxis, ..., tf.newaxis]
#print(image)

#################################################################################################################################
#3. 사진의 특징을 추출하는 과정에 대한 설명(Feature Extraction)
# filters: layer에서 나갈 때 몇 개의 filter를 만들 것인지 (a.k.a weights, filters, channels)  
# kernel_size: filter(Weight)의 사이즈  
# strides: 몇 개의 pixel을 skip 하면서 훑어지나갈 것인지 (사이즈에도 영향을 줌)  
# padding: zero padding을 만들 것인지. VALID는 Padding이 없고, SAME은 Padding이 있음 (사이즈에도 영향을 줌)  
# activation: Activation Function을 만들것인지. 당장 설정 안해도 Layer층을 따로 만들 수 있음

tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='VALID', activation='relu')
#################################################################################################################################

# (3, 3) 대신에 3으로도 대체 가능
#tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=(1, 1), padding='VALID', activation='relu')


#4. 아웃풋
# ### Visualization
# image는 내부 값이 int이면 오류가 나기 때문에 float로 바꿔준다
image = tf.cast(image, dtype=tf.float32)
layer = tf.keras.layers.Conv2D(3, 3, strides=(1, 1), padding='SAME')
output = layer(image)
# print(output)
# plt.subplot(1,2,1)
# plt.imshow(image[0,:,:,0],'gray')
# plt.subplot(1,2,2)
# plt.imshow(output[0,:,:,0], 'gray')
# plt.show()

#5. weight 불러오기
# - layer.get_weights()
weight = layer.get_weights()
# print(weight[0].shape)
# print(weight[1].shape)

weight = layer.get_weights()[0]
# print(weight[0].shape)
# print(weight[1].shape)

plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.hist(output.numpy().ravel(), range=[-2,2])
plt.ylim(0, 500)
plt.subplot(132)
plt.title(weight.shape)
plt.imshow(weight[:,:,0,0], 'gray')
plt.subplot(133)
plt.title(output.shape)
plt.imshow(output[0, :, :, 0], 'gray')
plt.colorbar()
plt.show()


#6. Activation Function 설정
# out = tf.keras.layers.Conv2D(3, 3, strides=(1, 1), padding='SAME')(out)
layer = tf.keras.layers.ReLU()
output = layer(output)
#print(output.shape)

plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.hist(output.numpy().ravel(), range=[-2,2])
plt.ylim(0, 500)
plt.subplot(132)
plt.title(output.shape)
plt.imshow(output[0, :, :, 0], 'gray')
plt.colorbar()
plt.show()


#7. Pooling
# - tf.keras.layers.MaxPool2D
layer = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')
output = layer(output)
#print(output.shape)

plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.hist(output.numpy().ravel(), range=[-2,2])
plt.ylim(0, 500)
plt.subplot(132)
plt.title(output.shape)
plt.imshow(output[0, :, :, 0], 'gray')
plt.colorbar()
plt.show()


##### Fully Connected

#8. Flatten
# - tf.keras.layers.Flatten()
layer = tf.keras.layers.Flatten()
output = layer(output)
#print(output.shape)

plt.figure(figsize=(10, 5))
plt.subplot(211)
plt.hist(output.numpy().ravel())
plt.subplot(212)
plt.imshow(output[:,:100])
plt.show()


#9. Dense
# - tf.keras.layers.Dense
layer = tf.keras.layers.Dense(32, activation='relu')
output = layer(output)
#print(output.shape)

plt.figure(figsize=(10, 5))
plt.subplot(211)
plt.hist(output.numpy().ravel())
plt.subplot(212)
plt.imshow(output[:,:100])
plt.show()


#10. DropOut
#학습할때마다 스스로 노드의 연결을 끊고 연결하는 작업
# - tf.keras.layers.Dropout
layer = tf.keras.layers.Dropout(0.7)
output = layer(output)
#print(output.shape)

plt.figure(figsize=(10, 5))
plt.subplot(211)
plt.hist(output.numpy().ravel())
plt.subplot(212)
plt.imshow(output[:,:100])
plt.show()



######################################################################################################
# 위의 과정을 요약해서 정리함
# Build Model
from tensorflow.keras import layers

input_shape = (28, 28, 1)
num_classes = 10

# 인풋
inputs = layers.Input(input_shape)

# 첫번째 convolution 블럭
net = layers.Conv2D(32, (3, 3), padding='SAME')(inputs)
net = layers.Activation('relu')(net)
net = layers.Conv2D(32, (3, 3), padding='SAME')(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)
net = layers.Dropout(0.25)(net)

# 두번째 convolution 블럭
net = layers.Conv2D(64, (3, 3), padding='SAME')(net)
net = layers.Activation('relu')(net)
net = layers.Conv2D(64, (3, 3), padding='SAME')(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)
net = layers.Dropout(0.25)(net)

# Fully connected
net = layers.Flatten()(net)
net = layers.Dense(512)(net)
net = layers.Activation('relu')(net)
net = layers.Dropout(0.5)(net)
#여기서 num_classes는 output 종류의 개수이다
#여기서는 0~9까지의 분류 문제니까 10이라고 할 수 있다  
net = layers.Dense(num_classes)(net)
net = layers.Activation('softmax')(net)

# 위에서 정의한 것들을 가지고 모델 정의
model = tf.keras.Model(inputs=inputs, outputs=net, name='Basic_CNN')


# 생성한 모델을 요약해서 볼수 있는 기능 Summary
model.summary()

