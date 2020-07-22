# # Data Preprocess (MNIST)
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf


#1. TensorFlow에서 제공해주는 데이터셋(mnist) 예제 불러오기
from tensorflow.keras import datasets
mnist = datasets.mnist
# - 데이터 shape 확인하기
(train_x, train_y), (test_x, test_y) = mnist.load_data()
# print(train_x.shape)
# (6000, 28, 28) : 28 x 28 사이즈의 데이터가 60000개 있다


#2. Image Dataset 들여다보기
# 불러온 데이터셋에서 이미지 데이터 하나만 뽑아서 시각화까지 확인
# - 데이터 하나만 뽑기
image = train_x[0]
image.shape
# - 시각화해서 확인
plt.imshow(image, 'gray')
#plt.show()


#3. Channel 확장
# shape의 구성은 [Batch Size, Height, Width, Channel] 이다
# 근데 mnist 예제는 grayscale이어서 channel이 1이므로 현재 저 항이 비어있다 따라서 데이터 차원수를 늘려서 channel을 추가해줘야한다
# # Channel 관련
# [Batch Size, Height, Width, Channel] 
# GrayScale이면 1, RGB이면 3으로 만들어줘야함
# - 다시 shape로 데이터 확인
train_x.shape
#print(train_x.shape)

#3-1. - 데이터 차원수 늘리기 (numpy)
# np.expand_dims을 사용해서 맨 마지막 열(-1)을 추가해줘서 차원을 늘려준다
new_train_x = np.expand_dims(train_x, -1)
new_train_x.shape
#print(new_train_x.shape)

#3-2. - 데이터 차원수 늘리기 (tensor)
# - TensorFlow 패키지 불러와 데이터 차원수 늘리기 (tensorflow)
new_train_x = tf.expand_dims(train_x, -1)
new_train_x.shape
#print(new_train_x.shape)

#3-3. - TensorFlow 공홈에서 가져온 방법 tf.newaxis
train_x.shape
train_x[..., tf.newaxis].shape
#print(train_x[..., tf.newaxis].shape)

#3-4. reshape 사용
reshaped = train_x.reshape([60000, 28, 28, 1])
reshaped.shape
#print(reshaped.shape)

#4. 시각화(그래프 생성)
# matplotlib로 시각화할때 주의 사항
# 위에서 추가해준 맨 마지막 채널을 다시 빼줘야 시각화가 된다 안 그러면 에러가 남
#
# *주의 사항
# matplotlib로 이미지 시각화 할 때는 gray scale의 이미지는 3번쨰 dimension이 없으므로,  
# 2개의 dimension으로 gray scale로 차원 조절해서 넣어줘야함
new_train_x = train_x[..., tf.newaxis]
new_train_x.shape
#print(new_train_x.shape)

# channel 빼는법 1
disp1 = np.squeeze(new_train_x[0])
disp1.shape
#print(disp1.shape)

# channel 빼는법 2
disp2 = new_train_x[0, :, :, 0]
disp2.shape
#print(disp2.shape)

# - 다시 시각화
plt.imshow(disp1, 'gray')
#plt.show()

# # Label Dataset 들여다보기  (label은 Target하고 똑같은 뜻이다)
# Label 하나를 열어서 Image와 비교하여 제대로 들어갔는지. 어떤 식으로 저장 되어있는지 확인
# - label 하나만 뽑아보기
train_y.shape
#print(train_y.shape)
train_y[0]
#print(train_y[0])

# - Label 시각화 
plt.title(train_y[0])
plt.imshow(train_x[0], 'gray')
#plt.show()


#5. 원핫인코딩(classification 분류 문제의 label 구성 형태)
# OneHot Encoding
# 컴퓨터가 이해할 수 있는 형태로 변환해서 Label을 주도록 함
# 5
[0,0,0,0,0,1,0,0,0,0]
# 9
[0,0,0,0,0,0,0,0,0,1]

#원핫인코딩 생성 방법
from tensorflow.keras.utils import to_categorical

# - 1을 예시로 one hot encoding
to_categorical(1, 5)
#print(to_categorical(1, 5))

# 정답이 5인 그림을 원핫인코딩을 사용하여 [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.] 로 바꿔서 표기해주는 방법
# to_categrical
# - label 확인해서 to_categorical 사용
label = train_y[0]
#print(label)
label_onehot = to_categorical(label, num_classes=10)
#print(label_onehot)


# - onehot encoding으로 바꾼 것과 이미지 확인
plt.title(label_onehot)
plt.imshow(train_x[0], 'gray')
#plt.show()

