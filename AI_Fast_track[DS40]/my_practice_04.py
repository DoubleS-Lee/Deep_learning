#%%
딥러닝
이미지의 경우 : 사람은 이미지를 처음 봤을때 조각조각 쪼개서 이해를 하면서 점점 합쳐서 뇌쪽으로 전달하는거고 최종적으로는 이미지를 전체적으로 이해할수있게 되는 것이다

convolution
= 필터
라디오를 예로 들면 모든 주파수가 다 들어올때 내가 원하는 주파수의 필터(91.9)를 곱해주면 91.9에 대한 데이터만 남고 다른 건 다 사라진다
원본 데이터로 부터 특정 주파수만 남기고 나머지는 다 사라지게 하는 것


CNN
데이터의 feature를 결정해주는 필터를 결정해주는 과정에서
컨볼루션 필터를 학습을 통해서 결정하겠다는게 컨셉


32x32x3 입력
5x5x6 필터 6개
28x28x6 필터 6개
5x5x6 필터 

inception 네트워크 -> 구글넷


오토인코더

#%%
import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
import tensorflow as tf

tf.debugging.set_log_device_placement(True)

batch_size = 128
num_classes = 10
epochs = 3

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

print(input_shape)

# 원핫인코딩
# convert class vectors to binary class matrices
y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)

# build your model
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', activation = 'relu', input_shape = x_train.shape[1:]))
model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(num_classes, activation = 'softmax'))
model.add(Dropout(0.5))

'''
# model API
from tensorflow.keras import Input, Model
inputs = Input(shape = x_train.shape[1:])
x = Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu')(inputs)
x = Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu')(x)
x = MaxPooling2D((2,2))(x)
x = Dropout(0.5)(x)
x = Flatten()(x)
x = Dense(128, activation = 'relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(num_classes, activation = 'softmax')(x)

model = Model(inputs = inputs, outputs = outputs)

model.summary()
'''

model.compile(loss='categorical_crossentropy',
              optimizer=tensorflow.keras.optimizers.Adam(),
              metrics=['accuracy'])
# 로스를 이걸 쓰면 위에 원핫인코딩을 따로 안해줘도 알아서 프로그램이 자동으로 해준다
# loss='sparse_categorical_crossentropy'

model.summary()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#%%
import tensorflow.keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
from tensorflow import keras

tf.debugging.set_log_device_placement(True)

batch_size = 32
num_classes = 10
epochs = 5

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

input_shape = x_train.shape[1:]
# = input_shape = x_train[0].shape

# 원핫인코딩
# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


### model add
# model API
from tensorflow.keras import Input, Model
inputs = Input(shape = input_shape)
x = Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', padding = 'same')(inputs)
x = Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', padding = 'same')(x)
x = MaxPooling2D(strides = 2, pool_size = (2,2))(x)
x = Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', padding = 'same')(x)
x = MaxPooling2D(strides = 2, pool_size = (2,2))(x)
x = Dropout(0.5)(x)

x = Flatten()(x)
x = Dense(512, activation = 'relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(num_classes, activation = 'softmax')(x)

model = Model(inputs = inputs, outputs = outputs)

model.summary()

# Let's train the model using categorical loss & adam
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['acc'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))


# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


#%%
open cv

import cv2

에 대해 알아보기



#%%
Transfer Learning

Fine Tuning

pretrained model
사전 훈련
전이 학습













#%%
# pretrained model

import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K

batch_size = 128
num_classes = 10
epochs = 3

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

print(input_shape)
# convert class vectors to binary class matrices
y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)

# build your model
# model API
from tensorflow.keras import Input, Model
inputs = Input(shape = x_train.shape[1:])
x = Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu')(inputs)
x = Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu')(x)
x = MaxPooling2D((2,2))(x)
x = Dropout(0.5)(x)
x = Flatten()(x)
x = Dense(128, activation = 'relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(num_classes, activation = 'softmax')(x)

model = Model(inputs = inputs, outputs = outputs)

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=tensorflow.keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('my_mnist_model.h5')


# 재학습 (pretrained)
model = load_model('my_mnist_model.h5')






model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=9,
          verbose=1,
          validation_split= 0.1)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



#################################### layers 추가
model = load_model('my_mnist_model.h5')
model.summary()

model.layers[0](x_train[0])

n = len(model.layers)
'''
top_model = Sequential()
top_model.add(Dense(256, activation='relu',input_shape=model.output_shape))
top_model.add(Dropout(0.5))
top_model.add(Dense(num_classes, activation='sigmoid'))
'''

inputs = inputs(shape = model.output_shape)
x = Dense(256, activation='relu')(inputs)
x = Dropout(0.5)
outputs = Dense(num_classes, activation = 'sigmoid')(x)

top_model = Model(inputs = inputs, outputs = outputs)

Full_model = Model(inputs = model.input, outputs = top_model.output)


for layer in model.layers[:n]:
    layer.trainable = False

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=9,
          verbose=1,
          validation_split= 0.1)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.summary()


#%%
# pretrained model_2
# 웹캠으로 사진 분석
# open cv

from tensorflow.keras.applications import VGG16

image_size = 224
#Load the VGG model
vgg_conv = VGG16(weights='imagenet', include_top=True, input_shape=(image_size, image_size, 3))

vgg_conv.summary()

len(vgg_conv.layers)

vss_conv = vgg_conv.layers[:-4]
vgg_conv.summary()

# Freeze the layers except the last 4 layers
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable)
    
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
 
# Create the model
model = models.Sequential()
 
# Add the vgg convolutional base model
model.add(vgg_conv)
 
# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='softmax'))
 
# Show a summary of the model. Check the number of trainable parameters
model.summary()

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

model = VGG16(weights='imagenet')

model.summary()

######
import cv2
import numpy as np
import matplotlib.pyplot as plt 
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions


cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    

    # Display the resulting frame
    
    dst = cv2.resize(frame, dsize=(224, 224), interpolation=cv2.INTER_AREA)    
    img = img_to_array(dst)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = preprocess_input(img)
    yhat = model.predict(img)

    label = decode_predictions(yhat)
    label = label[0][0]
    
    cv2.imshow('frame',dst)
    
    if cv2.waitKey(1) & 0xFF == ord('w'):
        print('%s (%.2f%%)' % (label[1], label[2]*100))    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


#%%
# 사진으로 예측하기

from tensorflow.keras.applications import VGG16

image_size = 224
#Load the VGG model
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

vgg_conv.summary()

# Freeze the layers except the last 4 layers
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable)
    
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
 
# Create the model
model = models.Sequential()
 
# Add the vgg convolutional base model
model.add(vgg_conv)
 
# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='softmax'))
 
# Show a summary of the model. Check the number of trainable parameters
model.summary()

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

model = VGG16(weights='imagenet')

model.summary()



import cv2
import numpy as np
import matplotlib.pyplot as plt 
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

# load an image from file
img = load_img('C:/Users/Seohee/Documents/AI_fast_track_DS40/img.jpg', target_size=(224, 224))
plt.imshow(img)
plt.show()
img = img_to_array(img)
img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

print(img.shape)

img = preprocess_input(img)

yhat = model.predict(img)

print(yhat.shape)

label = decode_predictions(yhat)
label = label[0][0]

# print the classification
print('%s (%.2f%%)' % (label[1], label[2]*100))














