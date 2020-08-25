# CNN 학습 실습

## Import modules

import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.models import Sequential

## 하이퍼파라미터 정의

EPOCHS = 10

## 네트워크 구조 정의

def MyModel():
    return Sequential([Conv2D(32, (3, 3), padding='same', activation='relu'), # 28x28x32
                       MaxPool2D(), # 14x14x32
                       Conv2D(64, (3, 3), padding='same', activation='relu'), # 14x14x64
                       MaxPool2D(), # 7x7x64
                       Conv2D(128, (3, 3), padding='same', activation='relu'), # 7x7x128
                       Flatten(), # 6272
                       Dense(128, activation='relu'),
                       Dense(10, activation='softmax')]) # 128

## 데이터 불러오기

fashion_mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

# NHWC
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32).prefetch(2048)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32).prefetch(2048)

## 모델 생성

model = MyModel()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

## 모델 학습

model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS)