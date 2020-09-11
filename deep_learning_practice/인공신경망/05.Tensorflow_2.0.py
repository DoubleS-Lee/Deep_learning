# Tensorflow 2.0의 이해

## Import modules

import tensorflow as tf

## 데이터 불러오기

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

## 네트워크 구조 정의

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                                    tf.keras.layers.Dense(128, activation='relu'),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(10, activation='softmax')])

## Keras 모델 Compile

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

## 학습 수행

model.fit(x_train, y_train, epochs=5)

## 학습 결과 테스트

model.evaluate(x_test, y_test)