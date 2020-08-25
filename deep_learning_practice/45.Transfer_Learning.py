# Transfer Learning 실습

## Import modules

# !pip install tensorflow-datasets

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

## 하이퍼파라미터 정의

EPOCHS = 100

## 네트워크 구조 정의

def MyModel():
    feat = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                             include_top=False)
    feat.trainable = False
    
    seq = tf.keras.models.Sequential()
    seq.add(feat) # h x w x c 
    seq.add(tf.keras.layers.GlobalAveragePooling2D()) # c
    seq.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return seq

## 데이터 불러오기 (Cats vs. Dogs)
# split = tfds.Split.TRAIN.subsplit(weighted=(8, 2))
dataset, meta = tfds.load('cats_vs_dogs',
                          split=('train[:80%]', 'train[80%:]'),
                          with_info=True,
                          as_supervised=True)

train_ds, test_ds = dataset

## 데이터 확인하기
l2s = meta.features['label'].int2str
for img, label in test_ds.take(2):
    plt.figure()
    plt.imshow(img)
    plt.title(l2s(label))

## 데이터 가공하기

def preprocess(img, label):
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.image.resize(img, (224, 224))
    return img, label

train_ds = train_ds.map(preprocess).batch(32).prefetch(1024)
test_ds = test_ds.map(preprocess).batch(32).prefetch(1024)

## 모델 생성

model = MyModel()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

## 모델 학습

model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS)