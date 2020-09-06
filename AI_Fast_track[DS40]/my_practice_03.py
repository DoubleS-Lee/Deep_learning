#%%
# adaline using keras
from tensorflow.keras import models, layers
import numpy as np

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

T = np.array([0, 0, 0, 1])

model = models.Sequential()
model.add(layers.Dense(units = 4, use_bias = False, activation='sigmoid',input_shape=(2,)))
model.add(layers.Dense(units = 1, activation = 'sigmoid'))


model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics = ['acc'])

model.fit(X,T,epochs = 10000,batch_size=4)


# test
Xtest_loss,Xtest_acc = model.evaluate(X,T)
print(Xtest_acc)
Ttest = model.predict(X)
print(Ttest)


#%%
# adaline using keras
from tensorflow.keras import models, layers
import numpy as np

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

T = np.array([0, 0, 0, 1])

model = models.Sequential()
model.add(layers.Dense(units = 4, use_bias = False, activation='sigmoid',input_shape=(2,)))
model.add(layers.Dense(units = 1, activation = 'sigmoid'))

def custom_loss(y_true, y_pred):
    loss = np.abs(y_true, y_pred)
    return loss


model.compile(optimizer='adam',
             loss = custom_loss,
             metrics = ['acc'])

model.fit(X, T,epochs = 1000, batch_size=4)


# test
Xtest_loss,Xtest_acc = model.evaluate(X,T)
print(Xtest_acc)
Ttest = model.predict(X)
print(Ttest)

#%%
# adaline using keras
from tensorflow.keras import models, layers
import numpy as np

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

T = np.array([0, 0, 0, 1])

model = models.Sequential()
model.add(layers.Dense(units = 4, activation='sigmoid',input_shape=(2,)))
model.add(layers.Dense(units = 1, activation = 'sigmoid'))

model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics = ['acc'])

model.fit(X, T, epochs = 10000, batch_size=4)

model.summary()

# test
Xtest_loss,Xtest_acc = model.evaluate(X,T)
print(Xtest_acc)
Ttest = model.predict(X)
print(Ttest)


#%%
# keras model API
# adaline using keras
from tensorflow.keras import models, layers, Input, Model
import numpy as np

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

T = np.array([0, 1, 1, 0])

inputs = Input(shape=(2,))
x = layers.Dense(units = 4, activation = "sigmoid")(inputs)
ouputs = layers.Dense(1, activation = 'sigmoid')(x)
model = Model(inputs = inputs, outputs = outputs, name = 'simple_models')

model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics = ['acc'])

model.fit(X,T,epochs = 10000,batch_size=4)

model.summary()

#%%
# multilayer perceptron
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

alpha = 0.1
dataset_num = 4
W1 = np.random.random((4,3))
W2 = np.random.random((1,4))
training_endnum  = 10000

X = np.array([[0, 0, 1],
             [0, 1, 1],
             [1, 0, 1],
             [1, 1, 1]])
T = np.array([0, 1, 1, 0])

for epoch in range(0,training_endnum) :
    for i in range(0,dataset_num) :
        x = np.transpose(X[i:i+1,:])
        v1 = np.dot(W1,x)
        y1 =  sigmoid(v1)
        v =  np.dot(W2,y1)
        y =  sigmoid(v)
       
        e =  T[i]-y
        delta = y*(1-y)*e
       
        e1 = np.transpose(W2)*delta
        delta1 =  y1*(1-y1)*e1
              
        dW1 = alpha*np.dot(delta1,x.T)
        W1 = W1 + dW1
       
        dW2 = alpha*np.dot(delta, np.transpose(y1))
        W2 = W2 + dW2
       
Y = []
for i in range(dataset_num) :
    x = np.transpose(X[i:i+1,:])
    v1 = np.dot(W1,x) # 4*1
    y1 = sigmoid(v1) # 4*1
    v = np.dot(W2,y1) # 1*1
    y = sigmoid(v) # 1*1
    print(y) 

#%%
# multilayer perceptron
from tensorflow.keras import models, layers
import numpy as np

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

T = np.array([0, 1, 1, 0])

model = models.Sequential()
model.add(layers.Dense(units = 4, activation='sigmoid',input_shape=(2,)))
model.add(layers.Dense(units = 4, activation = 'sigmoid'))
model.add(layers.Dense(units = 1, activation = 'sigmoid'))

model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics = ['acc'])

model.fit(X,T,epochs = 10000, batch_size=4)

# test
Xtest_loss,Xtest_acc = model.evaluate(X,T)
print(Xtest_acc)
Ttest = model.predict(X)
print(Ttest)

model.weights

#%%
# multilayer perceptron
from tensorflow.keras import models, layers, Input, Model
import numpy as np

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

T = np.array([0, 1, 1, 0])

inputs = Input(shape=(2,))
x = layers.Dense(units = 4, activation = "sigmoid")(inputs)
x1 = layers.Dense(units = 4, activation = "sigmoid")(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x1)
model = Model(inputs = inputs, outputs = outputs, name = 'simple_models')

model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics = ['acc'])

model.fit(X,T,epochs = 10000, batch_size=4)

# test
Xtest_loss,Xtest_acc = model.evaluate(X,T)
print(Xtest_acc)
Ttest = model.predict(X)
print(Ttest)

model.weights


#%%
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
import tensorflow.keras
from tensorflow.keras import models, layers, Input, Model
import matplotlib.pyplot as plt

batch_size = 4096
num_classes = 10
epochs = 10

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

plt.figure()
plt.imshow(x_train[0])
plt.show()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


y_train

#y_train과 y_test를 원핫인코딩으로 바꿔준다
# convert class vectors to binary class matrices
y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)

# 모델 생성
model = Sequential()
## model add
model.add(layers.Dense(units = 1028, activation='relu',input_shape=(784,)))
model.add(layers.Dense(units = 1028, activation = 'relu'))
model.add(layers.Dense(units = 10, activation = 'softmax'))


model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

 

# 학습한 모델을 저장하고 다시 불러와서 재학습 시키는 방법
'''
model.save('my_model.h5')


new_model = tensorflow.keras.models.load_model('my_model.h5')

new_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = new_model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))


new_model.add(layers.Dense(10, activation='softmax', input_shape=(784,)))

'''

#%%
# keras Merging layers 검색 고고
# 네트워크 모델 분기
# 네트워크 구조를 병합, 나누기 등의 작업을 할 수 있다

tf.keras.layers.Concatenate(axis=-1, **kwargs)







#%%
# 네트워크 모델 그래프화
# graphviz

from tensorflow.keras.utils import plot_model




#%% 64*3개 128*2개 10개
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
import tensorflow.keras
from tensorflow.keras import models, layers, Input, Model
import matplotlib.pyplot as plt
import tensorflow as tf

batch_size = 4096
num_classes = 10
epochs = 10

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

plt.figure()
plt.imshow(x_train[1])
plt.show()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

a=range(len(x_train[0][0]))

x_train.shape[0]
x_train.shape[1]
x_train.shape[2]

layer = []
inp = Input(shape=(x_train.shape[1],))
print(inp)
inputs=[]

for i in range(x_train.shape[1]):
    x = layers.Dense(units = 64, activation = "sigmoid")(inp)
    x = layers.Dense(units = 64, activation = "sigmoid")(x)
    x = layers.Dense(units = 64, activation = "sigmoid")(x)
    layer.append(x)
    inputs.append(inp)

print(inputs)

len(inputs)

x.shape
# x와 같은 크기의 텐서를 만들어준다
x = tf.zeros_like(x)
x.shape

# layer 리스트에 Dense들이 하나씩 들어가있다
print(layer)
layer

for k in layer:
    x = layers.Add()([x,k])
# 위의 for문과 같은 코드 x = layers.Add(layer)

x = layers.Dense(units = 128, activation = 'sigmoid')(x)
x = layers.Dense(units = 128, activation = 'sigmoid')(x)
outputs = layers.Dense(10, activation = 'softmax')(x)

model = Model(inputs = inputs, outputs = outputs, name = 'MNIST ADD Model')

model.summary()


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

history.history

history.history['loss']

plt.plot(history.history['loss'])

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])






#%%

import numpy as np
np.array([1,2,3]).shape
np.array([[1,2,3]]).shape

import tensorflow.keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf

tf.debugging.set_log_device_placement(True)

batch_size = 32
num_classes = 10
epochs = 100
num_predictions = 20

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

for i in range(4):
    print(x_train[i].shape)

for i in range(4):
    print(x_train.shape[i])


#%%
# 심층신경망

#%%
# callback 함수 및 early stopping
# 학습결과 validation의 loss와 train의 loss를 비교하고 validation loss가 증가하는 경우 학습을 중단시켜야한다
# 드랍아웃, BatchNormalization

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
import tensorflow.keras
from tensorflow.keras import models, layers, Input, Model, regulization
import matplotlib.pyplot as plt

batch_size = 4096
num_classes = 10
epochs = 10

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

plt.figure()
plt.imshow(x_train[0])
plt.show()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


y_train

#y_train과 y_test를 원핫인코딩으로 바꿔준다
# convert class vectors to binary class matrices
y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)

# 모델 생성
model = Sequential()
## model add
model.add(layers.Dense(units = 1028, activation='relu',input_shape=(784,)), kernel_regularizer = regularizers.l1_12(l1=1e-5, l2=1e-4))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.2))
model.add(layers.Dense(units = 1028, activation = 'relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(units = 10, activation = 'softmax'))


model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

checkpoint_filepath = 'checkpoint/{epoch}-{val_loss:.2f}-{val_acc:.2f}.h5'
model_checkpoint_callback = tensorflow.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, 
                                                                       save_weights_only=True, 
                                                                       monitor = 'val_acc', 
                                                                       save_best_only = True)

early_stopping = tensorflow.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5)


history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.2, 
                    callbacks = [model_checkpoint_callback, early_stopping])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

 




#%%
Overfitting을 줄여주기 위한 방법
- dropout : 학습시 노드를 번갈아가면서 꺼주면서 weight를 업데이트 한다, 
          : 모든 뉴런들이 출력을 다 내보내고 있으면 오버피팅이 되어간다
          : 각 노드의 독립성을 키워줘서 몇몇 출력이 없더라도 성능을 유지하게 만드는 방법
          
- batch normalization (각 레이어를 거치고 나온 데이터가 정규화가 안되어 있는 문제가 발생) : 매 레이어 뒤에 정규화 과정을 다시 거쳐준다
- weight regulization(정규화) : weight와 bias를 조정 분산과 편향의 싸움
                                weight를 줄여준다, 출력을 줄여준다  
                                sigmoid등의 activation func을 사용하면 결과값이 0 아니면 1에 엄청 몰려있고, 나머지 0~1사이에는 결과값이 거의 없는 불균형한 구조를 띄게 만든다

- xavier initialization : bias와 weight를 초기화하는 것



nvidia - cuda, 
tensorflow-gpu



Dense layer = fully-connected layer





























