#%%
AutoEncoder

Anomaly Detection
정상적인 데이터 셋을 오토인코더로 돌려서 특징을 뽑아내고
잘 모르는 데이터 셋을 먼저 돌렸던 오토인코더 모델을 그대로 들고와서 예측만 시켜보면
 그 결과가 복원이 잘되면 정상적인 데이터셋과 같은거고 
 그 결과가 복원이 안되면 정상적이지 않은 데이터 셋이라고 할 수 있다





#%%
import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras import layers

batch_size = 128
num_classes = 10
epochs = 100

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# 모델 생성
model = Sequential()
## model add
model.add(layers.Dense(units = 512, activation='relu',input_shape=(784,)))
model.add(layers.Dense(units = 256, activation = 'relu'))
model.add(layers.Dense(units = 128, activation = 'relu'))
model.add(layers.Dense(units = 64, activation = 'relu'))
model.add(layers.Dense(units = 128, activation = 'relu'))
model.add(layers.Dense(units = 256, activation = 'relu'))
model.add(layers.Dense(units = 512, activation = 'relu'))
model.add(layers.Dense(units = 784, activation = 'sigmoid'))

model.summary()

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, x_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)

# 노이즈 섞어주기
import numpy as np
x_test = x_test+np.random.uniform(low=0.0, high=0.3, size=x_test.shape)

y_pred = model.predict(x_test)
y_pred.shape

y_pred = model.predict(x_test)

y_pred = y_pred.reshape(10000,28,28)

plt.figure()
plt.imshow(y_pred[0])
plt.show()

x_test = x_test.reshape(10000,28,28)

plt.figure()
plt.imshow(x_test[0])
plt.show()


'''
## Anomaly Detection
############# 이 밑으로는 위에서 학습한 모델을 가지고 fashion_mnist를 오토인코더로 복원한 코드
## 잘 안 맞는게 정상이다
## 따라서 이 결과를 가지고 복원이 잘되면 기존에 들어갔던 데이터와 유사한 데이터라는걸 알 수 있고
## 여기서 복원이 잘 안되면 기존에 들어갔던 데이터와 다른 데이터라는걸 알 수 있다

(x_train_1, y_train_1), (x_test_1, y_test_1) = tf.keras.datasets.fashion_mnist.load_data()

x_train_1 = x_train_1.reshape(60000, 784)
x_test_1 = x_test_1.reshape(10000, 784)
x_train_1 = x_train_1.astype('float32')
x_test_1 = x_test_1.astype('float32')
x_train_1 /= 255
x_test_1 /= 255

y_pred_1 = model.predict(x_test_1)
y_pred_1.shape
y_pred_1 = y_pred_1.reshape(10000,28,28)

plt.figure()
plt.imshow(y_pred_1[8])
plt.show()

x_test_1 = x_test_1.reshape(10000,28,28)

plt.figure()
plt.imshow(x_test_1[8])
plt.show()
'''





#%%
# K-means clustering

import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras import layers

batch_size = 128
num_classes = 10
epochs = 100

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

####################
# Kmeans clustering
import numpy as np

kmeans = KMeans(n_clusters = 10, random_state=50)
kmeans.fit(x_train)
print(kmeans.labels_[:30])
print(y_test[:30])

y_pred = kmeans.labels_

y_pred = pd.Series(y_pred)
y_test = pd.Series(y_test)


result1 = np.where(kmeans.labels_ == 0)
result2 = np.where(y_train == 1)

print(result1)

#####################
# AutoEncoder + kmeans clustering

# 모델 생성
model = Sequential()
## model add
model.add(layers.Dense(units = 512, activation='relu',input_shape=(784,)))
model.add(layers.Dense(units = 256, activation = 'relu'))
model.add(layers.Dense(units = 128, activation = 'relu'))
model.add(layers.Dense(units = 64, activation = 'relu'))
model.add(layers.Dense(units = 128, activation = 'relu'))
model.add(layers.Dense(units = 256, activation = 'relu'))
model.add(layers.Dense(units = 512, activation = 'relu'))
model.add(layers.Dense(units = 784, activation = 'sigmoid'))

model.summary()

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, x_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)

### 이미 만들어진 레이어를 떼어서 새로운 레이어를 생성하는 방법
encoder = model.layers[0:3]
Encoder = Sequential()
for layer in encoder:
    Encoder.add(layer)
###
    

x_train_ae = Encoder.predict(x_train)
kmeans_ae = KMeans(n_clusters = 10, random_state=50)
kmeans_ae.fit(x_train_ae)

print(kmeans_ae.labels_[:30])
print(kmeans.labels_[:30])
print(y_train[:30])



'''
encoder = model.layers[0:3]
encoder.output_shape
new_model = Sequential()
new_model.add(Dense(64, activation = 'relu', input_shape = encoder.output_shape))
'''




#%%
# convolution autoencoder

import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras import layers

batch_size = 4096*4
num_classes = 10
epochs = 100

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train[0].shape)

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

x_train.shape

# 모델 생성
model = Sequential()
## model add
model.add(layers.Conv2D(filters = 64, padding = 'same', kernel_size = (3,3), activation='relu', input_shape = (28,28,1)))
model.add(layers.Conv2D(filters = 32, padding = 'same', kernel_size = (3,3), activation='relu'))
model.add(layers.Conv2D(filters = 16, padding = 'same', kernel_size = (3,3), activation='relu'))
model.add(layers.Conv2D(filters = 32, padding = 'same', kernel_size = (3,3), activation='relu'))
model.add(layers.Conv2D(filters = 64, padding = 'same', kernel_size = (3,3), activation='relu'))
model.add(layers.Conv2D(filters = 1, padding = 'same', kernel_size = (3,3), activation='relu'))



model.summary()

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, x_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)

# 노이즈 섞어주기
import numpy as np
x_test = x_test+np.random.uniform(low=0.0, high=0.3, size=x_test.shape)

y_pred = model.predict(x_test)
y_pred.shape

y_pred = model.predict(x_test)

y_pred = y_pred.reshape(10000,28,28)

plt.figure()
plt.imshow(y_pred[0])
plt.show()

x_test = x_test.reshape(10000,28,28)

plt.figure()
plt.imshow(x_test[0])
plt.show()


#%%
# Denoising AutoEncoder
from tensorflow.keras import optimizers
import glob
import numpy as np
np.set_printoptions(threshold=np.inf) #...없이 출력하기
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, array_to_img, img_to_array, ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Input, BatchNormalization
np.random.seed(111)

#####################################################
# Load train, train_cleaned, test data

train = glob.glob('C:/Users/Seohee/Documents/AI_fast_track_DS40/denoising-dirty-documents/train/*.png')
train_cleaned = glob.glob('C:/Users/Seohee/Documents/AI_fast_track_DS40/denoising-dirty-documents/train_cleaned/*.png')
test = glob.glob('C:/Users/Seohee/Documents/AI_fast_track_DS40/denoising-dirty-documents/test/*.png')

print("Total number of images in the training set: ", len(train)) # Training set 144장
print("Total number of cleaned images found: ", len(train_cleaned))       # Train cleaned set 144장
print("Total number of samples in the test set: ", len(test))     # Test set 72장

#####################################################
# Load train images and train labels
epochs = 40
batch_size = 16

X = [] # 인풋에 들어가는 image 리스트 : Train data
X_target = [] # 아웃풋에 들어가는 image 리스트 : Train_cleaned data


for img in train:
    img = load_img(img, grayscale=True,target_size=(420,540))
    img = img_to_array(img).astype('float32')/255.
    X.append(img)

for img in train_cleaned:
    img = load_img(img, grayscale=True,target_size=(420,540))
    img = img_to_array(img).astype('float32')/255.
    X_target.append(img)

X = np.array(X) # image 데이터, 리스트를 numpy array로 형변환, 이렇게 하면 shape 함수 등을 쓸 수 있음
X_target = np.array(X_target) # image 데이터, 리스트를 numpy array로 형변환

print("Size of X : ", X.shape) # Size of X : (144,420,540,1) , Training set 144장
print("Size of X_target : ", X_target.shape) # Size of Y : (144,420,540,1) , Train cleaned 144장



#####################################################
# Load test images

test_list=[]
for img in test:
    img = load_img(img, grayscale=True,target_size=(420,540))
    img = img_to_array(img).astype('float32')/255.
    test_list.append(img)

test_list = np.array(test_list)
print("Size of test_list : ", test_list.shape) # Size of X : (72,420,540,1) , test set 72장

#####################################################
# Define your model

def build_autoenocder():
    input_img = Input(shape=(420, 540, 1), name='image_input')

    #### enoder
    x = Conv2D(32, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same')(input_img)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    #### decoder
    x = Conv2D(64, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(1, (3, 3), activation='sigmoid', kernel_initializer='he_normal', padding='same')(x)

    # model
    autoencoder = Model(inputs=input_img, outputs=x)

    return autoencoder


#####################################################
# Define optimizer and loss function

model = build_autoenocder()
model.compile(optimizer=optimizers.Adam(), loss='MSE') # 모델의 Optimizer와 Loss 함수 정의


#####################################################
# Train your model

hist = model.fit(X, X_target, epochs=epochs, batch_size=batch_size)


#####################################################
# Predict test images(Get denoised version of test images)

predicted_list = []
for img in test_list:
    img = np.reshape(img, (1, 420, 540, 1))
    predicted = np.squeeze(model.predict(img, batch_size=1)) # 각 모델의 prediction을 predictions 리스트에 저장
    predicted_list.append(predicted)

#####################################################
# Plot original denoised version of test image

_, ax = plt.subplots(1,2, figsize=(12,9.338))
ax[0].imshow(np.squeeze(test_list[0]), cmap='gray') # 1열에 원본 이미지
ax[1].imshow(np.squeeze(predicted_list[0].astype('float32')), cmap='gray') # 2열에 Denoised 이미지
plt.show()

#####################################################
# Save into the denoised test image files
import imageio
i = 0
for img in predicted_list:
    img = np.reshape(img, (420, 540, 1))
    imageio.imwrite('./denoising-dirty-documents/test_result/'+str(i)+'.png', img)
    i+=1


