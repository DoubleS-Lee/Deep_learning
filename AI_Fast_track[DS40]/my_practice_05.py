#%%
RNN(Recurrent neural Network)
시간적인 흐름에 따라 패턴을 가지고 있는 데이터를 사용
Simple RNN의 경우 weight가 1보다 크면 발산, 1보다 작으면 소실되는 문제가 생긴다
관련 정보와 그 정보를 사용하는 지점 사이의 거리가 멀 경우 RNN의 학습능력이 저하된다

LSTM(Long-Short Term Memory)
Long term memory : 과거의 state cell에 의해서 넘어오는 값
Short term memory : input값
forget 게이트 : 과거의 정보
input 게이트 : 현재의 정보
output 게이트
cell state : 현재의 입력과 과거의 입력을 얼마나 사용할지 알려줌






#%%
# Many to one RNN

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, SimpleRNN
import numpy as np


x = np.array([[[1.], [2.], [3.], [4.], [5.]]])
y = np.array([[6.]])

# batch_shape = (데이터 사이즈(아직 모름), time step, feature 수)
xInput = Input(batch_shape=(None, 5, 1))
# 히든레이어에 노드가 3개
xLstm = SimpleRNN(3)(xInput)
xOutput = Dense(1)(xLstm)


model = Model(xInput, xOutput)
model.compile(loss='mean_squared_error', optimizer='adam')

print(model.summary())

model.fit(x, y, epochs=5000, batch_size=1, verbose=0)
model.predict(x)

#%%
# Many to Many
# return_sequences=True : RNN의 중간 스텝의 출력을 모두 사용
# TimeDistributed() : 각 스텝마다 cost (오류)를 계산해서 하위 스텝으로 오류를 전파하여 각 weight를 업데이트함

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, SimpleRNN, TimeDistributed
import numpy as np


x = np.array([[[1.], [2.], [3.], [4.], [5.]]])
y = np.array([[[2.], [3.], [4.], [5.], [6.]]])

xInput = Input(batch_shape=(None, 5, 1))
xLstm = SimpleRNN(3, return_sequences=True)(xInput)
xOutput = TimeDistributed(Dense(1))(xLstm)
model = Model(xInput, xOutput)
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary)

xInput = Input(batch_shape=(None, 5, 1))
xLstm = SimpleRNN(3, return_sequences=True)(xInput)
xOutput = Dense(1)(xLstm)
model = Model(xInput, xOutput)
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())

model.fit(x, y, epochs=50, batch_size=1, verbose=0)
model.predict(x)

#%%
# Multi-layer RNN
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM
import numpy as np


x = np.array([[[1.], [2.], [3.], [4.], [5.]]])
y = np.array([[6.]])


xInput = Input(batch_shape=(None, 5, 1))
xLstm_1 = LSTM(3, return_sequences=True)(xInput)
xLstm_2 = LSTM(3)(xLstm_1)
xOutput = Dense(1)(xLstm_2)


model = Model(xInput, xOutput)
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())

model.fit(x, y, epochs=50, batch_size=1, verbose=0)
model.predict(x)


#%%
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, SimpleRNN
import numpy as np
import matplotlib.pyplot as plt



 # 학습 데이터를 생성한다.
 # ex: data = [1,2,3,4,5,6,7,8,9,10]가 주어졌을 때 generateX(data, 5)를 실행하면
 # 아래와 같은 학습데이터 변환한다.
 #
 # x                      y
 # ---------              -
 # 1,2,3,4,5              6
 # 2,3,4,5,6              7
 # 3,4,5,6,7              8
 # ...
def generateX(a, n):
    x_train = []
    y_train = []
    for i in range(len(a)):
        x = a[i:(i + n)]
        if (i + n) < len(a):
            x_train.append(x)
            y_train.append(a[i + n])
        else:
            break
    return np.array(x_train), np.array(y_train)


# Sine 함수에 노이즈를 섞은 데이터로 학습 데이터 100개를 생성한다
data = np.sin(2 * np.pi * 0.03 * np.arange(0, 100)) + np.random.random(100)
x, y = generateX(data, 10)
x = x.reshape(-1,10,1)
y = y.reshape(-1,1)


 # 학습용 데이터와 시험용 데이터로 구분
x_train = x[:70, :, :]
y_train = y[:70:, :]
x_test = x[70:, :, :]
y_test = y[70:, :]


# RNN 모델을 생성 및 학습

# batch_shape = (데이터 사이즈(아직 모름), time step, feature 수)
xInput = Input(batch_shape=(None, 10, 1))
# ============ xInput = Input(shape=(10, 1))
xLstm = SimpleRNN(10, return_sequences=True)(xInput)
xLstm = SimpleRNN(10)(xLstm)
xOutput = Dense(5)(xLstm)
xOutput = Dense(1)(xLstm)

model = Model(xInput, xOutput)
model.compile(loss='mean_squared_error', optimizer='adam')

print(model.summary())

model.fit(x_train, y_train, epochs=10, batch_size=1, verbose=0)
y_hat = model.predict(x_test)


a_axis = np.arange(0, len(y_train))
b_axis = np.arange(len(y_train), len(y_train) + len(y_hat))

plt.figure(figsize=(10,6))
plt.plot(a_axis, y_train.reshape(70,), 'o-')
plt.plot(b_axis, y_hat.reshape(20,), 'o-', color='red', label='Predicted')
plt.plot(b_axis, y_test.reshape(20,), 'o-', color='green', alpha=0.2, label='Actual')
plt.legend()
plt.show()



#%%
# LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM
import numpy as np
import matplotlib.pyplot as plt



 # 학습 데이터를 생성한다.
 # ex: data = [1,2,3,4,5,6,7,8,9,10]가 주어졌을 때 generateX(data, 5)를 실행하면
 # 아래와 같은 학습데이터 변환한다.
 #
 # x                      y
 # ---------              -
 # 1,2,3,4,5              6
 # 2,3,4,5,6              7
 # 3,4,5,6,7              8
 # ...
def generateX(a, n):
    x_train = []
    y_train = []
    for i in range(len(a)):
        x = a[i:(i + n)]
        if (i + n) < len(a):
            x_train.append(x)
            y_train.append(a[i + n])
        else:
            break
    return np.array(x_train), np.array(y_train)


# Sine 함수에 노이즈를 섞은 데이터로 학습 데이터 100개를 생성한다
data = np.sin(2 * np.pi * 0.03 * np.arange(0, 100)) + np.random.random(100)
x, y = generateX(data, 10)
x = x.reshape(-1,10,1)
y = y.reshape(-1,1)


 # 학습용 데이터와 시험용 데이터로 구분
x_train = x[:70, :, :]
y_train = y[:70:, :]
x_test = x[70:, :, :]
y_test = y[70:, :]


# RNN 모델을 생성 및 학습

# batch_shape = (데이터 사이즈(아직 모름), time step, feature 수)
xInput = Input(batch_shape=(None, 10, 1))
# ============ xInput = Input(shape=(10, 1))
xLstm = LSTM(10, return_sequences=True)(xInput)
xLstm = LSTM(10)(xLstm)
xOutput = Dense(5)(xLstm)
xOutput = Dense(1)(xLstm)

model = Model(xInput, xOutput)
model.compile(loss='mean_squared_error', optimizer='adam')

print(model.summary())

model.fit(x_train, y_train, epochs=10, batch_size=1, verbose=0)
y_hat = model.predict(x_test)


a_axis = np.arange(0, len(y_train))
b_axis = np.arange(len(y_train), len(y_train) + len(y_hat))

plt.figure(figsize=(10,6))
plt.plot(a_axis, y_train.reshape(70,), 'o-')
plt.plot(b_axis, y_hat.reshape(20,), 'o-', color='red', label='Predicted')
plt.plot(b_axis, y_test.reshape(20,), 'o-', color='green', alpha=0.2, label='Actual')
plt.legend()
plt.show()

#%%
# GRU
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GRU
import numpy as np
import matplotlib.pyplot as plt



 # 학습 데이터를 생성한다.
 # ex: data = [1,2,3,4,5,6,7,8,9,10]가 주어졌을 때 generateX(data, 5)를 실행하면
 # 아래와 같은 학습데이터 변환한다.
 #
 # x                      y
 # ---------              -
 # 1,2,3,4,5              6
 # 2,3,4,5,6              7
 # 3,4,5,6,7              8
 # ...
def generateX(a, n):
    x_train = []
    y_train = []
    for i in range(len(a)):
        x = a[i:(i + n)]
        if (i + n) < len(a):
            x_train.append(x)
            y_train.append(a[i + n])
        else:
            break
    return np.array(x_train), np.array(y_train)


# Sine 함수에 노이즈를 섞은 데이터로 학습 데이터 100개를 생성한다
data = np.sin(2 * np.pi * 0.03 * np.arange(0, 100)) + np.random.random(100)
x, y = generateX(data, 10)
x = x.reshape(-1,10,1)
y = y.reshape(-1,1)


 # 학습용 데이터와 시험용 데이터로 구분
x_train = x[:70, :, :]
y_train = y[:70:, :]
x_test = x[70:, :, :]
y_test = y[70:, :]


# RNN 모델을 생성 및 학습

# batch_shape = (데이터 사이즈(아직 모름), time step, feature 수)
xInput = Input(batch_shape=(None, 10, 1))
# ============ xInput = Input(shape=(10, 1))
xLstm = GRU(10, return_sequences=True)(xInput)
xLstm = GRU(10)(xLstm)
xOutput = Dense(5)(xLstm)
xOutput = Dense(1)(xLstm)

model = Model(xInput, xOutput)
model.compile(loss='mean_squared_error', optimizer='adam')

print(model.summary())

model.fit(x_train, y_train, epochs=10, batch_size=1, verbose=0)
y_hat = model.predict(x_test)


a_axis = np.arange(0, len(y_train))
b_axis = np.arange(len(y_train), len(y_train) + len(y_hat))

plt.figure(figsize=(10,6))
plt.plot(a_axis, y_train.reshape(70,), 'o-')
plt.plot(b_axis, y_hat.reshape(20,), 'o-', color='red', label='Predicted')
plt.plot(b_axis, y_test.reshape(20,), 'o-', color='green', alpha=0.2, label='Actual')
plt.legend()
plt.show()

#%%
# GRU + Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GRU, Conv1D
import numpy as np
import matplotlib.pyplot as plt



 # 학습 데이터를 생성한다.
 # ex: data = [1,2,3,4,5,6,7,8,9,10]가 주어졌을 때 generateX(data, 5)를 실행하면
 # 아래와 같은 학습데이터 변환한다.
 #
 # x                      y
 # ---------              -
 # 1,2,3,4,5              6
 # 2,3,4,5,6              7
 # 3,4,5,6,7              8
 # ...
def generateX(a, n):
    x_train = []
    y_train = []
    for i in range(len(a)):
        x = a[i:(i + n)]
        if (i + n) < len(a):
            x_train.append(x)
            y_train.append(a[i + n])
        else:
            break
    return np.array(x_train), np.array(y_train)


# Sine 함수에 노이즈를 섞은 데이터로 학습 데이터 100개를 생성한다
data = np.sin(2 * np.pi * 0.03 * np.arange(0, 100)) + np.random.random(100)
x, y = generateX(data, 10)
x = x.reshape(-1,10,1)
y = y.reshape(-1,1)


 # 학습용 데이터와 시험용 데이터로 구분
x_train = x[:70, :, :]
y_train = y[:70:, :]
x_test = x[70:, :, :]
y_test = y[70:, :]


# RNN 모델을 생성 및 학습

# batch_shape = (데이터 사이즈(아직 모름), time step, feature 수)
xInput = Input(batch_shape=(None, 10, 1))
# ============ xInput = Input(shape=(10, 1))
xLstm = Conv1D(filters = 5, kernel_size = 3, activation='relu')(xInput)
xLstm = GRU(10, return_sequences=True)(xLstm)
xLstm = GRU(10)(xLstm)
xOutput = Dense(5)(xLstm)
xOutput = Dense(1)(xLstm)

model = Model(xInput, xOutput)
model.compile(loss='mean_squared_error', optimizer='adam')

print(model.summary())

model.fit(x_train, y_train, epochs=10, batch_size=1, verbose=0)
y_hat = model.predict(x_test)


a_axis = np.arange(0, len(y_train))
b_axis = np.arange(len(y_train), len(y_train) + len(y_hat))

plt.figure(figsize=(10,6))
plt.plot(a_axis, y_train.reshape(70,), 'o-')
plt.plot(b_axis, y_hat.reshape(20,), 'o-', color='red', label='Predicted')
plt.plot(b_axis, y_test.reshape(20,), 'o-', color='green', alpha=0.2, label='Actual')
plt.legend()
plt.show()


#%%
#호흡기 질환 사망자수 데이터
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GRU, Conv1D
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

data = sm.datasets.get_rdataset("deaths", "MASS")
df = data.data
df.info()
df.head()
df

df.value = df.value.map(lambda x : (x - df.value.mean()) / np.std(df.value))
df.value.plot()
df.head()

def generateX(a, n):
    x_train = []
    y_train = []
    for i in range(len(a)):
        x = a[i:(i + n)]
        if (i + n) < len(a):
            x_train.append(x)
            y_train.append(a[i + n])
        else:
            break
    return np.array(x_train), np.array(y_train)

x, y = generateX(df.value, 6)

x = x.reshape(-1,6,1)
y = y.reshape(-1,1)

x_train = x[:50,:,:]
x_test = x[50:,:,:]

y_train = y[:50,:]
y_test = y[50:,:]


# batch_shape = (데이터 사이즈(아직 모름), time step, feature 수)
xInput = Input(batch_shape=(None, 6, 1))
# ============ xInput = Input(shape=(10, 1))
xLstm = Conv1D(filters = 5, kernel_size = 3, activation='relu')(xInput)
xLstm = GRU(10, return_sequences=True)(xLstm)
xLstm = GRU(10)(xLstm)
xOutput = Dense(5)(xLstm)
xOutput = Dense(1)(xLstm)

model = Model(xInput, xOutput)
model.compile(loss='mean_squared_error', optimizer='adam')

print(model.summary())

model.fit(x_train, y_train, epochs=10, batch_size=1, verbose=0)
y_hat = model.predict(x_test)


a_axis = np.arange(0, len(y_train))
b_axis = np.arange(len(y_train), len(y_train) + len(y_hat))

plt.figure(figsize=(10,6))
plt.plot(a_axis, y_train.reshape(50,), 'o-')
plt.plot(b_axis, y_hat.reshape(16,), 'o-', color='red', label='Predicted')
plt.plot(b_axis, y_test.reshape(16,), 'o-', color='green', alpha=0.2, label='Actual')
plt.legend()
plt.show()











