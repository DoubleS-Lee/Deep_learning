import pandas as pd
import numpy as np
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

data = pd.read_csv('C:/Users/Seohee/Documents/AI_fast_track_DS40/stock/AMZN.csv')

data.info()

data_input = data.iloc[:,[1,2,3,5,6]]
data_output = data.loc[:,'Close']


data_input = (data_input - data_input.mean()) / data_input.std()
data_input.mean()
data_input.std()

x = data_input.to_numpy()[:]
y = data_output.to_numpy()[:]

x_train = x[:850]
x_test = x[850:-1]
y_train = y[1:851]
y_test = y[851:]

model = models.Sequential()
model.add(layers.Dense(units = 64, activation='relu',input_shape=(5,)))
model.add(layers.Dense(units = 1, activation = 'linear'))

model.compile(optimizer='adam',
             loss='mse',
             metrics = ['acc'])

model.fit(x_train, y_train, epochs = 100, batch_size=4, verbose=1)

model.summary()

score = model.evaluate(x_test, y_test, verbose=1)

y_hat = model.predict(x_test)

#a_axis = np.arange(0, len(y_train))
#b_axis = np.arange(len(y_train), len(y_train) + len(y_hat))

plt.figure(figsize=(10,6))
plt.plot(a_axis, y_train.reshape(850,), '-')
plt.plot(b_axis, y_hat.reshape(408,), '-', color='red', label='Predicted')
plt.plot(b_axis, y_test.reshape(408,), '-', color='green', alpha=0.2, label='Actual')
plt.legend()
plt.show()




#%%
import pandas as pd
import numpy as np
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

data = pd.read_csv('C:/Users/Seohee/Documents/AI_fast_track_DS40/stock/AMZN.csv')

data.info()


high_prices = data['High'].values
low_prices = data['Low'].values
mid_prices = (high_prices + low_prices)/2

print(mid_prices.shape)

import numpy as np

seq_len = 50

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

x, y = generateX(mid_prices, seq_len)
x = x.reshape(-1,seq_len,1)
y = y.reshape(-1,1)


train_num = 1000
test_num = x.shape[0]-train_num

x_train = x[:train_num, :, :]
y_train = y[:train_num:, :]
x_test = x[train_num:, :, :]
y_test = y[train_num:, :]


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, GRU, Conv1D
import matplotlib.pyplot as plt

# RNN 모델을 생성 및 학습

xInput = Input(batch_shape=(None, x_train.shape[1], x_train.shape[2]))
x = LSTM(64, return_sequences = True)(xInput)
x = LSTM(64)(x)
x = Dense(64,activation = 'relu')(x)
x = Dense(64,activation = 'relu')(x)
xOutput = Dense(1, activation = 'linear')(x)


model = Model(xInput, xOutput)
model.compile(loss='mse', optimizer='adam')


# 학습
model.fit(x_train, y_train, epochs=500, batch_size=train_num,verbose=1)


 # 예측
y_hat = model.predict(x_test, batch_size=1)

a_axis = np.arange(0, len(y_train))
b_axis = np.arange(len(y_train), len(y_train) + len(y_hat))

plt.figure(figsize=(10,6))
plt.plot(a_axis, y_train.reshape(x_train,), '-')
plt.plot(b_axis, y_hat.reshape(test_num,), '-', color='red', label='Predicted')
plt.plot(b_axis, y_test.reshape(test_num,), '-', color='green', alpha=0.2, label='Actual')
plt.legend()
plt.show()


















