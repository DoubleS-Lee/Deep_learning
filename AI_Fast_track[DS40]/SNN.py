import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import time


 
# QT
# label_num = 5

# 파일 train_file, valid_file, test_file
# acc_strain :
# acc :
# strain :

# QT acc + strain 데이터
# train_file_name = QT_Kerb_CC_01_convert_filter_csv.csv
# valid_file_name = QT_Start_Stop_30kph_convert_filter_csv.csv
# test_file_name = QT_Kerb_BG_01_convert_filter_csv.csv
# input_first = 1
# input_last = -5
# label_num = -5

# QT acc 데이터
# train_file_name = QT_Kerb_CC_01_convert_filter_csv.csv
# valid_file_name = QT_Start_Stop_30kph_convert_filter_csv.csv
# test_file_name = QT_Kerb_BG_01_convert_filter_csv.csv
# input_first = 1
# input_last = -17
# label_num = -5

# QT strain 데이터
# train_file_name = QT_Kerb_CC_01_convert_filter_csv.csv
# valid_file_name = QT_Start_Stop_30kph_convert_filter_csv.csv
# test_file_name = QT_Kerb_BG_01_convert_filter_csv.csv
# input_first = 25
# input_last = -5
# label_num = -5

# Mega


# QZ

 

# 데이터 로드
def data_load(file_name, input_first, input_last, label_num):
    data=pd.read_csv(file_name, sep=',', header=None, index_col=0, skiprows=[0,1,2,3,4,5,6,7], low_memory=False, dtype=float)
    data=np.array(data)
    # print(data.shape)
    # print(data)
    # print(data.shape[0])
    input_data=data[0 : data.shape[0], input_first : input_last]
    # print(input_data)
    # print(input_data.shape)
    label_data=data[0 : data.shape[0], label_num : data.shape[1]]
    # print(label_data)
    # print(label_data.shape)

    return input_data, label_data


# 데이터 정규화
def data_norm(train_input, valid_input, test_input):
    mean = train_input.mean()
    train_input=train_input-mean

    #==train_input-=mean
    std = train_input.std()
    train_input/=std

    valid_input-=mean
    valid_input/=std

    test_input-=mean
    test_input/=std

    return train_input, valid_input, test_input
# plt.hist(as_train_input, range=(-5,5), bins=20, histtype='bar')
# plt.hist(as_valid_input, range=(-5,5), bins=20, histtype='bar')
# plt.hist(as_test_input, range=(-5,5), bins=20, histtype='bar')

# plt.hist(a_train_input, range=(-5,5), bins=20, histtype='bar')
# plt.hist(a_valid_input, range=(-5,5), bins=20, histtype='bar')
# plt.hist(a_test_input, range=(-5,5), bins=20, histtype='bar')

# plt.hist(s_train_input, range=(-5,5), bins=20, histtype='bar')
# plt.hist(s_valid_input, range=(-5,5), bins=20, histtype='bar')
# plt.hist(s_test_input, range=(-5,5), bins=20, histtype='bar')

# 모델 정의하기
from keras import models
from keras import layers
from tensorflow import keras

def model_setting(file_name, train_input, train_label, valid_input, valid_label):
    #Sequential은 레이어를 층층이 쌓겠다는 의미로 선언한다
    model=models.Sequential()
    #레이어를 하나씩 쌓는다
    #input_shape는 행렬의 곱을 하기 전에 행과 열을 맞춰주기 위해서 설정해 놓는듯하다
    #여기서 train_data.shape[1]은 13이다(=현재 train_data의 변수 갯수)
    #1번째 층에서 지정해줬으면 2번째 층부터는 input_shape를 지정해줄 필요없음
    model.add(layers.Dense(64, activation='relu', input_shape=(train_input.shape[1],)))
    model.add(layers.Dense(train_label.shape[1]))
    #최적화를 위해 Compile을 한다
    print(model.summary())

    opt = keras.optimizers.Adam(lr=0.00005)

    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_squared_error','accuracy'])

    #최종 모델 훈련하기
    early_stop = keras.callbacks.EarlyStopping(monitor='mean_squared_error', patience=10)
    #t_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_images=False, update_freq='epoch')
    train_sequence=model.fit(train_input, train_label, epochs=500, batch_size=32, verbose=1, callbacks=[early_stop], shuffle=True)
    evaluate_sequence=model.evaluate(valid_input, valid_label, batch_size=32, verbose=1, callbacks=[early_stop])
   
    model_result = model.save(f'{file_name}_SNN_model.h5')
    return model, train_sequence, evaluate_sequence

def model_predict(model, test_input, output_name):

    #데이터 예측
    predict_sequence=model.predict(test_input, batch_size=32, verbose=1)

    predict=pd.DataFrame(predict_sequence)
    predict.to_excel(output_name)

# def model_use():
#     pre = model.predict_classes(입력데이터셋 넣기)


##########################################################################

# QT 학습
QT_as_train_input, QT_as_train_label = data_load(file_name = 'QT_Kerb_CC_01_convert_filter_csv.csv',input_first = 1,input_last = -5,label_num = -5)
QT_as_valid_input, QT_as_valid_label = data_load(file_name = 'QT_Start_Stop_30kph_convert_filter_csv.csv',input_first = 1,input_last = -5,label_num = -5)
QT_as_test_input, QT_as_test_label = data_load(file_name = 'QT_Kerb_BG_01_convert_filter_csv.csv',input_first = 1,input_last = -5,label_num = -5)

# QT_a_train_input, QT_a_train_label = data_load(file_name = 'QT_Kerb_CC_01_convert_filter_csv.csv',input_first = 1,input_last = -17,label_num = -5)
# QT_a_valid_input, QT_a_valid_label = data_load(file_name = 'QT_Start_Stop_30kph_convert_filter_csv.csv',input_first = 1,input_last = -17,label_num = -5)
# QT_a_test_input, QT_a_test_label = data_load(file_name = 'QT_Kerb_BG_01_convert_filter_csv.csv',input_first = 1,input_last = -17,label_num = -5)

# QT_s_train_input, QT_s_train_label = data_load(file_name = 'QT_Kerb_CC_01_convert_filter_csv.csv',input_first = 25,input_last = -5,label_num = -5)
# QT_s_valid_input, QT_s_valid_label = data_load(file_name = 'QT_Start_Stop_30kph_convert_filter_csv.csv',input_first = 25,input_last = -5,label_num = -5)
# QT_s_test_input, QT_s_test_label = data_load(file_name = 'QT_Kerb_BG_01_convert_filter_csv.csv',input_first = 25,input_last = -5,label_num = -5)

QT_as_train_input, QT_as_valid_input, QT_as_test_input = data_norm(QT_as_train_input, QT_as_valid_input, QT_as_test_input)
# QT_a_train_input, QT_a_valid_input, QT_a_test_input = data_norm(QT_a_train_input, QT_a_valid_input, QT_a_test_input)
# QT_s_train_input, QT_s_valid_input, QT_s_test_input = data_norm(QT_s_train_input, QT_s_valid_input, QT_s_test_input)

as_time_1 = time.time()
QT_model_as, QT_model_as_train_fit, QT_model_as_evaluate = model_setting('QT_as', QT_as_train_input, QT_as_train_label,QT_as_valid_input, QT_as_valid_label)
as_time_2 = time.time()
as_time_3 = as_time_2 - as_time_1

# a_time_1 = time.time()
# QT_model_a, QT_model_a_train_fit, QT_model_a_evaluate  = model_setting('QT_a', QT_a_train_input, QT_a_train_label,QT_a_valid_input, QT_a_valid_label)
# a_time_2 = time.time()
# a_time_3 = a_time_2 - a_time_1

# s_time_1 = time.time()
# QT_model_s, QT_model_s_train_fit, QT_model_s_evaluate = model_setting('QT_s', QT_s_train_input, QT_s_train_label,QT_s_valid_input, QT_s_valid_label)
# s_time_2 = time.time()
# s_time_3 = s_time_2 - s_time_1

QT_model_as_predict = model_predict(QT_model_as, QT_as_test_input, 'QT_as_SNN_predict.xlsx')
# QT_model_a_predict = model_predict(QT_model_a, QT_a_test_input, 'QT_a_SNN_predict.xlsx')
# QT_model_s_predict = model_predict(QT_model_s, QT_s_test_input, 'QT_s_SNN_predict.xlsx')

QT_result = []
QT_result.append(as_time_3)
QT_result.append(QT_model_as_train_fit.history["accuracy"])
QT_result.append(QT_model_as_train_fit.history["loss"])
QT_result.append(QT_model_as_evaluate[2])

# QT_result.append(a_time_3)
# QT_result.append(QT_model_a_train_fit.history["accuracy"])
# QT_result.append(QT_model_a_train_fit.history["loss"])
# QT_result.append(QT_model_a_evaluate[2])

# QT_result.append(s_time_3)
# QT_result.append(QT_model_s_train_fit.history["accuracy"])
# QT_result.append(QT_model_s_train_fit.history["loss"])
# QT_result.append(QT_model_s_evaluate[2])
QT_result=pd.DataFrame(QT_result)
QT_result.rows=['as_time', 'as_accuracy', "as_loss", 'as_varloss','a_time', 'a_accuracy', "a_loss", 'a_varloss','s_time', 's_accuracy', "s_loss", 's_varloss']
QT_result.to_excel("QT_SNN_result.xlsx")


plt.plot(QT_model_as_train_fit.history['accuracy'])
# plt.plot(QT_model_a_train_fit.history['accuracy'])
# plt.plot(QT_model_s_train_fit.history['accuracy'])
plt.title('QT model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
# plt.legend(['as','a','s'], loc='upper left')
plt.show()

plt.plot(QT_model_as_train_fit.history['loss'])
# plt.plot(QT_model_a_train_fit.history['loss'])
# plt.plot(QT_model_s_train_fit.history['loss'])
plt.title('QT model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
# plt.legend(['as','a','s'], loc='upper left')
plt.show()


#############################################################################################


# Mega 학습
Mega_as_train_input, Mega_as_train_label = data_load(file_name = '191129_Mega_Kerb_CC_01_SIF_convert_filter_csv.csv',input_first = 1,input_last = -8,label_num = -8)
Mega_as_valid_input, Mega_as_valid_label = data_load(file_name = '191129_Mega_Kerb_hill(33%_30%)_Stop_go_SIF_convert_filter_csv.csv',input_first = 1,input_last = -8,label_num = -8)
Mega_as_test_input, Mega_as_test_label = data_load(file_name = '191129_Mega_Kerb_BG_01_SIF_convert_filter_csv.csv',input_first = 1,input_last = -8,label_num = -8)

# Mega_a_train_input, Mega_a_train_label = data_load(file_name = '191129_Mega_Kerb_CC_01_SIF_convert_filter_csv.csv',input_first = 1,input_last = -20,label_num = -8)
# Mega_a_valid_input, Mega_a_valid_label = data_load(file_name = '191129_Mega_Kerb_hill(33%_30%)_Stop_go_SIF_convert_filter_csv.csv',input_first = 1,input_last = -20,label_num = -8)
# Mega_a_test_input, Mega_a_test_label = data_load(file_name = '191129_Mega_Kerb_BG_01_SIF_convert_filter_csv.csv',input_first = 1,input_last = -20,label_num = -8)

# Mega_s_train_input, Mega_s_train_label = data_load(file_name = '191129_Mega_Kerb_CC_01_SIF_convert_filter_csv.csv',input_first = 25,input_last = -8,label_num = -8)
# Mega_s_valid_input, Mega_s_valid_label = data_load(file_name = '191129_Mega_Kerb_hill(33%_30%)_Stop_go_SIF_convert_filter_csv.csv',input_first = 25,input_last = -8,label_num = -8)
# Mega_s_test_input, Mega_s_test_label = data_load(file_name = '191129_Mega_Kerb_BG_01_SIF_convert_filter_csv.csv',input_first = 25,input_last = -8,label_num = -8)

Mega_as_train_input, Mega_as_valid_input, Mega_as_test_input = data_norm(Mega_as_train_input, Mega_as_valid_input, Mega_as_test_input)
# Mega_a_train_input, Mega_a_valid_input, Mega_a_test_input = data_norm(Mega_a_train_input, Mega_a_valid_input, Mega_a_test_input)
# Mega_s_train_input, Mega_s_valid_input, Mega_s_test_input = data_norm(Mega_s_train_input, Mega_s_valid_input, Mega_s_test_input)

as_time_1 = time.time()
Mega_model_as, Mega_model_as_train_fit, Mega_model_as_evaluate = model_setting('Mega_as', Mega_as_train_input, Mega_as_train_label,Mega_as_valid_input, Mega_as_valid_label)
as_time_2 = time.time()
as_time_3 = as_time_2 - as_time_1

# a_time_1 = time.time()
# Mega_model_a, Mega_model_a_train_fit, Mega_model_a_evaluate  = model_setting('Mega_a', Mega_a_train_input, Mega_a_train_label,Mega_a_valid_input, Mega_a_valid_label)
# a_time_2 = time.time()
# a_time_3 = a_time_2 - a_time_1

# s_time_1 = time.time()
# Mega_model_s, Mega_model_s_train_fit, Mega_model_s_evaluate = model_setting('Mega_s', Mega_s_train_input, Mega_s_train_label,Mega_s_valid_input, Mega_s_valid_label)
# s_time_2 = time.time()
# s_time_3 = s_time_2 - s_time_1

Mega_model_as_predict = model_predict(Mega_model_as, Mega_as_test_input, 'Mega_as_SNN_predict.xlsx')
# Mega_model_a_predict = model_predict(Mega_model_a, Mega_a_test_input, 'Mega_a_SNN_predict.xlsx')
# Mega_model_s_predict = model_predict(Mega_model_s, Mega_s_test_input, 'Mega_s_SNN_predict.xlsx')

Mega_result = []
Mega_result.append(as_time_3)
Mega_result.append(Mega_model_as_train_fit.history["accuracy"])
Mega_result.append(Mega_model_as_train_fit.history["loss"])
Mega_result.append(Mega_model_as_evaluate[2])

# Mega_result.append(a_time_3)
# Mega_result.append(Mega_model_a_train_fit.history["accuracy"])
# Mega_result.append(Mega_model_a_train_fit.history["loss"])
# Mega_result.append(Mega_model_a_evaluate[2])

# Mega_result.append(s_time_3)
# Mega_result.append(Mega_model_s_train_fit.history["accuracy"])
# Mega_result.append(Mega_model_s_train_fit.history["loss"])
# Mega_result.append(Mega_model_s_evaluate[2])
Mega_result=pd.DataFrame(Mega_result)
Mega_result.rows=['as_time', 'as_accuracy', "as_loss", 'as_varloss','a_time', 'a_accuracy', "a_loss", 'a_varloss','s_time', 's_accuracy', "s_loss", 's_varloss']
Mega_result.to_excel("Mega_SNN_result.xlsx")


plt.plot(Mega_model_as_train_fit.history['accuracy'])
# plt.plot(Mega_model_a_train_fit.history['accuracy'])
# plt.plot(Mega_model_s_train_fit.history['accuracy'])
plt.title('Mega model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
# plt.legend(['as','a','s'], loc='upper left')
plt.show()

plt.plot(Mega_model_as_train_fit.history['loss'])
# plt.plot(Mega_model_a_train_fit.history['loss'])
# plt.plot(Mega_model_s_train_fit.history['loss'])
plt.title('Mega model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
# plt.legend(['as','a','s'], loc='upper left')
plt.show()


####################################################################################################


# QZ 학습

QZ_as_train_input, QZ_as_train_label = data_load(file_name = '190911_QZ_L_ENG_KERB_CC01_convert_filter_csv.csv',input_first = 1,input_last = -9,label_num = -9)
QZ_as_valid_input, QZ_as_valid_label = data_load(file_name = '190916_QZ_L_ENG_KERB_START_STOP_(5_9SHIFT)_40KPH_3SET_convert_filter_csv.csv',input_first = 1,input_last = -9,label_num = -9)
QZ_as_test_input, QZ_as_test_label = data_load(file_name = '190911_QZ_L_ENG_KERB_BG01_convert_filter_csv.csv',input_first = 1,input_last = -9,label_num = -9)

# QZ_a_train_input, QZ_a_train_label = data_load(file_name = '190911_QZ_L_ENG_KERB_CC01_convert_filter_csv.csv',input_first = 1,input_last = -21,label_num = -9)
# QZ_a_valid_input, QZ_a_valid_label = data_load(file_name = '190916_QZ_L_ENG_KERB_START_STOP_(5_9SHIFT)_40KPH_3SET_convert_filter_csv.csv',input_first = 1,input_last = -21,label_num = -9)
# QZ_a_test_input, QZ_a_test_label = data_load(file_name = '190911_QZ_L_ENG_KERB_BG01_convert_filter_csv.csv',input_first = 1,input_last = -21,label_num = -9)

# QZ_s_train_input, QZ_s_train_label = data_load(file_name = '190911_QZ_L_ENG_KERB_CC01_convert_filter_csv.csv',input_first = 22,input_last = -9,label_num = -9)
# QZ_s_valid_input, QZ_s_valid_label = data_load(file_name = '190916_QZ_L_ENG_KERB_START_STOP_(5_9SHIFT)_40KPH_3SET_convert_filter_csv.csv',input_first = 22,input_last = -9,label_num = -9)
# QZ_s_test_input, QZ_s_test_label = data_load(file_name = '190911_QZ_L_ENG_KERB_BG01_convert_filter_csv.csv',input_first = 22,input_last = -9,label_num = -9)

QZ_as_train_input, QZ_as_valid_input, QZ_as_test_input = data_norm(QZ_as_train_input, QZ_as_valid_input, QZ_as_test_input)
# QZ_a_train_input, QZ_a_valid_input, QZ_a_test_input = data_norm(QZ_a_train_input, QZ_a_valid_input, QZ_a_test_input)
# QZ_s_train_input, QZ_s_valid_input, QZ_s_test_input = data_norm(QZ_s_train_input, QZ_s_valid_input, QZ_s_test_input)

as_time_1 = time.time()
QZ_model_as, QZ_model_as_train_fit, QZ_model_as_evaluate = model_setting('QZ_as', QZ_as_train_input, QZ_as_train_label,QZ_as_valid_input, QZ_as_valid_label)
as_time_2 = time.time()
as_time_3 = as_time_2 - as_time_1

# a_time_1 = time.time()
# QZ_model_a, QZ_model_a_train_fit, QZ_model_a_evaluate  = model_setting('QZ_a', QZ_a_train_input, QZ_a_train_label,QZ_a_valid_input, QZ_a_valid_label)
# a_time_2 = time.time()
# a_time_3 = a_time_2 - a_time_1

# s_time_1 = time.time()
# QZ_model_s, QZ_model_s_train_fit, QZ_model_s_evaluate = model_setting('QZ_s', QZ_s_train_input, QZ_s_train_label,QZ_s_valid_input, QZ_s_valid_label)
# s_time_2 = time.time()
# s_time_3 = s_time_2 - s_time_1

QZ_model_as_predict = model_predict(QZ_model_as, QZ_as_test_input, 'QZ_as_SNN_predict.xlsx')
# QZ_model_a_predict = model_predict(QZ_model_a, QZ_a_test_input, 'QZ_a_SNN_predict.xlsx')
# QZ_model_s_predict = model_predict(QZ_model_s, QZ_s_test_input, 'QZ_s_SNN_predict.xlsx')

QZ_result = []
QZ_result.append(as_time_3)
QZ_result.append(QZ_model_as_train_fit.history["accuracy"])
QZ_result.append(QZ_model_as_train_fit.history["loss"])
QZ_result.append(QZ_model_as_evaluate[2])

# QZ_result.append(a_time_3)
# QZ_result.append(QZ_model_a_train_fit.history["accuracy"])
# QZ_result.append(QZ_model_a_train_fit.history["loss"])
# QZ_result.append(QZ_model_a_evaluate[2])

# QZ_result.append(s_time_3)
# QZ_result.append(QZ_model_s_train_fit.history["accuracy"])
# QZ_result.append(QZ_model_s_train_fit.history["loss"])
# QZ_result.append(QZ_model_s_evaluate[2])
QZ_result=pd.DataFrame(QZ_result)
QZ_result.rows=['as_time', 'as_accuracy', "as_loss", 'as_varloss','a_time', 'a_accuracy', "a_loss", 'a_varloss','s_time', 's_accuracy', "s_loss", 's_varloss']
QZ_result.to_excel("QZ_SNN_result.xlsx")


plt.plot(QZ_model_as_train_fit.history['accuracy'])
# plt.plot(QZ_model_a_train_fit.history['accuracy'])
# plt.plot(QZ_model_s_train_fit.history['accuracy'])
plt.title('QZ model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
# plt.legend(['as','a','s'], loc='upper left')
plt.show()

plt.plot(QZ_model_as_train_fit.history['loss'])
# plt.plot(QZ_model_a_train_fit.history['loss'])
# plt.plot(QZ_model_s_train_fit.history['loss'])
plt.title('QZ model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
# plt.legend(['as','a','s'], loc='upper left')
plt.show()

 