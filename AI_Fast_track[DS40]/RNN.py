import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import time
import tensorflow as tf

 
# QT
# label_num = 5

# 파일 train_file, valid_file, test_file
# acc_strain :
# acc :
# strain :

# Mega


# QZ

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

# 데이터 로드
def data_load(file_name, input_first, input_last, label_num, sequence_input, sequence_output):
    data=pd.read_csv(file_name, sep=',', header=None, index_col=0, skiprows=[0,1,2,3,4,5,6,7], low_memory=False, dtype=float)
    data=np.array(data)
    # print(data.shape)
    # print(data)
    # print(data.shape[0])
   
    input_data_v1=data[0 : data.shape[0], input_first : input_last]

    #input_data_v1 = data[0 : data.shape[0], input_first : input_last]
    #input_data_v1 = input_data_v1[:, tf.newaxis, :].astype(np.float64)


    # print(input_data_v1)
    # print(input_data_v1.shape)
    label_data_v1=data[0 : data.shape[0], label_num : data.shape[1]]
    # print(label_data_v1)
    # print(label_data_v1.shape)

    # 데이터 차원변경
    # RNN에서의 데이터는 3차원이어야한다
    # input = (가공한 데이터 길이(=m), 예측에 사용할 이전 데이터 수, input 변수의 종류수(=feature))
    # output = (가공한 데이터 길이(=m), 예측에 사용할 이전 데이터 수, output 변수의 종류수(=feature))
    # 101개의 데이터의 데이터중 100개의 데이터로 다음 1개를 예측하겠다
    # 가공한 데이터 길이(m) = 전체 input 개수 - (예측에 사용할 이전 데이터 수 + 예측할 데이터 수) + 1

    input_feature = len(input_data_v1[0])
    output_feature = len(label_data_v1[0])
    m = len(input_data_v1) - (sequence_input + sequence_output) + 1

    input_data = np.zeros((m,sequence_input,input_feature), np.float64)
    label_data = np.zeros((m,sequence_output,output_feature), np.float64)

   
    for i in range(m):
        input_data[i] = input_data_v1[i : i + sequence_input]

    for i in range(m):
        label_data[i] = label_data_v1[i + sequence_input : i + sequence_input + sequence_output, 0 : output_feature]

    return input_data, label_data

    # print(input_data_v1.shape)

    # dataX, dataY = [], []

    # for i in range(len(input_data_v1)-sequence_input-1):
    #     for j in range(sequence_input):
    #         a = input_data_v1[i+j:(i+sequence_input+j), 0]
    #         dataX.append(a)
       

    # for i in range(len(label_data_v1)-sequence_output-1-sequence_input):
    #     a = label_data_v1[(i+sequence_input):(i+sequence_input+sequence_output), 0]
    #     dataY.append(a)

    # return np.array(dataX), np.array(dataY)


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
# plt.hist(QT_as_train_input, range=(-5,5), bins=20, histtype='bar')
# plt.hist(QT_as_valid_input, range=(-5,5), bins=20, histtype='bar')
# plt.hist(QT_as_test_input, range=(-5,5), bins=20, histtype='bar')

# plt.hist(QT_a_train_input, range=(-5,5), bins=20, histtype='bar')
# plt.hist(QT_a_valid_input, range=(-5,5), bins=20, histtype='bar')
# plt.hist(QT_a_test_input, range=(-5,5), bins=20, histtype='bar')

# plt.hist(QT_s_train_input, range=(-5,5), bins=20, histtype='bar')
# plt.hist(QT_s_valid_input, range=(-5,5), bins=20, histtype='bar')
# plt.hist(QT_s_test_input, range=(-5,5), bins=20, histtype='bar')

 

# data_axis_add(QT_as_train_input, QT_as_valid_input, QT_as_test_input)

# 모델 정의하기
from keras import models
from keras import layers
from keras import Input
from tensorflow import keras

def model_setting(file_name, train_input, train_label, valid_input, valid_label):

    model=models.Sequential()

    model.add(layers.GRU(64, input_shape=(train_input.shape[1], train_input.shape[2]), return_sequences=True))
    model.add(layers.Dense(train_label.shape[2]))
   
    print(model.summary())

    opt = keras.optimizers.Adam(lr=0.00005)

    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_squared_error','accuracy'])

    #최종 모델 훈련하기
    early_stop = keras.callbacks.EarlyStopping(monitor='mean_squared_error', patience=10)
    #t_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_images=False, update_freq='epoch')
    train_sequence=model.fit(train_input, train_label, epochs=250, batch_size=32, verbose=1, callbacks=[early_stop], shuffle=True)
    evaluate_sequence=model.evaluate(valid_input, valid_label, verbose=1, callbacks=[early_stop])
   
    model_result = model.save(f'{file_name}_RNN_model.h5')
    return model, train_sequence, evaluate_sequence


def model_predict(model, test_input, output_name, sequence_input):

    #데이터 예측
    predict_sequence=model.predict(test_input, batch_size=32, verbose=1)
    a=predict_sequence[:,sequence_input-1,:]
    predict=pd.DataFrame(a)
    predict.to_excel(output_name)


sequence_input = 1
sequence_output = 1

# QT 학습
QT_as_train_input, QT_as_train_label = data_load(file_name = 'QT_Kerb_CC_01_convert_filter_csv.csv',input_first = 1,input_last = -5,label_num = -5, sequence_input=1, sequence_output=1)
QT_as_valid_input, QT_as_valid_label = data_load(file_name = 'QT_Start_Stop_30kph_convert_filter_csv.csv',input_first = 1,input_last = -5,label_num = -5, sequence_input=1, sequence_output=1)
QT_as_test_input, QT_as_test_label = data_load(file_name = 'QT_Kerb_BG_01_convert_filter_csv.csv',input_first = 1,input_last = -5,label_num = -5, sequence_input=1, sequence_output=1)

# QT_a_train_input, QT_a_train_label = data_load(file_name = 'QT_Kerb_CC_01_convert_filter_csv.csv',input_first = 1,input_last = -17,label_num = -5)
# QT_a_valid_input, QT_a_valid_label = data_load(file_name = 'QT_Start_Stop_30kph_convert_filter_csv.csv',input_first = 1,input_last = -17,label_num = -5)
# QT_a_test_input, QT_a_test_label = data_load(file_name = 'QT_Kerb_BG_01_convert_filter_csv.csv',input_first = 1,input_last = -17,label_num = -5)

# QT_s_train_input, QT_s_train_label = data_load(file_name = 'QT_Kerb_CC_01_convert_filter_csv.csv',input_first = 25,input_last = -5,label_num = -5)
# QT_s_valid_input, QT_s_valid_label = data_load(file_name = 'QT_Start_Stop_30kph_convert_filter_csv.csv',input_first = 25,input_last = -5,label_num = -5)
# QT_s_test_input, QT_s_test_label = data_load(file_name = 'QT_Kerb_BG_01_convert_filter_csv.csv',input_first = 25,input_last = -5,label_num = -5)

QT_as_train_input, QT_as_valid_input, QT_as_test_input = data_norm(QT_as_train_input, QT_as_valid_input, QT_as_test_input)
# QT_a_train_input, QT_a_valid_input, QT_a_test_input = data_norm(QT_a_train_input, QT_a_valid_input, QT_a_test_input)
# QT_s_train_input, QT_s_valid_input, QT_s_test_input = data_norm(QT_s_train_input, QT_s_valid_input, QT_s_test_input)
# print(QT_as_train_input.shape, QT_as_train_label.shape)

# QT_as_train_input = np.reshape(QT_as_train_input, (QT_as_train_input.shape[0], 36,QT_as_train_input.shape[1]))
# QT_as_train_label = np.reshape(QT_as_train_label, (QT_as_train_label.shape[0], 5,QT_as_train_label.shape[1]))
# print(QT_as_train_input.shape, QT_as_train_label.shape)


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

QT_model_as_predict = model_predict(QT_model_as, QT_as_test_input, 'QT_as_GRU_predict.xlsx', sequence_input)
# QT_model_a_predict = model_predict(QT_model_a, QT_a_test_input, 'QT_a_GRU_predict.xlsx')
# QT_model_s_predict = model_predict(QT_model_s, QT_s_test_input, 'QT_s_GRU_predict.xlsx')

 

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
# QT_result.rows=['as_time', 'as_accuracy', "as_loss", 'as_varloss','a_time', 'a_accuracy', "a_loss", 'a_varloss','s_time', 's_accuracy', "s_loss", 's_varloss']
QT_result.to_excel("QT_RNN_result.xlsx")


plt.plot(QT_model_as_train_fit.history['accuracy'])
# plt.plot(QT_model_a_train_fit.history['accuracy'])
# plt.plot(QT_model_s_train_fit.history['accuracy'])
plt.title('QT model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['as'], loc='upper left')
plt.show()

plt.plot(QT_model_as_train_fit.history['loss'])
# plt.plot(QT_model_a_train_fit.history['loss'])
# plt.plot(QT_model_s_train_fit.history['loss'])
plt.title('QT model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['as'], loc='upper left')
plt.show()


#############################################################################################


# Mega 학습
Mega_as_train_input, Mega_as_train_label = data_load(file_name = '191129_Mega_Kerb_CC_01_SIF_convert_filter_csv.csv',input_first = 1,input_last = -8,label_num = -8, sequence_input=1, sequence_output=1)
Mega_as_valid_input, Mega_as_valid_label = data_load(file_name = '191129_Mega_Kerb_hill(33%_30%)_Stop_go_SIF_convert_filter_csv.csv',input_first = 1,input_last = -8,label_num = -8, sequence_input=1, sequence_output=1)
Mega_as_test_input, Mega_as_test_label = data_load(file_name = '191129_Mega_Kerb_BG_01_SIF_convert_filter_csv.csv',input_first = 1,input_last = -8,label_num = -8, sequence_input=1, sequence_output=1)

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

Mega_model_as_predict = model_predict(Mega_model_as, Mega_as_test_input, 'Mega_as_GRU_predict.xlsx', sequence_input)
# Mega_model_a_predict = model_predict(Mega_model_a, Mega_a_test_input, 'Mega_a_GRU_predict.xlsx')
# Mega_model_s_predict = model_predict(Mega_model_s, Mega_s_test_input, 'Mega_s_GRU_predict.xlsx')

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
# Mega_result.rows=['as_time', 'as_accuracy', "as_loss", 'as_varloss','a_time', 'a_accuracy', "a_loss", 'a_varloss','s_time', 's_accuracy', "s_loss", 's_varloss']
Mega_result.to_excel("Mega_RNN_result.xlsx")


plt.plot(Mega_model_as_train_fit.history['accuracy'])
# plt.plot(Mega_model_a_train_fit.history['accuracy'])
# plt.plot(Mega_model_s_train_fit.history['accuracy'])
plt.title('Mega model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['as'], loc='upper left')
plt.show()

plt.plot(Mega_model_as_train_fit.history['loss'])
# plt.plot(Mega_model_a_train_fit.history['loss'])
# plt.plot(Mega_model_s_train_fit.history['loss'])
plt.title('Mega model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['as'], loc='upper left')
plt.show()


####################################################################################################


# QZ 학습

QZ_as_train_input, QZ_as_train_label = data_load(file_name = '190911_QZ_L_ENG_KERB_CC01_convert_filter_csv.csv',input_first = 1,input_last = -9,label_num = -9, sequence_input=1, sequence_output=1)
QZ_as_valid_input, QZ_as_valid_label = data_load(file_name = '190916_QZ_L_ENG_KERB_START_STOP_(5_9SHIFT)_40KPH_3SET_convert_filter_csv.csv',input_first = 1,input_last = -9,label_num = -9, sequence_input=1, sequence_output=1)
QZ_as_test_input, QZ_as_test_label = data_load(file_name = '190911_QZ_L_ENG_KERB_BG01_convert_filter_csv.csv',input_first = 1,input_last = -9,label_num = -9, sequence_input=1, sequence_output=1)

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

QZ_model_as_predict = model_predict(QZ_model_as, QZ_as_test_input, 'QZ_as_GRU_predict.xlsx', sequence_input)
# QZ_model_a_predict = model_predict(QZ_model_a, QZ_a_test_input, 'QZ_a_GRU_predict.xlsx')
# QZ_model_s_predict = model_predict(QZ_model_s, QZ_s_test_input, 'QZ_s_GRU_predict.xlsx')

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
# QZ_result.rows=['as_time', 'as_accuracy', "as_loss", 'as_varloss','a_time', 'a_accuracy', "a_loss", 'a_varloss','s_time', 's_accuracy', "s_loss", 's_varloss']
QZ_result.to_excel("QZ_RNN_result.xlsx")


plt.plot(QZ_model_as_train_fit.history['accuracy'])
# plt.plot(QZ_model_a_train_fit.history['accuracy'])
# plt.plot(QZ_model_s_train_fit.history['accuracy'])
plt.title('QZ model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['as'], loc='upper left')
plt.show()

plt.plot(QZ_model_as_train_fit.history['loss'])
# plt.plot(QZ_model_a_train_fit.history['loss'])
# plt.plot(QZ_model_s_train_fit.history['loss'])
plt.title('QZ model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['as'], loc='upper left')
plt.show()