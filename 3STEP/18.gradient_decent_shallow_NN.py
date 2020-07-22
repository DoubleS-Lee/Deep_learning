#$경사하강법을 이용한 얕은 신경망 학습

import tensorflow as tf
import numpy as np

#1. 하이퍼파라미터 설정
EPOCHS = 1000

#2. 네트워크 구조 정의
#얕은 신경망
#입력계층 : 2개 입력, 은닉계층 : 128개(sigmoid activation), 출력계층 : 10개 아웃풋(Softmax activation)
class MyModel(tf.keras.Model):
    # 변수 정의
    def __init__(self):
        #상속을 했다면 상속을 한 상위클래스 tf.keras.Model을 super().__init__() 하는 것을 잊으면 안됨
        super(MyModel, self).__init__()
        self.d1 = tf.keras.layers.Dense(128, input_dim=2, activation='sigmoid')
        self.d2 = tf.keras.layers.Dense(10, activation='softmax')
    # call 메소드를 만들어서 계산을 구현        
    def call(self, x, training=None, mask=None):
        x = self.d1(x)
        return self.d2(x)

#3. 학습루프정의
#@tf.function 데코레이터 이용
#labels은 타겟, loss, optimizer는 어떤걸 쓸건지 등을 정의
@tf.function
def train_step(model, inputs, labels, loss_object, optimizer, train_loss, train_metric):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = loss_object(labels, predictions)
    #gradient 계산
    #첫번째 args인 loss를 두번째 args인 model.trainable_variables로 미분하여 gradient를 계산한다는 뜻이다
    gradients = tape.gradient(loss, model.trainable_variables) # df(x)/dx
    
    #optimizer를 하는데 gradients를 이용한다
    #(gradients, model.trainable_variables)를 zip 하여 넣어준다
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_metric(labels, predictions)

#4. 데이터셋 생성, 전처리
np.random.seed(0)
#pts는 입력값
pts = list()
#labels는 타겟값
labels = list()

center_pts = np.random.uniform(-8.0, 8.0, (10, 2))

for label, center_pt in enumerate(center_pts):
    for _ in range(100):
        pts.append(center_pt + np.random.randn(*center_pt.shape))
        labels.append(label)

pts = np.stack(pts, axis=0).astype(np.float32)
labels = np.stack(labels, axis=0)

train_ds = tf.data.Dataset.from_tensor_slices((pts, labels)).shuffle(1000).batch(32)

#5. 모델 생성
model = MyModel()

#6. 손실함수(CrossEntropy), 최적화알고리즘(Adam Optimizer) 설정
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

#7. 평가지표 설정
#Accuracy
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

#8. 학습 루프
for epoch in range(EPOCHS):
    for x, label in train_ds:
        train_step(model, x, label, loss_object, optimizer, train_loss, train_accuracy)
        
    template = 'Epoch {}, Loss: {}, Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100))
    train_loss.reset_states()
    train_accuracy.reset_states()

#9. 데이터셋 및 학습 파라미터 저장
#savez는 여러개를 한꺼번에 저장한다는 뜻
#compressed는 압축하여 저장한다는 뜻
np.savez_compressed('ch2_dataset.npz', inputs=pts, labels=labels)

#hidden layer의 weight와 bias
W_h, b_h = model.d1.get_weights()
#oupput layer의 weight와 bias
W_o, b_o = model.d2.get_weights()
#weight를 transpose를 이용하여 변환하여 저장
W_h = np.transpose(W_h)
W_o = np.transpose(W_o)
np.savez_compressed('ch2_parameters.npz',
                    W_h=W_h,
                    b_h=b_h,
                    W_o=W_o,
                    b_o=b_o)