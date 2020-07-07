#$ 기본 합성곱 신경망 구현
import tensorflow as tf
import numpy as np

#1. 하이퍼 파라미터
EPOCHS = 10



#2. 모델 정의
class ConvNet(tf.keras.Model):
    def __init__(self):
        super(ConvNet, self).__init__()
        conv2d = tf.keras.layers.Conv2D
        maxpool = tf.keras.layers.MaxPool2D
        #히든 레이어를 담을 변수 선언
        self.sequence = list()
        #처음 입력이 28x28이다
        #채널수 = 16, 커널 = (3,3)
        #convolution layer = 6개, maxpooling layer = 2개, Flatten layer = 1개, Dense layer = 2개 선언
        self.sequence.append(conv2d(16, (3, 3), padding='same', activation='relu')) # 28x28x16 = 영상의 크기 가로 x 영상의 크기 세로 x 채널수
        self.sequence.append(conv2d(16, (3, 3), padding='same', activation='relu')) # 28x28x16
        #2x2 풀링 사용
        self.sequence.append(maxpool((2,2))) # 14x14x16
        self.sequence.append(conv2d(32, (3, 3), padding='same', activation='relu')) # 14x14x32
        self.sequence.append(conv2d(32, (3, 3), padding='same', activation='relu')) # 14x14x32
        self.sequence.append(maxpool((2,2))) # 7x7x32
        self.sequence.append(conv2d(64, (3, 3), padding='same', activation='relu')) # 7x7x64
        self.sequence.append(conv2d(64, (3, 3), padding='same', activation='relu')) # 7x7x64
        self.sequence.append(tf.keras.layers.Flatten()) # 1568 (=7x7x64)
        self.sequence.append(tf.keras.layers.Dense(128, activation='relu'))
        self.sequence.append(tf.keras.layers.Dense(10, activation='softmax'))

    #선언된 레이어들을 여기서 다 실행할수있도록 정의해줘야한다
    def call(self, x, training=False, mask=None):
        for layer in self.sequence:
            x = layer(x)
        return x



#4. 학습, 테스트 루프 정의
# Implement training loop
@tf.function
def train_step(model, images, labels, loss_object, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)

# Implement algorithm test
@tf.function
def test_step(model, images, labels, loss_object, test_loss, test_accuracy):
    predictions = model(images)

    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)



#5. 데이터셋 준비
#mnist 데이터 불러오기(실습을 위한 데이터)
mnist = tf.keras.datasets.mnist

#입력이 0~255로 표현되어 있음
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#255로 나눠서 0~1로 Nomalized를 해준다
x_train, x_test = x_train / 255.0, x_test / 255.0

# mnist 영상은 데이터가 28x28로 구성되어 있다
# x_train은 (NUM_SAMPLE(샘플 수), 28, 28) 이렇게 구성이 되어 있는데 뒤에 채널을 붙여줘야한다 (여기서는 1채널짜리 영상이므로 1을 추가)
# x_train : (NUM_SAMPLE, 28, 28) -> (NUM_SAMPLE, 28, 28, 1)
# x_train[...] : x_train에 있는 모든 axis를 표현할때 [...]를 사용한다
# 하나씩 지정할때는 [:,:,:] 이런식으로 한다
# x_train[..., tf.newaxis] : x_train axis에 tf.newaxis axis를 하나 추가해준다
# 현재 float64의 형태로 데이터가 구성되어 있는데 float32 형태로 변환해준다(tensorflow에서 에러 방지)
x_train = x_train[..., tf.newaxis].astype(np.float32)
x_test = x_test[..., tf.newaxis].astype(np.float32)

# 데이터셋에서 테스트용, 검증용 데이터를 나눠준다
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)



#6. 학습환경 정의
#6-1. 모델 생성, 손실함수, 최적화 알고리즘, 평가지표 정의
# Create model
model = ConvNet()

# Define loss and optimizer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# Define performance metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')



#7. 힉습루프 동작
for epoch in range(EPOCHS):
    for images, labels in train_ds:
        train_step(model, images, labels, loss_object, optimizer, train_loss, train_accuracy)

    for test_images, test_labels in test_ds:
        test_step(model, test_images, test_labels, loss_object, test_loss, test_accuracy)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100))
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()


