#$순환신경망 구현 및 학습
import tensorflow as tf


#1. 하이퍼 파라미터
EPOCHS = 10
#분석을 위해 10000개의 데이터를 사용하겠다
NUM_WORDS = 10000


#2. 모델 정의
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        #임베딩 :
        self.emb = tf.keras.layers.Embedding(NUM_WORDS, 16)
        #RNN 정의(지금은 SimpleRNN이다) (LSTM은 오류가 뜬다 이거 확인할것) (Gru로도 바꿔서 해볼것)
        self.rnn = tf.keras.layers.SimpleRNN(32)
        self.dense = tf.keras.layers.Dense(2, activation='softmax')
        
    def call(self, x, training=None, mask=None):
        x = self.emb(x)
        x = self.rnn(x)
        return self.dense(x)


#3. 학습, 테스트 루프 정의
# Implement training loop
@tf.function
def train_step(model, inputs, labels, loss_object, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)

# Implement algorithm test
@tf.function
def test_step(model, images, labels, loss_object, test_loss, test_accuracy):
    predictions = model(images, training=False)

    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)


#4. 데이터 셋 준비
imdb = tf.keras.datasets.imdb
#10000개의 단어만 사용(NUM_WORDS)
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=NUM_WORDS)

x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train,
                                                       value=0,
                                                       padding='pre',
                                                       maxlen=32)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test,
                                                      value=0,
                                                      padding='pre',
                                                      maxlen=32)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


#5. 학습 환경 정의
#5-1. 모델 생성, 손실함수, 최적화 알고리즘, 평가지표 정의
# Create model
model = MyModel()

# Define loss and optimizer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# Define performance metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


#6. 학습 루프 동작
for epoch in range(EPOCHS):
    for seqs, labels in train_ds:
        train_step(model, seqs, labels, loss_object, optimizer, train_loss, train_accuracy)

    for test_seqs, test_labels in test_ds:
        test_step(model, test_seqs, test_labels, loss_object, test_loss, test_accuracy)

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






























