import numpy as np
import tensorflow as tf

# ### Tensor 생성
# tf.constant()를 사용해서 Tensor를 생성한다
# list와 tuple 형태로 만들수 있다
tf.constant([1, 2, 3])
#print(tf.constant([1, 2, 3]))
tf.constant(((1, 2, 3), (1, 2, 3)))

# Array를 Tensor로 만들수 있다
arr = np.array([1, 2, 3])
#print(arr)

tensor = tf.constant(arr)
#print(tensor)

# ### Tensor에 담긴 정보 확인

# - shape 확인
shape=tensor.shape
#print(shape)

# - data type 확인
#     - 주의: Tensor 생성 할 때도 data type을 정해주지 않기 때문에 data type에 대한 혼동이 올 수 있음
#     - Data Type에 따라 모델의 무게나 성능 차이에도 영향을 줄 수 있음
dtype=tensor.dtype
#print(dtype)

# - data type 정의
tensor = tf.constant([1, 2, 3], dtype=tf.float32)
#print(tensor)

# - data type 변환
#     - Numpy에서 astype()을 주었듯이, TensorFlow에서는 tf.cast를 사용
# Numpy
arr = np.array([1, 2, 3], dtype=np.float32)
arr.astype(np.uint8)
#print(arr.dtype)

#Tensorflow
tensor = tf.constant([1, 2, 3], dtype=tf.float32)
tf.cast(tensor, dtype=tf.uint8)
#print(tensor)

# - Tensor에서 Numpy로 불러오기
# tensor를 담고 있는 변수 뒤에 .numpy() 를 붙여준다
tensor.numpy()


# - Tensor에서 Numpy 불러오기
# np.array() 를 사용한다
np.array(tensor)


# type()를 사용하여 numpy array로 변환된 것 확인
print(type(tensor))
print(type(tensor.numpy()))


# ## 난수 생성


# - Normal Distribution은 중심극한 이론에 의한 연속적인 모양   
# - Uniform Distribution은 중심 극한 이론과는 무관하며 불연속적이며 일정한 분포

# - numpy에서는 normal distribution을 기본적으로 생성
np.random.randn(9)
#print(np.random.randn(9))

# - tensor를 생성할때는 normal distribution인지 uniform distribution인지 결정해줄수 있음
tf.random.normal([3, 3])
#print(tf.random.normal([3, 3]))
tf.random.uniform([4, 4])
#print(tf.random.uniform([4, 4]))
