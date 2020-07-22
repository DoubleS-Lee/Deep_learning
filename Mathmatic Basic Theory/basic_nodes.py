import numpy as np

# x + y = z
class plus_node:
    def __init__(self):
        self._x, self._y = None, None
        self._z = None

    def forward(self, x, y):
        self._x, self._y = x, y
        self._z = self._x + self._y
        return self._z
    
    # dJ/dx = dJ/dz*dz/dx = dJ/dz*1 이다
    # dJ/dy = dJ/dz*dz/dy = dJ/dz*1 이다
    # 여기서 dz는 dJ/dz를 뜻한다
    def backward(self, dz):
        return dz, dz


# x - y = z
class minus_node:
    def __init__(self):
        self._x, self._y = None, None
        self._z = None

    def forward(self, x, y):
        self._x, self_y = x, y
        self._z = self._x - self_y
        return self._z
    
    # dJ/dx = dJ/dz*dz/dx = dJ/dz*1 이다
    # dJ/dy = dJ/dz*dz/dy = dJ/dz*-1 이다
    def backward(self, dz):
        return dz, -1*dz


# x * y = z
class mul_node:
    def __init__(self):
        self._x, self._y = None, None
        self._z = None
    
    def forward(self, x, y):
        self._x, self._y = x, y
        self._z = self._x * self._y
        return self._z
    
    # dJ/dx = dJ/dz*dz/dx = dJ/dz*y 이다
    # dJ/dy = dJ/dz*dz/dy = dJ/dz*x 이다
    def backward(self, dz):
        return dz*self._y, dz*self._x


# x ^ 2 = z
class square_node:
    def __init__(self):
        self._x = None
        self._z = None
    
    def forward(self, x):
        self._x = x
        self._z = self._x * self._x
        return self._z
    
    # dJ/dx = dJ/dz*dz/dx = dJ/dz*(2x) 이다
    def backward(self, dz):
        return 2*dz*self._x


# (x1 + x2 + ..... + xn)/n = z
class mean_node:
    def __init__(self):
        self._x = None
        self._z = None
    
    def forward(self, x):
        self._x = x
        self._z = np.mean(self._x)
        return self._z
    
    # np.ones_like(배열, dtype=자료형)  >  배열의 크기와 동일하며 모든 원소의 값이 1인 배열을 생성할 수 있습니다
    # dJ/dxi = dJ/dz*dz/dxi = dJ/dz*(1/n) 이다
    def backward(self, dz):
        dx = dz*1/len(self._x)*np.ones_like(self._x)
        return dx

















