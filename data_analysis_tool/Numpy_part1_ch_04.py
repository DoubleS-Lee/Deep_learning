import numpy as np

## 1. array 선언
arr1 = np.array([1,2,3,4], dtype=int)
# print(type(arr1))

mylist1_1 = [1,2,3,4]
mylist1_2 = [[1,2,3,4],[5,6,7,8]]

# array에서는 데이터 타입이 단일 타입이어야 한다

## 2. slicing, index
# 1d array
arr2 = np.array([0,1,2,3,4,5,6,7,8,9])
# print(arr2.shape)

# index(색인)
# print(arr2[0])
# print(arr2[5])
# print(arr2[-1])
# array의 개수를 넘어가면 에러가 뜬다(아래 코드)
#print(arr2[-11])

# slice index(범위 색인)
# print(arr2[1:])
# print(arr2[1:5])
# print(arr2[:-1])

# 2d array
arr2d2_1=np.array([[1,2,3,4],
                [5,6,7,8],
                [9,10,11,12]])

# print(arr2d2_1[0,2])
# print(arr2d2_1[2,1])

# row(행)을 모두  가져오려는 경우
# print(arr2d2_1[0,:])

# column(열)을 모두 가져오려는 경우
# print(arr2d2_1[:,2])

# 부분적으로 가져오려는 경우
# print(arr2d2_1[:2,:])
# print(arr2d2_1[:2,2:])

# fancy indexing(범위가 아닌 특정 index의 집합의 값을 선택하여 추출하고 싶을 때 활용한다)
# 반드시 [추출하고 싶은 인덱스] 꺾쇠 괄호로 묶어줘야한다
arr2 = np.array([0,10,215,36,44,58,67,76,58,49])
# print(arr2[[1,3,5]])
# idx2 = [1,3,5]
# print(arr2[idx2])

# # 2차원 fancy indexing
# print(arr2d2_1[[0,1],:])
# print(arr2d2_1[:,[1,2,3]])

# boolean indexing(조건 필터링을 통하여 boolean 값을 이용한 색인)
# 조건을 넣고 값이 True인 것들만 출력해준다
# 사용법 : array명[조건필터]
# 조건 필터를 걸어서 사용한다
arr2 = np.array([0,10,215,36,44,58,67])
arr2d2_2=np.array([[1,2,3,4],
                [5,6,7,8],
                [9,10,11,12]])

myTrueFalse = [True, False, True, False, True, False, True]
#print(arr2[myTrueFalse])

# 조건 필터
# 조건 연산자 활용가능
# 필터를 활용하여 필터에 True 조건만 색인
#print(arr2d2_2 > 2)
# 꺾쇠 괄호로 한번 더 묶어줘야함, 1차원 array로 반환한다
#print(arr2d2_2[ arr2d2_2>2 ])
#print(arr2d2_2[ arr2d2_2<5 ])


## 3. arrange(numpy 문법), range(python 문법)
# arange : 리스트에 값 생성하기, 이상 미만의 값으로 들어간다
arr3_1 = np.arange(1,11)
#print(arr3_1)

# keyword 인자 사용하여 만들기
# keyword를 지정해줌으로써 순서없이 지정해줄수 있다
arr3_2 = np.arange(start=1, stop=11)
#print(arr3_2)

# 홀수 값만 생성
arr3_3 = np.arange(1, 11, 2)
#print(arr3_3)
arr3_4 = np.arange(start=1, stop=11, step=2)
#print(arr3_4)

#range : 범위를 지정해줌
for i in range(1,11):
#    print(i)
    i

for i in range(1,11,2):
#    print(i)
    i


## 4. sort(정렬)
arr4_1 = np.array([1,10,9,8,7,2,5,3,4,6])
#print(arr4_1)
# 오름차순 정렬
#print(np.sort(arr4_1))
# 내림차순 정렬은  [::-1] 를 추가한다
#print(np.sort(arr4_1)[::-1])
# 위와 같은 방법으로 sort만 해주면 정렬된 상태가 유지되어 있지 않다
# 변수에 담아주거나 다음과 같이 써주면 정렬된 상태로 저장되게 된다
arr4_1.sort()
#print(arr4_1)

# N차원 정렬
arr2d4_1=np.array([[5,6,7,8],
                    [4,3,2,1],
                    [10,9,12,11]])
#print(arr2d4_1)
# 열 정렬(왼쪽에서 오른쪽으로)
# axis=0 은 행, axis=1 은 열, axis=2은 3차원 깊이 를 나타낸다
colunm_sort = np.sort(arr2d4_1, axis=1)
#print(colunm_sort)

# 행 정렬(위에서 아래로)
# axis=0 은 행, axis=1 은 열, axis=2은 3차원 깊이 를 나타낸다
row_sort = np.sort(arr2d4_1, axis=0)
#print(row_sort)



## 5. argsort(인덱스를 반환한다)
# 정렬된 값을 반환하는 것이 아닌 index를 반환한다
arr2d5_1=np.array([[5,6,7,8],
                    [4,3,2,1],
                    [10,9,12,11]])
# print(arr2d5_1)

# 해당 배열의 index를 반환한다
# 열 반환(왼쪽에서 오른쪽으로)
colunm_argsort = np.argsort(arr2d5_1, axis=1)
# print(colunm_argsort)

# 행 반환(위에서 아래로)
row_argsort = np.argsort(arr2d5_1, axis=0)
# print(row_argsort)


## 6. maxrix
# 덧셈, 뺄셈, 곱셈
# 덧셈, 뺄셈 : 2개의 matrix의 shape이 같아야한다(=row와 column의 차원이 같아야한다)
a6_1 = np.array([[1,2,3],
                [2,3,4]])
b6_1 = np.array([[3,4,5],
                [1,2,3]])
c6_1 = a6_1 + b6_1
# print(c6_1)

c6_2 = a6_1 - b6_1
# print(c6_2)

# sum (행렬 안에서 계산을 수행할 경우)
a6_1 = np.array([[1,2,3],
                [2,3,4]])
# 행끼리 더함
c6_3 = np.sum(a6_1, axis=0)
# print(c6_3)
# 열끼리 더함
c6_4 = np.sum(a6_1, axis=1)
# print(c6_4)

# 곱셈
# 곱셈은 6-1 일반곱셈, 6-2 행렬곱셈이 있다
# 곱셈은 앞 matrix의 뒷 차원과 뒤 matrix의 앞 차원이 같아야한다              i x t * t x z = i x z
# 6-1. 일반곱셈 (각 요소끼리 곱하는 일반 곱셈)
a6_2 = np.array([[1,2,3],
                [2,3,4]])
b6_2 = np.array([[3,4,5],
                [1,2,3]])
# print(a6_2.shape)
# print(b6_2.shape)
c6_5 = a6_2 * b6_2
# print(c6_5)

# 6-2. 행렬곱셈 (np.dot)

a6_3 = np.array([[1,2,3],
                [2,3,4]])
b6_3 = np.array([[1,2],
                [3,4],
                [5,6]])

c6_6 = np.dot(a6_3, b6_3)
# print(c6_6)
c6_7 = a6_3.dot(b6_3)
# print(c6_7)













