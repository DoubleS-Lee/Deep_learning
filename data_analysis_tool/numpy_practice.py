import numpy as np

a=np.array([1,2,3,4])

a.shape

b=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])

b.shape

arr = np.array([1,2,3,4], dtype=int)

type(arr)

list1 = [1,2,3,4]
list1[-1]
list2 = ([1,2,3,4],[5,6,7,8])

arr1 = np.array(list1)
arr2 = np.array(list2)

arr1.shape
arr2.shape

arr = np.array([1,2,3,3.14])
arr
arr = np.array([1,2,3,3.14], dtype=int)

arr = np.array([1,2,3,'테디'])
arr[1] + arr[3]

arr1d = np.array([0,1,2,3,4,5,6,7,8,9])

arr1d[1]

arr2d = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])

arr2d[0,2]
arr2d[2,2]


arr1d[2:5]
arr1d[:]
arr1d[:-1]

arr2d[0,:]
arr2d[:,2]
arr2d[1,2]
arr2d[:-1,:]
arr2d[:2,2:]

arr = np.array([1,3,6453,3425,123,4532,6452])
a=arr[[1,3]]
a=arr[[4]]

arr = np.arange(1,11)

np.arange(1,10,2)

np.sort(arr)

np.sort(arr)[::-1]

#2차원 정렬
arr2d = np.array([[5,6,7,8],[4,3,2,1],[10,9,12,11]])
arr2d.shape

# 열 정렬
np.sort(arr2d, axis=1)

# 행 정렬
np.sort(arr2d, axis=0)


a = np.array([[1,2,3],[2,3,4]])

b = np.array([[3,4,5],[1,2,3]])

c = np.array([[1,2],[3,4],[5,6]])

np.sum(a, axis=1)
np.sum(a, axis=0)

np.sort(b, axis=0)
np.sort(b, axis=1)


a*b
np.dot(a,c)
np.dot(b,c)


