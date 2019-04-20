import numpy as np


a = np.array([
    [[[0,0,0],
    [0, 0, 0]]],
    [[[1,1,1],
    [1, 1, 1]]],
    [[[2,2,2],
    [2, 2, 2]]],
    [[[3,3,3],
    [3, 3, 3]]]
])

# print(a.shape)
# print(a)

# b = np.transpose(a, (0,3,1,2))
# print(b.shape)
# print(b)

# c = np.transpose(a, (0,3,1,2))
# print(b.shape)
# print(b)


A = np.ones((1, 2, 3), dtype = 'int32')
B = np.ones((1, 2, 3), dtype = 'float')

C = A+B
print(C)
print(C.dtype)
