import numpy as np
import os
# import numpy as np

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


# l = np.load('train_data_raw_int32.npy')
# print(l[0])

seq1 = ['hello','good','boy','doiido']
print (os.path.join('hello','123'))
