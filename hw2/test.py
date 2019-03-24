import numpy as np
import math
import csv
import sys

def preprocess(A):
    augmented_weight = 1

    A[:, [2, 5]] = A[:, [5, 2]]
    
    new_x = np.empty((A.shape[0], A.shape[1] + (augmented_weight - 1) * 5))
    
    new_sum = A[:,:5].sum(axis=0, keepdims=True)
    new_avr = new_sum/A.shape[0]
    A[:,:5] = (A[:,:5]-new_avr)/np.var(A[:,:5], axis=0)

    # print(A[:, 2])
    
    for i in range(5):
        for j in range(augmented_weight):
            new_x[:, i*augmented_weight + j] = np.power(A[:, i], j + 1)
    new_x[:, 5 * augmented_weight:] = A[:, 5:]
    # new_x[:, augmented_weight*5] = A[:,2]
    # print (new_x[0])            

    return new_x

if __name__ == "__main__":
    
    x_test = np.genfromtxt(sys.argv[5], delimiter=',')[1:]
    # x_test = preprocess(x_test)

    w = np.load('terminal/w_0.001_100000_100_l=0.319262872112.npy') 
    b = np.load('terminal/b_0.001_100000_100_l=0.319262872112.npy') 
    z = (w * x_test).sum(axis = 1) + b
    # for i in range(len(w)):
    #     print(w[i])
    result_raw = 1.0 / (1.0 + np.exp(-z))
    n = x_test.shape[0]

    # result_idx = np.argsort(result_raw)
    
    # result = np.ones(x_test.shape[0])
    
    # for i in range(n):
    #     print(result_raw[result_idx[i]])
    # for i in range(n * 3 // 4):
    #     result[result_idx[i]] = 0

    output_file = sys.argv[6]
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'label'])
        for i in range(n):
            if (result_raw[i] > 0.5):
                writer.writerow([i+1, 1])
            else:
                writer.writerow([i+1, 0])

            # writer.writerow([i+1, result[i]])
