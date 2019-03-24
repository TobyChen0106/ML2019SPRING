import numpy as np
import math
import csv
from decimal import Decimal

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
    
    x_test = np.genfromtxt('X_test', delimiter=',')[1:]
    x_test = preprocess(x_test)

    w = np.load('w_0.001_100000_1000_l=0.5602377864937284.npy') 
    b = np.load('b_0.001_100000_1000_l=0.5602377864937284.npy') 
    z = (w * x_test).sum(axis = 1) + b
    # z = z*0.00001
    result_raw = 1.0 / (1.0 + np.exp(-z))

    result_idx = np.argsort(result_raw)
    n = x_test.shape[0]
    result = np.ones(x_test.shape[0])
    for i in range(n*3//4):
        result[result_idx[i]] = 0

    output_file = "result.csv"
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'label'])
        for i in range(result.shape[0]):
            writer.writerow([i+1, result[i]])