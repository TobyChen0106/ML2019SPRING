import numpy as np
import math
import csv
from decimal import Decimal
def preprocess(A):
    augmented_weight = 3
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

    w = np.load('w.npy') 
    b = np.load('b.npy') 
    z = (w * x_test).sum(axis = 1) + b
    # z = z*0.00001
    result = 1.0 / (1.0 + np.exp(-z))


    output_file = "result.csv"
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'label'])
        for i in range(result.shape[0]):
            if ( result[i] > 0.5 ): 
                writer.writerow([i+1, 1])
            else:
                writer.writerow([i+1, 0])
