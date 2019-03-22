import numpy as np
import math
import csv
from decimal import Decimal
def preprocess(A):
    return np.array(A)

if __name__ == "__main__":
    
    x_test = np.genfromtxt('X_test', delimiter=',')[1:]
    x_test = preprocess(x_test)

    w = np.load('w_0.01_100000_100_l=9531.459645349094.npy') 
    b = np.load('b_0.01_100000_100_l=9531.459645349094.npy') 
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
