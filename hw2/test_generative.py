import numpy as np
import math
import csv
from scipy.stats import multivariate_normal
import sys

def preprocess(A):
    augmented_weight = 1
    A[:, [2, 5]] = A[:, [5, 2]]
    
    new_x = np.empty((A.shape[0], A.shape[1] + (augmented_weight - 1) * 5))
    
    new_sum = A[:,:5].sum(axis=0, keepdims=True)
    new_avr = new_sum/A.shape[0]
    A[:,:5] = (A[:,:5]-new_avr)/np.var(A[:,:5], axis=0)

    for i in range(5):
        for j in range(augmented_weight):
            new_x[:, i*augmented_weight + j] = np.power(A[:, i], j + 1)
    new_x[:, 5 * augmented_weight:] = A[:, 5:]
    # new_x[:, augmented_weight*5] = A[:,2]
    # print (new_x[0])            

    return new_x


if __name__ == "__main__":

    mu_0 = np.load('mu_0.npy')
    mu_1 = np.load('mu_1.npy')
    cov = np.load('cov.npy')
    x_test = np.genfromtxt(sys.argv[5], delimiter=',')[1:]
    x_test = preprocess(x_test)
    
    prob0 = multivariate_normal.pdf(x_test, mean = mu_0, cov = cov, allow_singular=True)
    prob1 = multivariate_normal.pdf(x_test, mean = mu_1, cov = cov, allow_singular=True)
    
    
    # print(np.where(prob0 == 0))
    p_1 = len(prob1) / ( len(prob0) + len(prob1))
    from_p1 = (prob1 * p_1) / (prob1 * p_1 + prob0 * (1 - p_1)+1e-8)
    
    # print(from_p1)

    output_file = sys.argv[6]
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'label'])
        for i in range(len(x_test)):
            if (from_p1[i] > 0.5):
                writer.writerow([i+1, 1])
            else:
                writer.writerow([i+1, 0])
