import numpy as np
import math
import csv
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

    x_train = np.array(np.genfromtxt('X_train', delimiter=',')[1:])
    y_train = np.array(np.genfromtxt('Y_train', delimiter=',')[1:])
    
    x_train = preprocess(x_train)

    idx_0 = np.where(y_train == 0)
    idx_1 = np.where(y_train == 1)
    
    mu_0 = x_train[idx_0].mean(axis=0)
    mu_1 = x_train[idx_1].mean(axis=0)

    cov_0 = np.cov(x_train, rowvar = False)
    cov_1 = np.cov(x_train, rowvar = False)

    cov = (cov_0 * len(idx_0) + cov_1 * len(idx_1) ) / (len(idx_1)+ len(idx_0))

    # print(mu_0 , mu_1, cov)
    np.save('mu_0', mu_0)
    np.save('mu_1', mu_1)
    np.save('cov', cov)
