import numpy as np
import math
import csv
# import matplotlib.pyplot as plt


def preprocess(A):
    month = A
    for kind in range(month.shape[0]):
        for n in range(9):
            if math.isnan(month[kind][n]):
                month[kind][n] = 0
            if month[kind][n] < 0:
                left_ref = n
                right_ref = n
                while (left_ref >= 0 and month[kind][left_ref] < 0):
                    left_ref -= 1
                while (right_ref <= 8 and month[kind][right_ref] < 0):
                    right_ref += 1

                if left_ref == -1:
                    month[kind][n] = month[kind][right_ref]
                elif right_ref == 9:
                    month[kind][n] = month[kind][left_ref]
                else:
                    month[kind][n] = month[kind][left_ref] + (
                        month[kind][right_ref]-month[kind][left_ref])*(n-left_ref)/(right_ref-left_ref)
    # x = np.array(x)
    # print(month)
    x = []
    for id in range(month.shape[0] // 18):
        for n in range(month.shape[1]-5):
            x.append(month[id*18:id*18+18, n:n+6])
    x = np.array(x)
    print(x.shape)
    return x

def gradient_decent(x):
    global lr
    global epoch
    global batch
    global w
    global b
    
    # w = np.random.rand(18,5)
    # b = np.random.rand(1)

    
    n = x.shape[0]
    w_var = np.zeros((18,5))
    b_var = 0
    for epo in range(epoch):
        #np.random.shuffle(x)
        y = np.expand_dims(np.array(x[:,9,5]), axis=1)
        # print('y_shape', y.shape)

        loss = 0
        # w_gradient = np.zeros((18, 5))
        # b_gradient = np.zeros((1))
        
        batch_s = 0
        if batch_s + batch < n:
            batch_e = batch_s + batch-1
        else:
            batch_e = n - 1
        while (batch_s < n):
            
            delta = y[batch_s:batch_e+1] - (w * x[batch_s:batch_e+1,:,:5]).sum(axis=1).sum(axis=1, keepdims = True) -b
            loss += (delta ** 2).sum()
            
            delta = np.expand_dims(delta, axis=1)
            w_g = (delta * x[batch_s:batch_e + 1,:,:5])
            w_gradient = -1*w_g.sum(axis=0)
            b_gradient = -1*delta.sum()

            # w_gradient -= w_g.sum(axis=0)
            # b_gradient -= delta.sum()
            w_var += (w_gradient ** 2)
            b_var += (b_gradient ** 2)

            w -= lr * w_gradient / (w_var ** 0.5)
            b -= lr * b_gradient / (b_var ** 0.5)
            
            batch_s = batch_e + 1
            if batch_e + batch < n:
                batch_e = batch_e + batch
            else:
                batch_e = n - 1
            
        print("[epoch] ", epo+1, "       [loss] ", loss/n)
    return w, b, loss/n


if __name__ == "__main__":
    lr = 0.00001
    epoch = 100000
    batch = 960
    
    w = np.load('w.npy')
    b = np.load('b.npy')
    all_data = np.genfromtxt('test.csv', delimiter=',', encoding="big5")
    all_data = all_data[:, 2:]
    train_data = preprocess(all_data)
    w, b, lo = gradient_decent(train_data)

    np.save('w_'+str(lr)+'_'+str(epoch)+'_'+str(batch), w)
    np.save('b_'+str(lr)+'_'+str(epoch)+'_'+str(batch), b)
