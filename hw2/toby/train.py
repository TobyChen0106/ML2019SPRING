import numpy as np
import math
import csv
from decimal import Decimal
def preprocess(A):
    return np.array(A)


def gradient_decent(x, y):
    global lr
    global epoch
    global batch
    global w
    global b
    global loss
    global w_var
    global b_var
    sss = 0.00000001
    
    n = x.shape[0]
    
    for epo in range(epoch):
        p = np.random.permutation(n)
        x = x[p]
        y = y[p]
        # np.random.shuffle(train_data)
        # x = train_data[:,:106]
        # y = train_data[:,106]
        loss = 0
        w_gradient = np.zeros((106))
        b_gradient = np.zeros((1))
        
        batch_s = 0
        if batch_s + batch < n:
            batch_e = batch_s + batch-1
        else:
            batch_e = n - 1
        while (batch_s < n):
            z = (w * x[batch_s:batch_e + 1]).sum(axis = 1) + b
            # z = z*0.00001
            result = 1.0 / (1.0 + np.exp(-z))
            # print("-z", np.exp(-9999))            
            # print("result", result)
            # print("y", y[batch_s:batch_e + 1])
            # print('log1',np.log(result+sss))
            # print('log2',np.log(1-result+sss))
            loss -= ( y[batch_s:batch_e + 1] * np.log(result+sss) + (1-y[batch_s:batch_e + 1]) * np.log(1-result+sss)).sum()
            # print(loss.shape)
            if batch_s == 0:
                w_gradient = (np.expand_dims(y[batch_s:batch_e + 1]-result, axis = 1)*x[batch_s:batch_e + 1]).sum(axis=0)
                b_gradient = (y[batch_s:batch_e + 1]-result).sum()
            else:
                w_gradient += (np.expand_dims(y[batch_s:batch_e + 1]-result, axis = 1)*x[batch_s:batch_e + 1]).sum(axis=0)
                b_gradient += (y[batch_s:batch_e + 1]-result).sum()
            w_var += (w_gradient ** 2)
            b_var += (b_gradient ** 2)

            w += lr * w_gradient / (w_var ** 0.5+sss)
            b += lr * b_gradient / (b_var ** 0.5+sss)
            
            batch_s = batch_e + 1
            if batch_e + batch < n:
                batch_e = batch_e + batch
            else:
                batch_e = n - 1
        loss = loss/n
        print("[epoch] ", epo+1, "       [loss] ", loss)
    # return w, b, loss/n

if __name__ == "__main__":
    lr = 0.01
    epoch = 100000
    batch = 100
    w = np.random.rand(106)
    b = np.random.rand(1)
    # w = np.load('w.npy')
    # b = np.load('b.npy')
    w_var = np.zeros(106)
    b_var = 0
    # w_var = np.load('w_v.npy')
    # b_var = np.load('b_v.npy')
    loss = 0

    x_train = np.array(np.genfromtxt('X_train', delimiter=',')[1:])
    y_train = np.array(np.genfromtxt('Y_train', delimiter=',')[1:])

    x_train = preprocess(x_train)

    # train_data = np.concatenate((x_train, np.expand_dims(y_train, axis = 1)), axis = 1)
    try:
        gradient_decent(x_train, y_train)
    except (KeyboardInterrupt):
        np.save('result/w_'+str(lr)+'_'+str(epoch)+'_'+str(batch)+'_l='+str(loss), w)
        np.save('result/b_'+str(lr)+'_'+str(epoch)+'_'+str(batch)+'_l='+str(loss), b)
        np.save('w',w)
        np.save('b',b)
        np.save('w_v',w_var)
        np.save('b_v',b_var)

    # np.save('w_'+str(lr)+'_'+str(epoch)+'_'+str(batch)+'_l='+str(loss), w)
    # np.save('b_'+str(lr)+'_'+str(epoch)+'_'+str(batch)+'_l='+str(loss), b)
    np.save('w',w)
    np.save('b',b)
