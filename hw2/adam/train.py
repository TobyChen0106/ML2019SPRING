import numpy as np
import math
import csv
from decimal import Decimal
def preprocess(A):
    augmented_weight = 2
    new_x = np.empty((A.shape[0], A.shape[1] + (augmented_weight - 1) * 5))
    
    A[:,[2,5]] = A[:, [5,2]]
    
    for i in range(5):
        for j in range(augmented_weight):
            new_x[:, i*augmented_weight + j] = np.power(A[:, i], j + 1)
    new_x[:, 5*augmented_weight:] = A[:, 5:]
    # new_x[:, augmented_weight*5] = A[:,2]
    # print (new_x[0])            

    return new_x

def gradient_decent(x, y):
    global lr
    global epoch
    global batch
    global w
    global b
    global loss
    global w_var
    global b_var
    sss = 1e-8

    n = x.shape[0]
    
    for epo in range(epoch):
        p = np.random.permutation(n)
        x = x[p]
        y = y[p]
        loss = 0
        w_gradient = np.zeros((x.shape[1]))
        b_gradient = np.zeros((1))
        
        w_m_t = np.zeros((x.shape[1]))
        w_v_t = np.zeros((x.shape[1]))
        w_b1 = 0.9
        w_b2 = 0.9

        b_m_t = np.zeros((1))
        b_v_t = np.zeros((1))
        b_b1 = 0.9
        b_b2 = 0.9

        batch_s = 0
        if batch_s + batch < n:
            batch_e = batch_s + batch-1
        else:
            batch_e = n - 1
        while (batch_s < n):
            z = (w * x[batch_s:batch_e + 1]).sum(axis = 1) + b
            # z = z*0.00001
            result = 1.0 / (1.0 + np.exp(-z))

            loss -= ( y[batch_s:batch_e + 1] * np.log(result+sss) + (1-y[batch_s:batch_e + 1]) * np.log(1-result+sss)).sum()
            
            w_gradient = (np.expand_dims(y[batch_s:batch_e + 1]-result, axis = 1)*x[batch_s:batch_e + 1]).sum(axis=0)
            b_gradient = (y[batch_s:batch_e + 1]-result).sum()
            
            w_var += (w_gradient ** 2)
            b_var += (b_gradient ** 2)

            # adam
            w_m_t = w_b1*w_m_t -(1-w_b1)*w_gradient
            w_v_t = w_b2*w_v_t +(1-w_b1)*(w_gradient**2)
            w_m_t_h = w_m_t/(1-w_b1**(epo+1))
            w_v_t_h = w_v_t/(1-w_b2**(epo+1))

            b_m_t = b_b1*b_m_t -(1-b_b1)*b_gradient
            b_v_t = b_b2*b_v_t +(1-b_b1)*(b_gradient**2)
            b_m_t_h = b_m_t/(1-b_b1**(epo+1))
            b_v_t_h = b_v_t/(1-b_b2**(epo+1))

            w -= lr * w_m_t_h / (w_v_t_h ** 0.5+sss)
            b -= lr * b_m_t_h / (b_v_t_h ** 0.5+sss)

            # w += lr * w_gradient / (w_var ** 0.5+sss)
            # b += lr * b_gradient / (b_var ** 0.5+sss)
            
            batch_s = batch_e + 1
            if batch_e + batch < n:
                batch_e = batch_e + batch
            else:
                batch_e = n - 1
        loss = loss/n
        print("[epoch] ", epo+1, "       [loss] ", loss)
    # return w, b, loss/n

if __name__ == "__main__":
    lr = 100
    epoch = 100000
    batch = 50

    x_train = np.array(np.genfromtxt('X_train', delimiter=',')[1:])
    y_train = np.array(np.genfromtxt('Y_train', delimiter=',')[1:])

    x_train = preprocess(x_train)
    print(x_train.shape[1])
    w = np.random.rand(x_train.shape[1])
    b = np.random.rand(1)
    # w = np.load('w.npy')
    # b = np.load('b.npy')
    w_var = np.zeros(x_train.shape[1])
    b_var = 0
    # w_var = np.load('w_v.npy')
    # b_var = np.load('b_v.npy')
    loss = 0

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
