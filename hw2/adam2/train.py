import numpy as np
import math
import csv
from decimal import Decimal
def preprocess(A):
    augmented_weight = 6
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


def gradient_decent(x, y):
    global lr
    global batch
    global w
    global b
    global w_var
    global b_var
    global w_gradient
    global b_gradient
    sss = 0.00000001
    
    n = x.shape[0]
    loss = 0
    w_gradient = np.zeros((x.shape[1]))
    b_gradient = np.zeros((1))
    p = np.random.permutation(n)
    x = x[p]
    y = y[p]
    w_m_t = np.zeros((x.shape[1]))
    w_v_t = np.zeros((x.shape[1]))
    w_b1 = 0.9
    w_b2 = 0.9

    b_m_t = np.zeros((1))
    b_v_t = np.zeros((1))
    b_b1 = 0.9
    b_b2 = 0.9

    for ba in range(n // batch):
        z = (w * x[ba*batch:ba*batch+batch]).sum(axis = 1) + b
        result = 1.0 / (1.0 + np.exp(-z))
        loss -= ( y[ba*batch:ba*batch+batch] * np.log(result+sss) + (1-y[ba*batch:ba*batch+batch]) * np.log(1-result+sss)).sum()
        
        w_gradient = (np.expand_dims(y[ba*batch:ba*batch+batch]-result, axis = 1)*x[ba*batch:ba*batch+batch]).sum(axis=0)
        b_gradient = (y[ba*batch:ba*batch+batch]-result).sum()
        
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

    return loss/n
def validation(x, y):
    sss = 0.00000001
    z = (w * x).sum(axis = 1) + b
    result = 1.0 / (1.0 + np.exp(-z))
    loss = -1*(y * np.log(result + sss) + (1 - y) * np.log(1 - result + sss)).sum()
    return loss/x.shape[0]

if __name__ == "__main__":
    lr = 0.00005
    epoch = 100000
    batch = 30000
    num_val = 2561

    x_train = np.array(np.genfromtxt('X_train', delimiter=',')[1:])
    y_train = np.array(np.genfromtxt('Y_train', delimiter=',')[1:])
    
    x_train = preprocess(x_train)

    p = np.random.permutation(x_train.shape[0])
    x_train = x_train[p]
    y_train = y_train[p]


    w = np.zeros((x_train.shape[1]))
    b = np.zeros(1)
    # w = np.load('w.npy')
    # b = np.load('b.npy')
    w_var = np.zeros(x_train.shape[1])
    b_var = 0
    # w_var = np.load('w_v.npy')
    # b_var = np.load('b_v.npy')
    loss = 0
    
    try:
        for epo in range(epoch):
            loss = gradient_decent(x_train[num_val:], y_train[num_val:])
            val_loss = validation(x_train[0:num_val], y_train[0:num_val])
            print("[epoch] {0:6d}    [loss] {1:10.20f}   [val_loss] {2:10.20f}".format(epo+1,loss,val_loss))
        
    except (KeyboardInterrupt):
        np.save('result/w_'+str(lr)+'_'+str(epoch)+'_'+str(batch)+'_l='+str(loss), w)
        np.save('result/b_'+str(lr)+'_'+str(epoch)+'_'+str(batch)+'_l='+str(loss), b)
        np.save('w',w)
        np.save('b',b)
        np.save('w_v',w_var)
        np.save('b_v',b_var)

    np.save('w_'+str(lr)+'_'+str(epoch)+'_'+str(batch)+'_l='+str(loss), w)
    np.save('b_'+str(lr)+'_'+str(epoch)+'_'+str(batch)+'_l='+str(loss), b)
    # np.save('w',w)
    # np.save('b',b)
