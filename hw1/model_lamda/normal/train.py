import numpy as np
import math
import csv
# import matplotlib.pyplot as plt


def preprocess(A):
    month_data = np.zeros( (12, 18, 480) )
    for month in range(12):
        for day in range(20):
            month_data[month, 0:18, 24*day:24*day+24] = A[18 *
                                                          (day+20*month):18*(day+20*month)+18, 0:24]
    # print(month_data)

    for month in month_data:
        for kind in range(month_data.shape[0]):
            for n in range(month_data.shape[2]):
                if math.isnan(month[kind][n]):
                    month[kind][n] = 0
                if month[kind][n] < 0:
                    left_ref = n
                    right_ref = n
                    while (left_ref >= 0 and month[kind][left_ref] < 0):
                        left_ref -= 1
                    while (right_ref <= 479 and month[kind][right_ref] < 0):
                        right_ref += 1

                    if left_ref == -1:
                        month[kind][n] = month[kind][right_ref]
                    elif right_ref == 480:
                        month[kind][n] = month[kind][left_ref]
                    else:
                        month[kind][n] = month[kind][left_ref] + (
                            month[kind][right_ref]-month[kind][left_ref])*(n-left_ref)/(right_ref-left_ref)
    with open('new_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for month in month_data:
            for kind in range(month.shape[0]):
                # plt.plot(month[kind])
                writer.writerow(month[kind])
                # print(month_data[0,10])
    # print(month_data.shape)
    x = []
    # y = []
    for month in month_data:
        for i in range(471):
            x.append(month[:, i:i+10])
            # y.append([month[9, i+9]])

    x = np.array(x)
    # y = np.array(y)
    # print(y.shape)
    # print(month_data.shape)
    return x


def gradient_decent(x):
    global lr
    global epoch
    global batch
    global lamda
    global result
    w = np.random.rand(18,9)
    b = np.random.rand(1)

    
    n = x.shape[0]
    w_var = np.zeros((18,9))
    b_var = 0
    for epo in range(epoch):
        np.random.shuffle(x)
        y = np.expand_dims(np.array(x[:,9,9]), axis=1)
        # print('y_shape', y.shape)

        loss = 0
        w_gradient = np.zeros((18, 9))
        b_gradient = np.zeros((1))
        
        batch_s = 0
        if batch_s + batch < n:
            batch_e = batch_s + batch-1
        else:
            batch_e = n - 1
        while (batch_s < n):
            
            delta = y[batch_s:batch_e+1] - (w * x[batch_s:batch_e+1,:,:9]).sum(axis=1).sum(axis=1, keepdims = True) -b
            loss += (delta ** 2).sum() + (lamda*w*w).sum()
            
            delta = np.expand_dims(delta, axis=1)
            w_g = (delta * x[batch_s:batch_e + 1,:,:9])-2*lamda*w
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
        if (epo % 100 == 0):
            result.append(loss/n)
        print("[epoch] ", epo+1, "       [loss] ", loss/n)
    return w, b, loss/n


if __name__ == "__main__":
    lr = 0.1
    epoch = 100000
    batch = 471
    lamda = 0.01
    result = []
    
    all_data = np.genfromtxt('train.csv', delimiter=',', encoding="latin1")
    all_data = all_data[1:, 3:]
    train_data= preprocess(all_data)
    w, b, lo = gradient_decent(train_data)

    np.save('w_'+str(lr)+'_'+str(epoch)+'_'+str(batch)+'_l='+str(lo)+'_lamda='+str(lamda), w)
    np.save('b_'+str(lr)+'_'+str(epoch)+'_'+str(batch)+'_l='+str(lo)+'_lamda='+str(lamda), b)
    np.save('loss2epo_lamda='+str(lamda), result)