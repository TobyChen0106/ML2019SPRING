import numpy as np
import math
import csv
import matplotlib.pyplot as plt


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
    print(month)
    return month

if __name__ == "__main__":
    all_data = np.genfromtxt('test.csv', delimiter=',', encoding="big5")
    all_data = all_data[:, 2:]
    x = preprocess(all_data)
    # print(x.shape)
    w = np.load('w_0.1_100000_471_l=35.30375642737103.npy')
    b = np.load('b_0.1_100000_471_l=35.30375642737103.npy')
    with open('result.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'value'])
        for id in range(x.shape[0]//18):     
            y = (w * x[id*18+9, 4:9]).sum() + b
            # (w * np.expand_dims(x[batch_s:batch_e+1,9,:5], axis = 1)).sum(axis=1).sum(axis=1, keepdims = True) -b
            writer.writerow(['id_' + str(id), y[0]])
            # writer.writerow(x[id])
            # print(y)
    # print(all_data[10][0])
    # if math.isnan(all_data[10][0]):
    #     print("haha")
