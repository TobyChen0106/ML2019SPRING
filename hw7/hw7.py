import numpy as np
from sklearn.manifold import TSNE
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader
from skimage import io
import numpy as np
from PIL import Image
import time
from sklearn.cluster import KMeans
import pandas as pd
import gc

def read_image(input_dir):
    print('reading images...')
    # images = []
    # for i in range(40000):
    #     image = Image.open(input_dir+'%06d.jpg' % (i+1))
    #     image = np.array(image)
    #     image = (image - 127.5)/127.5   # (-1 ~ 1)
    #     images.append(image)
    # images = np.array(images)
    # np.save('images',images)
    images = np.load('data/images.npy')
    print('iamges shape: ', images.shape)
    print('iamges dtype: ', images.dtype)
    # print(images[0])

    return images


if __name__ == '__main__':
    # images = read_image('data/images/')
    x_tsne = np.load('data/x_tsne_4_m3.npy')
    kmeans = KMeans(n_clusters=2, random_state=0).fit(x_tsne)
    test_case = pd.read_csv('data/test_case.csv').values[:, 1:3]
    print(x_tsne.shape)
    print(test_case.shape)

    gc.collect()
    result_csv = []
    result_csv.append(['id', 'label'])
    test_num = test_case.shape[0]

    x_tsne_label = np.zeros(test_case.shape[0], dtype =  'int32')
    for i, feature in enumerate( x_tsne):
        result = kmeans.predict([feature])
        x_tsne_label[i] = int(result[0])
    print('x_tsne_label', x_tsne_label)
    for i, test in enumerate(test_case):
        # print(x_tsne[test[0]])
        result1 = x_tsne_label[test[0]-1]
        result2 = x_tsne_label[test[1]-1]
        # print(result1)
        # print(result2)
        if(result1 == result2):
            result = 1
        else:
            result = 0

        result_csv.append([i, int(result)])
        # gc.collect()
        msg = 'solving [%04d/%04d]'%(i+1,test_num)
        print(msg, end = '',flush = True)
        back = '\b'*len(msg)
        print(back, end = '',flush = True)

    result_csv = np.array(result_csv)
    print(result_csv.shape)
    np.savetxt('result.csv', result_csv, delimiter=",", fmt="%s")