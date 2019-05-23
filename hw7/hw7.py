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
import sys
from sklearn.cluster import KMeans
import pandas as pd
import gc



if __name__ == '__main__':
    x_tsne = np.load('models/my_x_tsne.npy')
    kmeans = KMeans(n_clusters=2, random_state=0).fit(x_tsne)
    test_case = pd.read_csv(sys.argv[2]).values[:, 1:3]
    print(x_tsne.shape)
    print(test_case.shape)

    gc.collect()
    result_csv = []
    result_csv.append(['id', 'label'])
    test_num = test_case.shape[0]

    x_tsne_label = np.zeros(x_tsne.shape[0], dtype =  'int32')

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
    np.savetxt(sys.argv[3], result_csv, delimiter=",", fmt="%s")