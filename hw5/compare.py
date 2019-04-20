from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader
import torchvision.models as models
from torch.autograd import Variable
import requests, io
import sys
import csv
import time
import numpy as np
import pandas as pd
from PIL import Image
import scipy.misc


def load_img(idx, dir):
    file_name = dir+'%03d'%(idx)+'.png'
    img = Image.open(file_name)
    # img = preprocess(img)
    return np.array(img, dtype = 'int32')

img_1_dir = 'hw5_data/images/'
img_2_dir = 'output (19)/'
sum = 0
for i in range(200):
    img_1 = load_img(i, img_1_dir)
    img_2 = load_img(i, img_2_dir)
    # print(img_1[0],'**********',img_2[0])
    dif = np.absolute(img_2-img_1).max()
    print('[%00d]: %d' % (i, int(dif)))
    sum+=dif
    # print( dif)
    
print('Total L-infinity: %d'%(sum))
print('Average L-infinity: %3.6f'%(sum/200))