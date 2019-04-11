from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader
import sys
import csv
import time
import numpy as np
import pandas as pd
from PIL import Image


data_transformations = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomResizedCrop(size=48,
                                    scale=(0.7, 1.0),
                                    ratio=(0.75, 1.3333333333333333),
                                    interpolation=2),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=30,
                                    resample=Image.BILINEAR,
                                    expand=False,
                                    center=None),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # transforms.nparray()
])

train_data = np.load('train_data.npy')
label = np.load('label.npy')


augmented_data = [] 
augmented_label = []

print('Processing Data Augmentation...')
train_data = train_data.reshape((-1,48, 48))
img_num = len(label)
epoch_start_time = time.time()
for i in range(len(label)):
    
    augmented_data.append(train_data[i])
    augmented_label.append(label[i])
    _img = Image.fromarray(np.uint8(train_data[i]))
    if (i == 0):
        # _img = Image.fromarray(np.uint8(train_data[i]))
        _img.save('data/0_0.png')
    for j in range(9):
        new_img = data_transformations(_img)
        new_img = np.array(new_img)
        augmented_data.append(new_img)
        augmented_label.append(label[i])
        if (i == 0):
            img = Image.fromarray(np.uint8(new_img))
            img.save('data/0_'+str(j+1)+'.png')

    progress = (u"\u2588" * (round(int(float(i)/img_num*40)))).ljust(40,'.')
    msg = 'solving : [%03d/%03d] %2.2f sec(s) |%s|' % (i+1, img_num, \
            (time.time() - epoch_start_time), progress)
    
    print(msg,end = '')
    back = '\b'*len(msg)
    print(back,end = '')
print('')
save_data = np.array(augmented_data)
save_label = np.array(augmented_label)
print('save_data_shape', save_data.shape)
print('save_label_shape', save_label.shape)

# np.save('data/aug_train_data', save_data )
# np.save( 'data/aug_label', save_label)




