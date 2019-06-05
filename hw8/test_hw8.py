from __future__ import print_function
# import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader
import sys
# import csv
# import time
import numpy as np
import pandas as pd
from PIL import Image
import io
from extract import extract
import gc 

data_transformations = transforms.Compose([
    transforms.ToPILImage(),
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
    transforms.ToTensor()
])

def readfile_from_np(test_file_path):
    # print("Reading CSV File...")
    
    test_data = np.load(test_file_path)
    test_data = test_data.reshape(-1,1,48,48)

    test_data = np.array(test_data, dtype=float) / 255.0
    test_data = torch.FloatTensor(test_data)
    
    # for i in range(len(test_data)):
    #     test_data[i] = data_transformations(test_data[i])
    return test_data

def readfile_from_csv(test_file_path):
    # print("Reading csv File...")
    
    data = io.StringIO(open(test_file_path).read().replace(',',' '))
    test_data = np.genfromtxt(data, delimiter=' ',  skip_header=1)[:, 1:]

    test_data = test_data.reshape(-1,1,48,48)

    test_data = np.array(test_data, dtype=float) / 255.0
    test_data = torch.FloatTensor(test_data)

    # np.save('data/test_data',test_data)

    for i in range(len(test_data)):
        test_data[i] = data_transformations(test_data[i])

    return test_data

def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(oup),
                nn.Dropout(0.1)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(inp),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(oup),
                nn.Dropout(0.1)
            )

        self.model = nn.Sequential(
            conv_bn(1,  16, 2),     # 16, 24, 24
            conv_dw(16, 32, 1),    # 32, 24, 24
            conv_dw(32, 32, 1),    # 32, 24, 24
            conv_dw(32, 64, 2),   # 64, 12, 12
            conv_dw(64, 64, 1),    # 64, 12, 12
            conv_dw(64, 128, 1),    # 128, 12, 12
            conv_dw(128, 128, 1),   # 128, 12, 12
            conv_dw(128, 128, 2),   # 128, 6, 6
            conv_dw(128, 128, 1),   # 128, 6, 6
            conv_dw(128, 256, 1),   # 256, 6, 6


            nn.AvgPool2d(6),
        )
        self.fc = nn.Sequential(
            # nn.Linear(512, 512),
            # nn.LeakyReLU(0.2),
            # nn.Dropout(p=0.5),
            nn.Linear(256, 7),
            nn.Dropout(0.1)
        )
        self.model.apply(gaussian_weights_init)
        self.fc.apply(gaussian_weights_init)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x


def main():

    batch_size = 256
    test_num = 20

    model = Net().cuda()
    _dict = torch.load('models/best_model.pth')
    model.load_state_dict(_dict)

    # model = extract('models/compressed_weight.npz')
    # test_data = readfile_from_csv('data/test.csv')
    gc.collect()
    for ni in range(test_num):
        msg = 'Solving input augmentation [%03d/%03d]'%( ni+1, test_num )
        print(msg, flush = True)
        back = '\b'*len(msg)
        # print(back, flush = True)
        
        test_data = readfile_from_csv(sys.argv[1])
        test_set = TensorDataset(test_data)     
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)
        
        
        model.eval()
        val_batch_num = len(test_loader)
        for i, data in enumerate(test_loader):
            val_pred =  model(data[0].cuda())
            result_raw = val_pred.cpu().data.numpy()
            
            if (i == 0):
                result_con = result_raw
            else:
                result_con = np.append(result_con, result_raw, axis = 0)
        
        if (ni == 0):
            result_all = result_con
        else:
            result_all = result_all + result_con
        gc.collect()
        
    result = np.argmax(result_all, axis=1)
    print(result.shape)

    output_file = sys.argv[2]


    result_csv = []
    result_csv.append(['id','label'])
    for i in range(len(result)):
        result_csv.append([i,result[i]])
    result_csv = np.array(result_csv)
    print(result_csv.shape)
    np.savetxt(output_file,result_csv, delimiter=",", fmt="%s")
    
           

if __name__ == '__main__':
    main()