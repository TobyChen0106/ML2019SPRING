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
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(size=48,
                                    scale=(0.99, 1.0),
                                    ratio=(0.75, 1.3333333333333333),
                                    interpolation=2),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10,
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
    
    for i in range(len(test_data)):
        test_data[i] = data_transformations(test_data[i])
    return test_data

def readfile_from_csv(test_file_path):
    print("Reading csv File...")
   
    data = io.StringIO(open(test_file_path).read().replace(',',' '))
    test_data = np.genfromtxt(data, delimiter=' ',  skip_header=1)[:, 1:]
    
    for i in range(len(test_data)):
            test_data[i] = data_transformations(test_data[i])
    return test_data

def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)
class L_Classifier(nn.Module):
    def __init__(self):
        super(L_Classifier, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),  # [64, 48, 48]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0),      # [64, 24, 24]

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0),      # [128, 12, 12]

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0)       # [256, 6, 6]
        )

        self.fc = nn.Sequential(
            nn.Linear(256*6*6, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            nn.Linear(512, 7)
        )

        self.cnn.apply(gaussian_weights_init)
        self.fc.apply(gaussian_weights_init)

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),  # [64, 48, 48]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0),      # [64, 24, 24]

            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0),      # [128, 12, 12]

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0)       # [256, 6, 6]
        )

        self.fc = nn.Sequential(
            nn.Linear(128*6*6, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            nn.Linear(512, 7)
        )
        self.cnn.apply(gaussian_weights_init)
        self.fc.apply(gaussian_weights_init)

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

def main():

    batch_size = 256
    test_num = 10
    
#     models = ['000_1_best_model.pth','0.6705_model.pth','0.6771_model.pth', 
# '001_1_best_model.pth','0.6707_model.pth','0.6773_model.pth', 
# '002_1_best_model.pth','0.6714_model.pth','0.6776_model.pth', 
# '003_1_best_model.pth','0.6722_model.pth','0.6794_model.pth', 
# '004_1_best_model.pth','0.6736_model.pth','p0.68013_model.pth', 
# '005_1_best_model.pth','0.6764_model.pth','v0.68286_model.pth', 
# '0.6701_model.pth','0.6766_model.pth']
    models = ['L_001_1_best_model.pth', 'L_002_1_best_model.pth', 'L_003_1_best_model.pth', 'L_004_1_best_model.pth', 'L_005_1_best_model.pth',
    '000_1_best_model.pth', '001_1_best_model.pth', '002_1_best_model.pth',
    '003_1_best_model.pth', '004_1_best_model.pth','p0.68013_model.pth','005_1_best_model.pth','v0.68286_model.pth']

    for mi in range(len(models)):
        for ni in range(test_num):
            msg = 'Solving Model [%03d/%03d]: %s input augmentation [%03d/%03d]'%(mi+1, len(models), models[mi], ni+1, test_num )
            print(msg, flush = True)
            # back = '\b'*len(msg)
            
            # test_data = readfile_from_np("/content/drive/My Drive/ML2019/hw3_torch/test.npy")
            test_data = readfile_from_csv("/content/drive/My Drive/ML2019/hw3_torch/test.npy")
            # test_data = readfile_from_np("test.npy")

            test_set = TensorDataset(test_data)     
            test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)

        #########GPU#########
            if (models[mi][0] == 'L'):
                # print('L_model')
                model = L_Classifier().cuda()
            else:
                model = Classifier().cuda()
            # model = Classifier()
            _dict = torch.load('/content/drive/My Drive/ML2019/hw3_torch/save/models/'+models[mi])
            # _dict = torch.load('save/model.pth', map_location='cpu')
            model.load_state_dict(_dict)
            
            model.eval()
            val_batch_num = len(test_loader)
            for i, data in enumerate(test_loader):
        #########GPU#########        
                # val_pred =  F.softmax(  model(data[0].cuda()))
                val_pred =  model(data[0].cuda())
                # val_pred = model(data[0])
            
                result_raw = val_pred.cpu().data.numpy()
                # result = np.argmax(result_raw, axis=1)
                
                # print(result.shape)
                if (i == 0):
                    result_con = result_raw
                else:
                    result_con = np.append(result_con, result_raw, axis = 0)
            
            if (mi == 0 and ni == 0):
                result_all = result_con
            else:
                result_all = result_all + result_con
            # print(back, end = '', flush = True)
    result = np.argmax(result_all, axis=1)
    print(result.shape)

        # result_all.append(result_raw)
    # result_con = np.array(result_con)
    # print(result_con.shape)
    # result_con = result_con.reshape((result_con.shape[0]))
    # print(result_con.shape)

    output_file = '/content/drive/My Drive/ML2019/hw3_torch/save/L_result.csv'
    # output_file = '/content/drive/My Drive/ML2019/hw3_torch/save/result.csv'

    result_csv = []
    result_csv.append(['id','label'])
    for i in range(len(result)):
        result_csv.append([i,result[i]])
    result_csv = np.array(result_csv)
    print(result_csv.shape)
    np.savetxt(output_file,result_csv, delimiter=",", fmt="%s")
    
           

if __name__ == '__main__':
    main()