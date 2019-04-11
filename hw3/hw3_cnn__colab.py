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
import io

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

def readfile_from_np(train_file_path, label_file_path, val_num = 0.2,  augmentation = 1):
    print("Reading File...")
    
    train_data = np.load(train_file_path)
    label = np.load(label_file_path)
    
    train_data = train_data.reshape(-1, 1, 48, 48)
    
    n = len(label)
    p = np.random.permutation(n)
    train_data = train_data[p]
    label = label[p]

    n_val = round(n * (1 - val_num))
    
    x_train = train_data[:n_val]
    x_label = label[:n_val]
    val_data = train_data[n_val:]
    val_label = label[n_val:]

    x_train = np.array(x_train, dtype=float) / 255.0
    val_data = np.array(val_data, dtype=float) / 255.0
    x_label = np.array(x_label, dtype=int)
    val_label = np.array(val_label, dtype=int)
    x_train = torch.FloatTensor(x_train)
    val_data = torch.FloatTensor(val_data)
    x_label = torch.LongTensor(x_label)
    val_label = torch.LongTensor(val_label)
    
    return x_train, x_label, val_data, val_label

def readfile_from_csv(train_file_path, val_num = 0.2):
    print("Reading csv File...")
   
    data = io.StringIO(open(train_file_path).read().replace(',',' '))
    train = np.genfromtxt(data, delimiter=' ',  skip_header=1)
    
    print(train.shape)
    
    train_data = train[:,1:]
    label = train[:, 0]
    
    print(train_data.shape)
    print(label.shape)

    train_data = train_data.reshape(-1, 1, 48, 48)
    
    n = len(label)
    p = np.random.permutation(n)
    train_data = train_data[p]
    label = label[p]

    n_val = round(n * (1 - val_num))
    
    x_train = train_data[:n_val]
    x_label = label[:n_val]
    val_data = train_data[n_val:]
    val_label = label[n_val:]

    x_train = np.array(x_train, dtype=float) / 255.0
    val_data = np.array(val_data, dtype=float) / 255.0
    x_label = np.array(x_label, dtype=int)
    val_label = np.array(val_label, dtype=int)
    x_train = torch.FloatTensor(x_train)
    val_data = torch.FloatTensor(val_data)
    x_label = torch.LongTensor(x_label)
    val_label = torch.LongTensor(val_label)
    
    return x_train, x_label, val_data, val_label

def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)

class MyDataset(Dataset):
    def __init__(self, t_data, l_data, augmentation = 1):
        # self.train_data = np.genfromtxt(file_path, dtype=bytes, delimiter=' ')
        # self.label = np.genfromtxt(file_path, dtype=bytes, delimiter=' ')
        self.train_data = t_data
        self.label = l_data
        self.aug_size = augmentation
  
        # self.label = pd.read_csv(label_file_path, delimiter=' ', header = -1)
        # print(self.train_data.shape)
        # print(self.label.shape)
    def __len__(self):
        return len(self.label) * self.aug_size
    
    def __getitem__(self, idx):
       	
        # imread: a function that reads an image from path
        
        i  = int(idx/self.aug_size)
        if (idx % self.aug_size != 0):
            # _img = Image.fromarray(self.train_data[i])
            img = data_transformations(self.train_data[i])
            # img = np.array(new_img)
            
            l = self.label[i]

        else:
            img = self.train_data[i]
            l = self.label[i]

        
        # some operations/transformations
        return img, l
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
class DNN_Classifier(nn.Module):
    def __init__(self):
        super(DNN_Classifier, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(48*48, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 7)
        )

        self.fc.apply(gaussian_weights_init)

    def forward(self, x):
        return self.fc(x)



def main():
    # x_train, x_label, val_data, val_label = readfile(sys.argv[1])
    
    # x_train, x_label, val_data, val_label = readfile("train_data.npy", "label.npy", val_num = 0.2)
    # x_train, x_label, val_data, val_label = readfile_from_np("/content/drive/My Drive/ML2019/hw3_torch/train_data.npy", "/content/drive/My Drive/ML2019/hw3_torch/label.npy", val_num = 0.2, augmentation = 1)
    x_train, x_label, val_data, val_label = readfile_from_csv("/content/drive/My Drive/ML2019/hw3_torch/train.csv", val_num = 0.2)
    # x_train, x_label, val_data, val_label = readfile_from_np("train_data.npy", "label.npy", val_num = 0.2, augmentation = 5)
    
    train_set = MyDataset(x_train, x_label, augmentation = 1)     
    val_set = TensorDataset(val_data, val_label)
    
    num_epoch = 50
    batch_size = 256

    # if (len(sys.argv) >= 3):
    #     num_epoch = int(sys.argv[1])
    #     batch_size = int(sys.argv[2])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)

#########GPU#########
    model = Classifier().cuda()
    # model = Classifier()
    train_loss_log = []
    train_acc_log = []
    val_loss_log = []
    val_acc_log = []
    # print(model)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_acc = 0.0

    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        print('Epoch [%03d/%03d]' % (epoch+1, num_epoch))
        model.train()
        batch_num = len(train_loader)
        # msg = '...........'
        print('...........')
        for i, data in enumerate(train_loader):
            # print (msg,end = '', flush=True)
            optimizer.zero_grad()
#########GPU#########
            train_pred = model(data[0].cuda())
            batch_loss = loss(train_pred, data[1].cuda())
            # train_pred = model(data[0])
            # batch_loss = loss(train_pred, data[1])
            
            batch_loss.backward()
            optimizer.step()

            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            train_loss += batch_loss.item()

            # progress = (u"\u2588" * (int(float(i)/batch_num*40))).ljust(40,'.')
            # # progress = ('#' * int(float(i)/len(train_loader)*40)).ljust(40)
            # back = '\b'*len(msg)            
            # msg = '[%0 3d/%03d] %2.2f sec(s) |%s|' % (i+1, batch_num, \
            #         (time.time() - epoch_start_time), progress)
            # print(back,end = '', flush=True)
        
        model.eval()
        val_batch_num = len(val_loader)
        for i, data in enumerate(val_loader):
#########GPU#########
            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].cuda())
            # val_pred = model(data[0])
            # batch_loss = loss(val_pred, data[1])

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()

            # progress = ('=' * (int(float(i)/len(val_loader)*40)-1)+'>').ljust(40)
            # print ('[%03d/%03d] %2.2f sec(s) | %s |' % (i+1, num_epoch, \
                    # (time.time() - epoch_start_time), progress), end='\r', flush=True)

        val_acc = val_acc / val_set.__len__()
        train_acc = train_acc/train_set.__len__()
        print('[%0 3d/%03d] %2.2f sec(s) train_acc: %3.6f Loss: %3.6f | val_acc: %3.6f val_loss: %3.6f' % \
                (batch_num, batch_num, time.time()-epoch_start_time, \
                train_acc, train_loss/batch_num, val_acc, val_loss/val_batch_num))
        print('')

        #log 
        train_loss_log.append(train_loss/batch_num)
        train_acc_log.append(train_acc)
        val_loss_log.append(val_loss/val_batch_num)
        val_acc_log.append(val_acc)

        if (val_acc > best_acc):
            with open('/content/drive/My Drive/ML2019/hw3_torch/save/acc.txt','w') as f:
                f.write('-BEST MODEL -\nepoch: ' + str(epoch)+'/'+str(num_epoch)+'\t'+'val_acc: '+str(val_acc)+'\n')
            torch.save(model.state_dict(), '/content/drive/My Drive/ML2019/hw3_torch/save/L_best_model.pth')
            best_acc = val_acc
            print ('** Best Model Updated! ***\n')
        # if (val_acc > 0.67):
        #     path = '/content/drive/My Drive/ML2019/hw3_torch/save/models/L_%1.4f_model.pth'%(val_acc)
        #     torch.save(model.state_dict(), path)
        #     print ('**  Model Saved! ***\n')
    
    path = '/content/drive/My Drive/ML2019/hw3_torch/save/'
    np.save(path+'train_loss_log', train_loss_log)
    np.save(path+'train_acc_log', train_acc_log)
    np.save(path+'val_loss_log', val_loss_log)
    np.save(path+'val_acc_log', val_acc_log)


    # dataset = MyDataset("train_data.npy", "test.npy")
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    

if __name__ == '__main__':
    main()
