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


def readfile_from_np(train_file_path, label_file_path, val_num=0.2,  augmentation=1):
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


def readfile_from_csv(train_file_path, val_num=0.2):
    print("Reading csv File...")

    data = io.StringIO(open(train_file_path).read().replace(',', ' '))
    train = np.genfromtxt(data, delimiter=' ',  skip_header=1)
    # np.save('data/train',train)
    delete_list = [59, 223, 418, 2059, 2171, 2809, 2892, 3262, 3931, 4275, \
               5274, 5439, 5509, 5722, 5881, 6102, 6458, 6893, 7172, 7496, \
               7527, 7629, 8030, 8737, 8856, 9026, 9500, 9679, 10585, 11244, \
               11286, 11295, 11846, 12289, 12352, 13148, 13988, 14279, 15144, 15838, \
               15894, 16540, 17081, 18012, 19238, 19632, 20222, 20712, 20817, 21817, \
               22198, 22927, 23596, 23894, 24053, 24408, 24891, 25219, 25603, 25909, \
               26383, 26561, 26860, 26897, 27292]

    # train = np.load('data/train.npy')

    train = np.delete(train, delete_list, axis = 0)

    print(train.shape)

    train_data = train[:, 1:]
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
    def __init__(self, t_data, l_data, augmentation=1):
        self.train_data = t_data
        self.label = l_data
        self.aug_size = augmentation

    def __len__(self):
        return len(self.label) * self.aug_size

    def __getitem__(self, idx):

        i = int(idx/self.aug_size)
        if (idx % self.aug_size != 0):
            img = data_transformations(self.train_data[i])
            # print(img)
            # new_img = np.array(img.permute(1,2,0)*255).squeeze(2).astype('uint8')
            # new_img = Image.fromarray(new_img)
            # new_img.save('output/%0.6d.png'%idx)
            l = self.label[i]

        else:
            img = self.train_data[i]
            l = self.label[i]

        return img, l


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
            nn.Dropout(0.2)        )
        self.model.apply(gaussian_weights_init)
        self.fc.apply(gaussian_weights_init)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x


def main():

    x_train, x_label, val_data, val_label = readfile_from_csv(sys.argv[1], val_num=0.2)

    train_set = MyDataset(x_train, x_label, augmentation=16)
    val_set = TensorDataset(val_data, val_label)

    num_epoch = 1000
    batch_size = 512

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=8)

    model = Net().cuda()

    loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_acc = 0.0
    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        print('Epoch [%03d/%03d] Training...' % (epoch+1, num_epoch))
        model.train()
        batch_num = len(train_loader)

        for i, data in enumerate(train_loader):

            optimizer.zero_grad()

            train_pred = model(data[0].cuda())
            batch_loss = loss(train_pred, data[1].cuda())

            batch_loss.backward()
            optimizer.step()

            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(),
                                          axis=1) == data[1].numpy())
            train_loss += batch_loss.item()


            progress = (u"\u2588" * (int(float(i)/batch_num*40))
                        ).ljust(40, '.')
            msg = '[%03d/%03d] %2.2f sec(s) |%s|' % (i+1, batch_num,
                                                     (time.time() - epoch_start_time), progress)
            print(msg, end='', flush=True)
            back = '\b'*len(msg)
            print(back, end='', flush=True)

        model.eval()
        val_batch_num = len(val_loader)
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].cuda())

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(),
                                        axis=1) == data[1].numpy())
            val_loss += batch_loss.item()

        val_acc = val_acc/val_set.__len__()
        train_acc = train_acc/train_set.__len__()
        print('[%0 3d/%03d] %2.2f sec(s) train_acc: %3.6f Loss: %3.6f | val_acc: %3.6f val_loss: %3.6f' %
              (batch_num, batch_num, time.time()-epoch_start_time,
               train_acc, train_loss/batch_num, val_acc, val_loss/val_batch_num))
        print('')

        if (val_acc > best_acc):
            with open('models/acc.txt', 'w') as f:
                f.write('-BEST MODEL -\nepoch: ' + str(epoch)+'/' +
                        str(num_epoch)+'\t'+'val_acc: '+str(val_acc)+'\n')

            torch.save(model.state_dict(), 'models/best_model.pth')
            best_acc = val_acc
            print('** Best Model Updated! ***\n')

    torch.save(model.state_dict(), 'models/final_model.pth')

if __name__ == '__main__':
    main()
