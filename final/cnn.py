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

num_epochs = 50
batch_size = 100
lr = 0.0005
val = 0.1

data_path = ''
label_path = ''
output_model_path = 'best_model.pth'


def load_data_from_npy(train_data_path, train_label_path, val=0.2):
    print('start loading data...')
    data = np.load(train_data_path)
    label = np.load(train_label_path)
    # label = np.squeeze(label, axis=1)
    print('data shape = ', data.shape)
    print('label shape = ', label.shape)

    n = data.shape[0]
    p = np.random.permutation(n)
    data = data[p]
    label = label[p]

    n_val = round(n * (1 - val))
    train_x = data[:n_val]
    val_x = data[n_val:]
    train_y = label[:n_val]
    val_y = label[n_val:]

    # train_x = torch.FloatTensor(train_x)
    # val_x = torch.FloatTensor(val_x)
    # train_y = torch.LongTensor(train_y)
    # val_y = torch.LongTensor(val_y)

    print('finishing loading data!\n')
    return train_x, val_x, train_y, val_y


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0),      # [128, 64, 64]

            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0),      # [256, 32, 32]

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0)       # [256, 16, 16]
        )

        self.fc = nn.Sequential(
            nn.Linear(256*16*16, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            nn.Linear(256, 2)
        )

        self.cnn.apply(self.gaussian_weights_init)
        self.fc.apply(self.gaussian_weights_init)

    def gaussian_weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 and classname.find('Conv') == 0:
            m.weight.data.normal_(0.0, 0.02)

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

class MyDataset(Dataset):
    def __init__(self, train_data, label):
        self.train_data = train_data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):

        return torch.FloatTensor(self.train_data[idx].astype('Float64')/255), torch.LongTensor(self.label[idx])


def main():
    train_data, val_data, train_labels, val_labels = load_data_from_npy(
        'data/train_data.npy', 'data/train_label.npy', val=0.2)

    print(train_data.shape[0])
    print(train_labels.shape[0])

    train_data_n = train_data.shape[0]
    val_data_n = val_data.shape[0]


    train_set = MyDataset(train_data, train_labels)
    test_set = MyDataset(val_data, val_labels)
    train_iter = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    print(train_iter)
    model = Classifier().cuda()
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    best_acc = 0
    for epoch in range(num_epochs):
        start = time.time()
        train_loss, test_losses = 0, 0
        train_acc, test_acc = 0, 0
        n, m = 0, 0
        for feature, label in train_iter:
            label = label.squeeze(1)
            #print('feature shape',feature.shape)
            #print('label shape',label.shape)
            model.train()
            n += 1
            optimizer.zero_grad()
            score = model(feature.cuda())
            loss = loss_function(score, label.cuda())
            loss.backward()
            optimizer.step()
            train_acc += np.sum(np.argmax(score.cpu().data.numpy(),
                                          axis=1) == label.cpu().data.numpy())
            train_loss += loss.item()

        for test_feature, test_label in test_iter:
            test_label = test_label.squeeze(1)
            model.eval()
            m += 1
            test_score = model(test_feature.cuda())
            test_loss = loss_function(test_score, test_label.cuda())
            test_acc += np.sum(np.argmax(test_score.cpu().data.numpy(),
                                         axis=1) == test_label.cpu().data.numpy())
            test_losses += test_loss.item()
        if (test_acc/m > best_acc):
            torch.save(model.state_dict(), output_model_path)
            best_acc = test_acc/m
            print('** Best Model Updated! ***\n')
        end = time.time()
        runtime = end - start
        print('epoch: %d, train loss: %.6f, train acc: %.5f, test loss: %.6f, test acc: %.5f, time: %.3f' %
              (epoch, train_loss / n, train_acc / train_data_n, test_losses / m, test_acc / val_data_n, runtime))


if __name__ == '__main__':
    main()
