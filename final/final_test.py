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
import cv2
import gc

cnn_model_path = 'models/best_model_lily.pth'
img_path = 'data/train/train'
test_img_num = 100  ## 4998
test_img_num_start = int(sys.argv[1])  ## 4998
test_img_num_end = int(sys.argv[1])+test_img_num  ## 4998

if test_img_num_end>4998:
    test_img_num_end = 4998
batch_size = 19
map_stride = 16
n = 896//map_stride + 1

def load_img(path):
    print('start loading image[%d]...'%test_img_num_start)
    img_ar = []
    for i in range(test_img_num_start, test_img_num_end):
        # if i%1 == 0:
            # print('loading img', i)
        img_path = path + '%05d.png'%i
        # img = Image.open(img_path)
        img = cv2.imread(img_path, 0)
        # print(img.shape)
        img_ar.append(np.array(img).reshape(1024, 1024, 1))
        gc.collect()
        msg = "loading image [%06d/%06d]" % (i+1, test_img_num_end )
        print(msg, end='', flush=True)
        back = '\b'*len(msg)
        print(back, end='', flush=True)
    img_ar = np.array(img_ar)
    print('\nimg_size = ', img_ar.shape)
    # np.save('/content/drive/My Drive/ML/hw7/data/images.npy', img_ar)
    print("finish loading image!\n")
    return img_ar

def create_map(model, img):
    print('start creating map...')
    print('n = ', n)
    
    for id in range(0, test_img_num_end-test_img_num_start): ## 4998
        test_features = []
        msg = "solving [%06d/%06d]" % (id+1, test_img_num_end-test_img_num_start)
        print(msg, end='', flush=True)
        back = '\b'*len(msg)
        print(back, end='', flush=True)
        for i in range(n):
            for j in range(n):
                test_features.append(img[id, i*map_stride : i*map_stride + 128, j*map_stride : j*map_stride + 128])
        test_features = np.array(test_features).reshape(-1, 1, 128, 128)
        # print('test_feature shape = ', test_features.shape) ## [4998*n*n, 128, 128, 3]
        test_set = TensorDataset(torch.FloatTensor(test_features)/255)
        test_iter = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    
        map = []
        for test_feature in test_iter:
            score = model(test_feature[0].cuda())
    #         sc = score[:, 1] - score[:, 0]
    #         print('score size = ', score.shape)
    #         print(np.array(sc.cpu().data.numpy()).reshape(-1).shape)
            map.append(np.argmax(score.cpu().data.numpy(), axis = 1).reshape(-1))
        map = np.array(map).reshape(n, n, 1)
        # print('map shape = ', map.shape) ## should be [4998, n*n]
#     x = np.tanh(map[0,:,:,0])
#     x = (x+1)*127.5
#     x = x.transpose((2, 0, 1))
    # for id in range(test_img_num):
        

        x = map[:,:,0]*255
        x = x.astype('uint8')
    #     print('x type = ', x.dtype)
    #     print('x shape = ', x.shape)
    # #     x = x.permute()
    #     print('max = ', x.max())
    #     print('min = ', x.min())
    #     print('x = ', x)
        image = Image.fromarray(x)
        image.save('train_map/train_img' + str(id+test_img_num_start) + '.png')
        # print(x)
        gc.collect()
    return map 



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

def main():
    all_img = load_img(img_path)
    cnn_model = Classifier().cuda()
#     cnn_model.load_state_dict(torch.load(cnn_model_path, map_location = {'cuda:0': 'cpu'}))
    cnn_model.load_state_dict(torch.load(cnn_model_path))
    cnn_model.eval()
    map = create_map(cnn_model, all_img)
    # imgOut = selective_search(map)

if __name__ == '__main__':
    main()