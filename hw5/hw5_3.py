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
import gc
import imageio

mean=[0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ])
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

def readfile_from_img(file_path, label_path):
    # images = 
    for i in range(1):
        file_name = file_path+'/%03d.png'%(i)
        img = Image.open(file_name)
        img = preprocess(img)

    
    label = np.genfromtxt(label_path, delimiter=",")
    
    return img, label

def load_img(idx):
    file_name = 'hw5_data/images/%03d'%(idx)+'.png'
    img = Image.open(file_name)
    img = preprocess(img)
    return img

def load_data(train_path, label_path):
    label = np.genfromtxt(label_path, delimiter=",")
    train_data = []

    for i in range(200):
        file_name = train_path+'/%03d.png'%(i)
        img = Image.open(file_name)
        img = preprocess(img)
        img = np.array(img)
        train_data.append(img)

    train_data = np.array(train_data, dtype = 'int32')

#     train_data = torch.FloatTensor(train_data)
    train_data.requires_grad = True
    label = torch.LongTensor(label)

    return train_data, label
  
def load_data_from_np(train_path, label_path):
    label = np.genfromtxt(label_path, delimiter=",")

    train_data = np.load(train_path)

#     train_data = torch.FloatTensor(train_data)
#     train_data.requires_grad = True
    label = torch.LongTensor(label)

    return train_data, label
def main():

    model = models.resnet50(pretrained=True).cuda()
    model.eval()
    
    origin_images, labels = load_data_from_np('/content/drive/My Drive/ML2019/hw5/train_data_raw_int32.npy', '/content/drive/My Drive/ML2019/hw5/hw5_data/my_labels.csv')
#     print(images[0])
    images = np.array(origin_images, dtype = 'int32')
#     images = Variable(images, requires_grad=True)

    limits = np.zeros((200))
    for l in limits:
        l = 0.01
    
    num_epoch = 50
    batch_size = 40
    lr = 100
    limit = 3
    grad_var = (np.ones((batch_size, 224, 224, 3))/10e8)
    
#     train_set = TensorDataset(images, labels)     
#     train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=8)
    
    loss = torch.nn.CrossEntropyLoss()
    start_time = time.time()
    gc.enable()
    
    for epoch in range(num_epoch):
        train_acc = 0
        x_grad = 0
        update_value = 0
#         batch_images = Variable(images[0:batch_size], requires_grad = True)
        for i in range(round(len(images)/batch_size)):
#             msg = 'Solving (%02d/%02d)'%(i+1,round(len(images)/batch_size) )
#             print(msg, end='', flush=True)
#             back = '\b' * len(msg)
#             print(back, end='', flush=True)
#             batch_images = 
            _batch_images = np.array(images[i*batch_size:i*batch_size+batch_size])
            batch_images = []
            for ba in range(batch_size):
#                 print(_batch_images[ba].shape)
                batch_images.append( preprocess(Image.fromarray(_batch_images[ba].astype('uint8'), mode = "RGB")).numpy())
#                 print(batch_images_2.shape)
            batch_images = np.array(batch_images)
            batch_images = torch.FloatTensor(batch_images)
            batch_images = Variable(batch_images, requires_grad = True)
#             batch_images = images.data[i*batch_size:i*batch_size+batch_size]
#             batch_images.is_leaf=True
#             batch_images.requires_grad=True
    
#             print(batch_images.shape)
            train_pred = model(batch_images.cuda())
#             print(np.argmax(train_pred.cpu().data.numpy()))
            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == labels[i*batch_size:i*batch_size+batch_size].numpy())

            batch_loss = loss(train_pred, labels[i*batch_size:i*batch_size+batch_size].cuda())
            batch_loss.backward(retain_graph=True)
            # print(batch_loss.requires_grad)
            # print(batch_loss.is_leaf)
            x_grad = np.transpose(batch_images.grad.data.numpy(),(0,2,3,1))  
#             print(x_grad.shape)
            grad_var += x_grad*x_grad
            update_value = _batch_images + lr*x_grad/grad_var - origin_images[i*batch_size:i*batch_size+batch_size]
            
            # print(update_value.shape)
#             for up in range(batch_size):
            #     update_value[up] = torch.clamp(update_value[up], min=-1*limits[i*batch_size+up], max=limits[i*batch_size+up])           
            update_value = np.clip(update_value, -1*limit, limit)
#             print('update:',update_value.max())
            # print(update_value)
#             print(update_value.detach().numpy().max())
            
            images[i*batch_size:i*batch_size+batch_size] = np.round(update_value) + origin_images[i*batch_size:i*batch_size+batch_size]
            images[i*batch_size:i*batch_size+batch_size] = np.clip(images[i*batch_size:i*batch_size+batch_size], 0, 255)
#             print('image:',(images[i*batch_size:i*batch_size+batch_size]-origin_images[i*batch_size:i*batch_size+batch_size]).max())
#             print('origin image:', origin_images[0][0][0])
            gc.collect()
            
        train_acc = train_acc/len(images)
#         print(gc.get_referrers())
        msg = 'Epoch[%03d/%03d] acc = %1.6f  %4.2f sec(s)' \
        % (epoch+1, num_epoch, 1-train_acc, (time.time() - start_time))
        print(msg, flush=True)
        # back = '\b' * len(msg)
        # print(back, end='', flush=True)
#         print('max L: ',np.absolute(images-origin_images).max())
#         print('image_max; ', images.max(),'image_min; ', images.min())
#         print(origin_images.dtype)
#         print(images.dtype)
    print('Train done! Now saving...')
    for i in range(len(images)):
#         x_adv = images[i].squeeze(0)
        x_adv = images[i]
#         x_adv = x_adv.mul(torch.FloatTensor(std).view(3, 1, 1)).add(torch.FloatTensor(mean).view(3, 1, 1)).detach().numpy()
#         x_adv = np.transpose(x_adv, (1, 2, 0))
#         x_adv = scipy.misc.toimage(x_adv, cmin=0, cmax=255)
#         x_adv = Image.fromarray(x_adv.astype('uint8'), mode = 'RGB')
        imageio.imwrite('/content/drive/My Drive/ML2019/hw5/output/%03d' % (i) + '.png', x_adv.astype('uint8'))
        
    print('Save done!')
if __name__ == '__main__':
    main()