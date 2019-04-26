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
import os
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
def load_data(train_path, label_path):
    global input_dir
    global output_dir
    label = np.genfromtxt(label_path, delimiter=",", skip_header = 1, usecols = 3)
    
    train_data = []

    for i in range(200):
        file_name = os.path.join(input_dir, '%03d.png'%(i))
        img = Image.open(file_name)
        img = np.array(img)
        train_data.append(img)
    
    #train_data = np.transpose(np.array(train_data, dtype = 'int32'), (0, 2, 3, 1))
    train_data = np.array(train_data, dtype = 'int32')
    label = torch.LongTensor(label)

    return train_data, label
  
def load_data_from_np(train_path, label_path):
    label = np.genfromtxt(label_path, delimiter=",")

    train_data = np.load(train_path)
    print(train_data[0])
#     train_data = torch.FloatTensor(train_data)
#     train_data.requires_grad = True
    label = torch.LongTensor(label)

    return train_data, label
def main():
    global input_dir
    global output_dir

    model = models.resnet50(pretrained=True).cuda()
    model.eval()
    
    origin_images, labels = load_data(input_dir, 'labels.csv')
    #origin_images, labels = load_data_from_np('/content/drive/My Drive/ML2019/hw5/train_data_raw_int32.npy', '/content/drive/My Drive/ML2019/hw5/hw5_data/my_labels.csv')
    images = np.array(origin_images, dtype = 'float')
    # s_labels = np.load('/content/drive/My Drive/ML2019/hw5/2_labels.npy')
    s_labels = np.load('2_labels.npy')
    s_labels = torch.LongTensor(s_labels)
    
    num_L_size = 2
    num_epoch = 100
    batch_size = 40
    lr = 1
    limit = 1
    grad_var = (np.ones((batch_size, 224, 224, 3))/10e8)
    
    limits = np.zeros((200))
    for i in range(len(limits)):
        limits[i] = limit
    
    loss = torch.nn.CrossEntropyLoss()
    start_time = time.time()
    gc.enable()
    for l_in in range(num_L_size):
        for epoch in range(num_epoch):
            train_acc = 0
            acc_rec = []
            for i in range(round(len(images)/batch_size)):
                _batch_images = np.array(images[i*batch_size:i*batch_size+batch_size])
                batch_images = []
                for ba in range(batch_size):
                    batch_images.append( preprocess(Image.fromarray(_batch_images[ba].astype('uint8'), mode = "RGB")).numpy())

                batch_images = np.array(batch_images)
                batch_images = torch.FloatTensor(batch_images)
                batch_images = Variable(batch_images, requires_grad = True)

                train_pred = model(batch_images.cuda())

                train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == labels[i*batch_size:i*batch_size+batch_size].numpy())
                acc_rec.append( np.argmax(train_pred.cpu().data.numpy(), axis=1) == labels[i*batch_size:i*batch_size+batch_size].numpy() )

                batch_loss = loss(train_pred, labels[i*batch_size:i*batch_size+batch_size].cuda())-loss(train_pred, s_labels[i*batch_size:i*batch_size+batch_size].cuda())
                batch_loss.backward(retain_graph=True)
                
                x_grad = np.transpose(batch_images.grad.data.numpy(),(0,2,3,1))  
                #x_grad = batch_images.grad.data.numpy()
                #print(x_grad.shape)
                
                grad_var += x_grad*x_grad
                update_value = _batch_images + lr*x_grad/grad_var - origin_images[i*batch_size:i*batch_size+batch_size]

                for up in range(batch_size):
                  update_value[up] =  np.clip(update_value[up], -1*limits[i*batch_size+up], limits[i*batch_size+up])

                images[i*batch_size:i*batch_size+batch_size] = (update_value) + origin_images[i*batch_size:i*batch_size+batch_size]
                images[i*batch_size:i*batch_size+batch_size] = np.clip(images[i*batch_size:i*batch_size+batch_size], 0, 255)
                gc.collect()
        
            train_acc = train_acc/len(images)
            acc_rec = np.reshape(np.array(acc_rec), (200))

            indices = (acc_rec == 1).nonzero()[0]
            print('NOT SOLVED: ',indices)

            msg = 'L[%02d/%02d] Epoch[%03d/%03d] acc = %1.6f  %4.2f sec(s)' \
            % (l_in+1, num_L_size, epoch+1, num_epoch, 1-train_acc, (time.time() - start_time))
            print(msg, flush=True)
#         grad_var = (np.ones((batch_size, 224, 224, 3))/10e8)
        limits = limits + (acc_rec)*int(1)
        indices = (acc_rec == 1).nonzero()[0]
        print('NOT SOLVED: ',indices)
        print('NOT SOLVED L: ',limits[indices])
    images = np.round(images)
    print('Train done! Now saving...')
    for i in range(len(images)):
        x_adv = np.array(images[i], dtype = 'uint8')
        imageio.imwrite(os.path.join(output_dir, '%03d.png' % (i)), x_adv)
        
    print('Save done!')
if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    main()