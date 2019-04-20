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

def load_img_raw(idx, dir):
    file_name = dir+'%03d'%(idx)+'.png'
    img = Image.open(file_name)
    # img = preprocess(img)
    return np.array(img, dtype = 'int32')

def load_img_tensor(idx, dir):
    file_name = dir+'%03d'%(idx)+'.png'
    img = Image.open(file_name)
    img = preprocess(img)
    return img
    
def main():
    model = models.resnet50(pretrained=True)
    model.eval()

    labels_link = "https://savan77.github.io/blog/files/labels.json"    
    labels_json = requests.get(labels_link).json()
    data_labels = {int(idx):label for idx, label in labels_json.items()}
    labels = np.genfromtxt('hw5_data/my_labels.csv', delimiter=",")
  
    acc_rec = []
    # train_acc = 0
    progress = ['/', '-', '\\', '|']
        
    img_121_origin = load_img_raw(121, 'hw5_data/images/')
    new_img = np.array(img_121_origin)

    for i in range(1000):
        msg = 'trying %s %6d entries...' % (  progress[(i+1) % 4], i+1)
        print(msg, end = '', flush  = True)
        back = '\b' * len(msg)
        print(back, end = '', flush  = True)
        
        addition_mask = np.random.random_integers(low=0, high=2, size=(224, 224, 3)) - 1
        # print(addition_mask)
        new_img = np.array(img_121_origin + addition_mask)
        new_img = np.clip(new_img, 0, 255)

        image_tensor = preprocess(Image.fromarray(new_img.astype('uint8'), mode = "RGB"))
        image_tensor = image_tensor.unsqueeze(0) # add batch dimension.  C X H X W ==> B X C X H X W
        img_variable = Variable(image_tensor, requires_grad=True) #convert tensor into a variable
        
        train_pred = model(img_variable)
        train_acc = np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == labels[121])
        # acc_rec.append(np.argmax(train_pred.cpu().data.numpy(), axis=1) == labels[i])
        if (train_acc == 0):
            print('\nSuccess!')
            break
        gc.collect()
    
    print('Now Saving...')
    imageio.imwrite('121.png', new_img.astype('uint8'))

if __name__ == '__main__':
    main()


    
