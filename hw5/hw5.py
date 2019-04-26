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

mean=[0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ])

def readfile_from_img(file_path, label_path):
    for i in range(1):
        file_name = file_path+'/%03d'%(i)+'.png'
        img = Image.open(file_name)
        img = preprocess(img)
    
    label = np.genfromtxt(label_path, delimiter=",")
    
    return img, label

def load_img(idx):
    global input_dir
    global output_dir
    # file_name = 'hw5_data/images/%03d'%(idx)+'.png'
    file_name = os.path.join(input_dir,('%03d.png'%(idx)))
    img = Image.open(file_name)
    img = preprocess(img)
    return img

def main():
    global input_dir
    global output_dir
    vgg19 = models.resnet50(pretrained=True)
    vgg19.eval()
    labels = np.genfromtxt('labels.csv', delimiter=",", skip_header = 1, usecols = 3)
    start_time = time.time()

    for i in range(200):
        progress = (u"\u2588" * (int(float(i)/200*40))).ljust(40,'.')
        msg = 'Solving [%3d/%03d] %2.2f sec(s) |%s|' % (i+1, 200, \
                (time.time() - start_time), progress)
        print(msg, end='', flush=True)
        back = '\b' * len(msg)
        print(back, end='', flush=True)
                     
        # image_tensor, labels = readfile_from_img('hw5_data/images', 'hw5_data/my_labels.csv')
        image_tensor = load_img(i)
        # print(image_tensor.shape)
        image_tensor = image_tensor.unsqueeze(0) # add batch dimension.  C X H X W ==> B X C X H X W
        img_variable = Variable(image_tensor, requires_grad=True) #convert tensor into a variable

        output = vgg19.forward(img_variable)
        loss = torch.nn.CrossEntropyLoss()
        loss_cal = loss(output, torch.LongTensor([labels[i]]))
        loss_cal.backward(retain_graph=True)
        
        eps = 0.097
        x_grad = torch.sign(img_variable.grad.data)                #calculate the sign of gradient of the loss func (with respect to input X) (adv)
        x_adv = img_variable.data + eps * x_grad
        
        x_adv = x_adv.squeeze(0)
        x_adv = x_adv.mul(torch.FloatTensor(std).view(3, 1, 1)).add(torch.FloatTensor(mean).view(3, 1, 1)).numpy()
        x_adv = np.transpose(x_adv, (1, 2, 0))
        scipy.misc.imsave(os.path.join(output_dir,'%03d.png' % (i)), x_adv)
        
    print('\n done!')
if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    main()
