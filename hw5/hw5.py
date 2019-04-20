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
    file_name = 'hw5_data/images/%03d'%(idx)+'.png'
    img = Image.open(file_name)
    img = preprocess(img)
    return img

def main():
    vgg19 = models.resnet50(pretrained=True)
    vgg19.eval()
    labels = np.genfromtxt('hw5_data/my_labels.csv', delimiter=",")
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

        # *****predict*****
        label_idx = torch.max(output.data, 1)[1][0].numpy()   #get an index(class number) of a largest element
        
        labels_link = "https://savan77.github.io/blog/files/labels.json"    
        labels_json = requests.get(labels_link).json()
        labels = {int(idx):label for idx, label in labels_json.items()}
        x_pred = labels[int(label_idx)]

        output_probs = F.softmax(output, dim=1)
        x_pred_prob =  round((torch.max(output_probs.data, 1)[0][0]).numpy() * 100,4)
        print(label_idx,' ',x_pred, ' ',  x_pred_prob ,'%')
        
        # y_true = labels[i]   
        # target = Variable(torch.LongTensor([y_true]), requires_grad=False)

        # loss = torch.nn.CrossEntropyLoss()
        # loss_cal = loss(output, target)
        # loss_cal.backward(retain_graph=True)
        
        # eps = 0.085
        # x_grad = torch.sign(img_variable.grad.data)                #calculate the sign of gradient of the loss func (with respect to input X) (adv)
        # x_adv = img_variable.data + eps * x_grad
        
        # x_adv = x_adv.squeeze(0)
        # x_adv = x_adv.mul(torch.FloatTensor(std).view(3, 1, 1)).add(torch.FloatTensor(mean).view(3, 1, 1)).numpy()
        # x_adv = np.transpose(x_adv, (1, 2, 0))
        # scipy.misc.imsave('output/%03d' % (i) + '.png', x_adv)
        
    print('\n done!')
if __name__ == '__main__':
    main()