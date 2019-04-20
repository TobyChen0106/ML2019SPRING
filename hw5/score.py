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
    
    img_1_dir = 'hw5_data/images/' #origin images directory
    img_2_dir = sys.argv[1]  #new images directory
    L_sum = 0
    
    acc_rec = []
    train_acc = 0
    progress = ['/', '-', '\\', '|']
        
    for i in range(200):
        msg = 'solving [%03d%s%03d] ' % (  i + 1,progress[(i+1) % 4], 200)
        print(msg, end = '', flush  = True)
        back = '\b' * len(msg)
        print(back, end = '', flush  = True)
        
        img_1 = load_img_raw(i, img_1_dir)
        img_2 = load_img_raw(i, img_2_dir)
        img_2_tensor =  load_img_tensor(i, img_2_dir)
        dif = np.absolute(img_2-img_1).max()
        # print('[%00d]: %d' % (i, int(dif)))
        L_sum += dif
        
        image_tensor = img_2_tensor.unsqueeze(0) # add batch dimension.  C X H X W ==> B X C X H X W
        img_variable = Variable(image_tensor, requires_grad=True) #convert tensor into a variable
        train_pred = model(img_variable)
        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == labels[i])
        acc_rec.append(np.argmax(train_pred.cpu().data.numpy(), axis=1) == labels[i])

        # PREDICT
        label_idx = np.argsort(np.squeeze(train_pred.data.detach().numpy(),axis = 0))[-2]  #get an index(class number) of a largest element
        # second_id.append(label_idx)
        # print('output shape ',np.argsort(np.squeeze(output.data.detach().numpy(),axis = 0)))
        # print('label ', label_idx)
        
        # x_pred = data_labels[int(label_idx)]
        # output_probs = F.softmax(output, dim=1)
        # x_pred_prob =  round(np.squeeze(output_probs.data.detach().numpy(),axis = 0)[label_idx] * 100,4)
        # print('image ',i,' : [',label_idx,'] ',x_pred, ' ',  x_pred_prob ,'%')
        
    train_acc = 1 - train_acc / 200
    print('\nSuccess Rate: %f'%train_acc)
    print('Total L-infinity: %d'%(L_sum))
    print('Average L-infinity: %3.6f' % (L_sum / 200))
    
if __name__ == '__main__':
    main()


    
