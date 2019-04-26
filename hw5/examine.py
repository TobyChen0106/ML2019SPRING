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
import matplotlib.pyplot as plt

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
    
    img_dir = sys.argv[1]  #new images directory
    L_sum = 0
    
    acc_rec = []
    train_acc = 0

    while (True):
        _id = input('input image idx: ')
        if (_id == 'q' or _id == 'Q'):
            break
        try:
            _id = int(_id)
        except:
            continue
        img = load_img_tensor(_id, img_dir)
        
        image_tensor = img.unsqueeze(0) # add batch dimension.  C X H X W ==> B X C X H X W
        img_variable = Variable(image_tensor, requires_grad=True) #convert tensor into a variable
        train_pred = model(img_variable)

        # PREDICT
        label_ids = np.argsort(np.squeeze(train_pred.data.detach().numpy(),axis = 0)) #get an index(class number) of a largest element
        print(label_ids[-1], label_ids[-2],label_ids[-3])
        # print('label ', label_idx)
        
        output_probs = F.softmax(train_pred, dim=1)
        x_pred_probs =  np.round(np.squeeze(output_probs.data.detach().numpy(),axis = 0),4)
        msg =  'image: %s%03d.png\n \
                #1 [%3d] %s (%3.8f %%)\n \
                #2 [%3d] %s (%3.8f %%)\n \
                #3 [%3d] %s (%3.8f %%)' \
                % (img_dir, _id, \
                    label_ids[-1], data_labels[int(label_ids[-1])], output_probs[0,label_ids[-1]]*100, \
                    label_ids[-2], data_labels[int(label_ids[-2])], output_probs[0,label_ids[-2]]*100, \
                    label_ids[-3], data_labels[int(label_ids[-3])], output_probs[0,label_ids[-3]]*100)
        print(msg)

        # bar(x, height, width=0.8, bottom=None, *, align='center', data=None, **kwargs)
        fig, ax = plt.subplots()

        index = np.arange(3)
        bar_width = 0.35

        opacity = 0.4
        error_config = {'ecolor': '0.3'}
        
        fig_y = [   output_probs[0,label_ids[-1]].detach().numpy()*100, \
                    output_probs[0,label_ids[-2]].detach().numpy()*100,\
                    output_probs[0,label_ids[-3]].detach().numpy()*100]  
        rects1 = ax.bar(index, fig_y, bar_width,
                        alpha=opacity, color='b',
                        error_kw=error_config,
                        label='Probability')

        ax.set_ylabel('Probability')
        ax.set_title('%s: %2.2f%%'%(data_labels[int(label_ids[-1])], \
                                    output_probs[0,label_ids[-1]]*100), y=-0.2)
        ax.set_xticks(index)
        ax.set_xticklabels((    data_labels[int(label_ids[-1])], \
                                data_labels[int(label_ids[-2])], \
                                data_labels[int(label_ids[-3])]))
        for i, v in enumerate(fig_y):
            ax.text(i -0.12, v + 1 , str(np.round(v,2)), color='blue', fontweight='bold')
        
        fig.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()


    
