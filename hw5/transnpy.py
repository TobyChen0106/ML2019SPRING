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
def load_data(train_path, label_path):
    label = np.genfromtxt(label_path, delimiter=",")
    train_data = []

    for i in range(200):
        file_name = train_path+'/%03d.png'%(i)
        img = Image.open(file_name)
        img = preprocess(img)
        img = np.array(img)
        train_data.append(img)

    train_data = np.array(train_data)
    np.save('images',train_data)

    train_data = torch.FloatTensor(train_data)/255.0
    train_data.requires_grad = True
    label = torch.LongTensor(label)

    return train_data, label

images, labels = load_data('hw5_data/images/', 'hw5_data/my_labels.csv')