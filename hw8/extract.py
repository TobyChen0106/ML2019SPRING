from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader
import sys
import numpy as np
import pandas as pd
from PIL import Image
import io

def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(oup),
                nn.Dropout(0.2)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(inp),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(oup),
                nn.Dropout(0.2)
            )

        self.model = nn.Sequential(
            conv_bn(1,  16, 2),     # 16, 24, 24
            conv_dw(16, 32, 1),    # 32, 24, 24
            conv_dw(32, 32, 1),    # 32, 24, 24
            conv_dw(32, 64, 2),   # 64, 12, 12
            conv_dw(64, 64, 1),    # 64, 12, 12
            conv_dw(64, 128, 1),    # 128, 12, 12
            conv_dw(128, 128, 1),   # 128, 12, 12
            conv_dw(128, 128, 2),   # 128, 6, 6
            conv_dw(128, 128, 1),   # 128, 6, 6
            conv_dw(128, 256, 1),   # 256, 6, 6


            nn.AvgPool2d(6),
        )
        self.fc = nn.Sequential(
            # nn.Linear(512, 512),
            # nn.LeakyReLU(0.2),
            # nn.Dropout(p=0.5),
            nn.Linear(256, 7),
            nn.Dropout(0.2)        )
        self.model.apply(gaussian_weights_init)
        self.fc.apply(gaussian_weights_init)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x

def extract(npzfile):
    model = Net().cuda()

    weight_log = np.array(
        [[16, 1, 3, 3], [16],

         [16, 1, 3, 3], [16], [32, 16, 1, 1], [32],
         [32, 1, 3, 3], [32], [32, 32, 1, 1], [32],
         [32, 1, 3, 3], [32], [64, 32, 1, 1], [64],
         [64, 1, 3, 3], [64], [64, 64, 1, 1], [64],
         [64, 1, 3, 3], [64], [128, 64, 1, 1], [128],
         [128, 1, 3, 3], [128], [128, 128, 1, 1], [128],
         [128, 1, 3, 3], [128], [128, 128, 1, 1], [128],
         [128, 1, 3, 3], [128], [128, 128, 1, 1], [128],
         [128, 1, 3, 3], [128], [256, 128, 1, 1], [256],

         [7, 256]])

    # bias_log = np.array([[64], [64], [190], [190], [512], [7]])
    bias_log = np.array(
        [[16], [16], [16],

         [16], [16], [16], [32], [32], [32],
         [32], [32], [32], [32], [32], [32],
         [32], [32], [32], [64], [64], [64],
         [64], [64], [64], [64], [64], [64],
         [64], [64], [64],  [128], [128], [128],
         [128], [128], [128], [128], [128], [128],
         [128], [128], [128], [128], [128], [128],
         [128], [128], [128], [128], [128], [128],
         [128], [128], [128], [256], [256], [256],

         [7]])

    new_param = np.load(npzfile)
    new_weight = new_param['weights']
    new_bias = new_param['bias']

    # print(new_weight.shape)
# layer 0
    weights = np.zeros(weight_log[0])
    model.model[0][0].weight.data = extract_tensor(
        new_weight[:int(weights.flatten().shape[0])]).reshape(weights.shape)
    # weight_log.append(weights.shape)
    new_weight = new_weight[int(weights.flatten().shape[0]):]

    weights = np.zeros(weight_log[1])
    model.model[0][2].weight.data = extract_tensor(
        new_weight[:int(weights.flatten().shape[0])]).reshape(weights.shape)
    # weight_log.append(weights.shape)
    new_weight = new_weight[int(weights.flatten().shape[0]):]

    bias = np.zeros(bias_log[0])
    model.model[0][2].bias.data = extract_tensor(
        new_bias[:int(bias.flatten().shape[0])]).reshape(bias.shape)
    # bias_log.append(bias.shape)
    new_bias = new_bias[int(bias.flatten().shape[0]):]

    bias = np.zeros(bias_log[1])
    model.model[0][2].running_mean.data = extract_tensor(
        new_bias[:int(bias.flatten().shape[0])]).reshape(bias.shape)
    # bias_log.append(bias.shape)
    new_bias = new_bias[int(bias.flatten().shape[0]):]

    bias = np.zeros(bias_log[2])
    model.model[0][2].running_var.data = extract_tensor(
        new_bias[:int(bias.flatten().shape[0])]).reshape(bias.shape)
    # bias_log.append(bias.shape)
    new_bias = new_bias[int(bias.flatten().shape[0]):]

# layer dw 1~9
    model, new_weight, new_bias = modify_dw(
        1, weight_log, bias_log, model, new_weight, new_bias)
    model, new_weight, new_bias = modify_dw(
        2, weight_log, bias_log, model, new_weight, new_bias)
    model, new_weight, new_bias = modify_dw(
        3, weight_log, bias_log, model, new_weight, new_bias)
    model, new_weight, new_bias = modify_dw(
        4, weight_log, bias_log, model, new_weight, new_bias)
    model, new_weight, new_bias = modify_dw(
        5, weight_log, bias_log, model, new_weight, new_bias)
    model, new_weight, new_bias = modify_dw(
        6, weight_log, bias_log, model, new_weight, new_bias)
    model, new_weight, new_bias = modify_dw(
        7, weight_log, bias_log, model, new_weight, new_bias)
    model, new_weight, new_bias = modify_dw(
        8, weight_log, bias_log, model, new_weight, new_bias)
    model, new_weight, new_bias = modify_dw(
        9, weight_log, bias_log, model, new_weight, new_bias)

# layer fc
    weights = np.zeros(weight_log[38])
    model.fc[0].weight.data = extract_tensor(
        new_weight[:int(weights.flatten().shape[0])]).reshape(weights.shape)
    # weight_log.append(weights.shape)
    new_weight = new_weight[int(weights.flatten().shape[0]):]

    bias = np.zeros(bias_log[57])
    model.fc[0].bias.data = extract_tensor(
        new_bias[:int(bias.flatten().shape[0])]).reshape(bias.shape)
    # bias_log.append(bias.shape)
    new_bias = new_bias[int(bias.flatten().shape[0]):]

    print(new_bias)
    print(new_weight)
    return model.cuda()


def extract_tensor(new_weight):
    # return torch.FloatTensor(new_weight.astype('float32')/65535*(max-min)+min)
    return torch.FloatTensor(new_weight.astype('float32'))


def modify_dw(id, weight_log, bias_log, model, new_weight, new_bias):
    w_start = (id-1)*4+2
    b_start = (id-1)*6+3
    weights = np.zeros(weight_log[w_start])
    model.model[id][0].weight.data = extract_tensor(
        new_weight[:int(weights.flatten().shape[0])]).reshape(weights.shape)
    # weight_log.append(weights.shape)
    new_weight = new_weight[int(weights.flatten().shape[0]):]

    weights = np.zeros(weight_log[w_start+1])
    model.model[id][2].weight.data = extract_tensor(
        new_weight[:int(weights.flatten().shape[0])]).reshape(weights.shape)
    # weight_log.append(weights.shape)
    new_weight = new_weight[int(weights.flatten().shape[0]):]

    bias = np.zeros(bias_log[b_start])
    model.model[id][2].bias.data = extract_tensor(
        new_bias[:int(bias.flatten().shape[0])]).reshape(bias.shape)
    # bias_log.append(bias.shape)
    new_bias = new_bias[int(bias.flatten().shape[0]):]

    bias = np.zeros(bias_log[b_start+1])
    model.model[id][2].running_mean.data = extract_tensor(
        new_bias[:int(bias.flatten().shape[0])]).reshape(bias.shape)
    # bias_log.append(bias.shape)
    new_bias = new_bias[int(bias.flatten().shape[0]):]

    bias = np.zeros(bias_log[b_start+2])
    model.model[id][2].running_var.data = extract_tensor(
        new_bias[:int(bias.flatten().shape[0])]).reshape(bias.shape)
    # bias_log.append(bias.shape)
    new_bias = new_bias[int(bias.flatten().shape[0]):]

    weights = np.zeros(weight_log[w_start+2])
    model.model[id][3].weight.data = extract_tensor(
        new_weight[:int(weights.flatten().shape[0])]).reshape(weights.shape)
    # weight_log.append(weights.shape)
    new_weight = new_weight[int(weights.flatten().shape[0]):]

    weights = np.zeros(weight_log[w_start+3])
    model.model[id][5].weight.data = extract_tensor(
        new_weight[:int(weights.flatten().shape[0])]).reshape(weights.shape)
    # weight_log.append(weights.shape)
    new_weight = new_weight[int(weights.flatten().shape[0]):]

    bias = np.zeros(bias_log[b_start+3])
    model.model[id][5].bias.data = extract_tensor(
        new_bias[:int(bias.flatten().shape[0])]).reshape(bias.shape)
    # bias_log.append(bias.shape)
    new_bias = new_bias[int(bias.flatten().shape[0]):]

    bias = np.zeros(bias_log[b_start+4])
    model.model[id][5].running_mean.data = extract_tensor(
        new_bias[:int(bias.flatten().shape[0])]).reshape(bias.shape)
    # bias_log.append(bias.shape)
    new_bias = new_bias[int(bias.flatten().shape[0]):]

    bias = np.zeros(bias_log[b_start+5])
    model.model[id][5].running_var.data = extract_tensor(
        new_bias[:int(bias.flatten().shape[0])]).reshape(bias.shape)
    # bias_log.append(bias.shape)
    new_bias = new_bias[int(bias.flatten().shape[0]):]

    # print(new_bias.shape)
    return model, new_weight, new_bias