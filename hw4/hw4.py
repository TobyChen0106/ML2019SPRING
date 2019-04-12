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
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import os

try:
    import lime
except:
    sys.path.append(os.path.join('..', '..')) # add the current directory
    import lime
from lime import lime_image



mean=[0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ])
def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),  # [64, 48, 48]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0),      # [64, 24, 24]

            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0),      # [128, 12, 12]

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0)       # [256, 6, 6]
        )

        self.fc = nn.Sequential(
            nn.Linear(128*6*6, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            nn.Linear(512, 7)
        )
        self.cnn.apply(gaussian_weights_init)
        self.fc.apply(gaussian_weights_init)

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)
    def p2_forward(self, x):
        out = self.cnn[0](x)
        out = self.cnn[1](out)
        out = self.cnn[2](out)
        out = self.cnn[3](out)
        out = self.cnn[4](out)
        out = self.cnn[5](out)
        out = self.cnn[6](out)
        out = self.cnn[7](out)
        out = self.cnn[8](out)
        out = self.cnn[9](out)
        out = self.cnn[10](out)
        out = self.cnn[11](out)
        out = self.cnn[12](out)
        # out = self.cnn[13](out)
        # out = self.cnn[14](out)
        # out = self.cnn[15](out)

        return out

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

def readfile_from_np(test_file_path, label_file_path):
    # print("Reading CSV File...")
    
    test_data = np.load(test_file_path)
    labels = np.load(label_file_path)

    test_data = test_data.reshape(-1,1,48,48)

    test_data = np.array(test_data, dtype=float)/255
    test_data = torch.FloatTensor(test_data)
    labels = np.array(labels, dtype=int)
    labels = torch.LongTensor(labels)
    return test_data, labels
def readfile_from_csv_2(train_file_path):
    print("Reading csv File...")
   
    data = io.StringIO(open(train_file_path).read().replace(',',' '))
    train = np.genfromtxt(data, delimiter=' ', skip_header=1)
    
    test_data = train[:,1:]
    labels = train[:, 0]

    test_data = test_data.reshape(-1,1,48,48)

    test_data = np.array(test_data, dtype=float)/255
    test_data = torch.FloatTensor(test_data)
    labels = np.array(labels, dtype=int)
    labels = torch.LongTensor(labels)
    return test_data, labels

def readfile_from_csv(path):
    x_train = []
    y_train = []
    print ("Reading csv file...")
    with open(path, newline='') as csvfile:
        next(csvfile)
        rows = csv.reader(csvfile)
        for row in rows:
            y_train.append(int(row[0]))
            x_train.append(np.array( row[1].split() ).astype(np.float))

    x_train = np.reshape(np.array(x_train),[-1,48,48,1])
    y_train = np.array(y_train)

    test_data = x_train

    labels = y_train
    test_data = test_data.reshape(-1,1,48,48)

    test_data = np.array(test_data, dtype=float)/255
    test_data = torch.FloatTensor(test_data)
    labels = np.array(labels, dtype=int)
    labels = torch.LongTensor(labels)
    return test_data, labels

def compute_saliency_maps(X, y, model):

    X_var = Variable(X, requires_grad=True)
    y_var = Variable(y)
    
    scores = model(X_var)
    scores = scores.gather(1, y_var.view(-1, 1)).squeeze() 
    scores.backward(torch.FloatTensor([1.0,1.0,1.0,1.0,1.0]))
    saliency = X_var.grad.data
    saliency = saliency.abs()
    saliency, i = torch.max(saliency,dim=1)
    saliency = saliency.squeeze()
#     print(saliency)
    return saliency

def main():

    image_idxs = [9956, 19991, 5173, 2559, 27400, 2716, 28165]
    
    # input_files = sys.argv[1]
    # output_files = sys.argv[2]

    # train_data, all_label = readfile_from_np('train_data.npy', 'label.npy')
    train_data, all_label = readfile_from_csv(sys.argv[1])
    output_path = sys.argv[2]
    images = []
    labels = []
    for id in image_idxs:
        images.append(train_data[id])
        labels.append(all_label[id])
    
    global model

    
    saliency_images = []
    print ("Solving 1")
    images_p1 = images
    for i in range(len(images)):
        images_p1[i] = images_p1[i].unsqueeze(0) 
        # print(images[i].shape)
        saliency_images.append(compute_saliency_maps(images_p1[i], labels[i], model).numpy())
    
    w = np.array(saliency_images)
    for i in range(w.shape[0]):
        z = w[i]
        fig, ax = plt.subplots(1, 1 ,figsize=(8,6))
        # ax[0].set_title('Original Image')
        # ax[0].imshow(images[i].numpy().reshape((48, 48)), cmap="gray")
        ax.set_title('Saliency Map')
        cax = ax.imshow(z, cmap = 'jet')
        fig.colorbar(cax, ax = ax)
        print ("Save img ", i)
        plt.savefig(output_path+'fig1_%1d' % (i) + '.jpg')
        plt.clf()
    
    
    
    print ("Solving 2_1")
    noise = np.random.rand(1, 1, 48, 48)
    p2_image = torch.FloatTensor(noise)
    p2_image_variable = Variable(p2_image, requires_grad=True) #convert tensor into a variable

    epoch = 100
    lr = 0.1
    grad_var = 10e-8
    # optimizer = torch.optim.Adam([p2_image_variable], lr=0.1)
    
    # print(p2_image.shape)
    for e in range(epoch):
        # optimizer.zero_grad()
        # p2_image = model.my_forward(p2_image)
        
        out = model.p2_forward(p2_image_variable)
        loss = out.sum()
        # print(loss.requires_grad)
        # print(loss.is_leaf)

        loss.backward()
        # optimizer.step()
        
        _grad = p2_image_variable.grad.data
        # print('grad=',_grad)
        grad_var += (_grad.numpy()*_grad.numpy()).sum()
        
        p2_image += lr * _grad
        p2_image_variable = Variable(p2_image, requires_grad=True)
    # p2_image = p2_image_variable
    p2_image = model.p2_forward(p2_image)    
    p2_image = p2_image.squeeze(0)

    for i in range(8):
        for j in range(8):
            plt.figure(num='filters', figsize=(8, 8))
            plt.subplot(8, 8, 8 * i + j + 1)
            plt.axis('off') 
            plt.imshow(p2_image[i * 8 + j].detach().numpy().reshape((24, 24)), cmap = 'gray')
    # plt.show()
    plt.savefig(output_path+'fig2_1.jpg')
    plt.clf()
    
    print ("Solving 2_2")
    p22_img = images[0] 
    p22_img = model.p2_forward(p22_img)
    p22_img = p22_img.squeeze(0)
    print(p22_img.shape)

    for i in range(64):
        plt.figure(num='filters', figsize=(8, 8))
        plt.subplot(8, 8, i + 1)
        plt.axis('off') 
        plt.imshow(p22_img[i].detach().numpy().reshape((24, 24)), cmap = 'gray')
    # plt.show()
    plt.savefig(output_path+'fig2_2.jpg')
    plt.clf()
    

    np.random.seed(0)
    explainer = lime_image.LimeImageExplainer()
    for ip3 in range(len(images)):

        p_33img = np.zeros((48, 48, 3))
        for i in range(3):
            p_33img[:,:, i] = images[ip3].numpy().reshape(48, 48)
        # print(model_predict(np.expand_dims(p_33img, axis = 0)))

        # explanation = explainer.explain_instance(p_33img, model_predict, top_labels=5, hide_color=0, num_samples=1000)
        
        explaination = explainer.explain_instance(
                                image=p_33img, 
                                classifier_fn=model_predict,
                                segmentation_fn=segmentation
                            )
        p33_rt_image, mask = explaination.get_image_and_mask(
                                    label=ip3,
                                    positive_only=False,
                                    hide_rest=False,
                                    num_features=5,
                                    min_weight=0.0
                                )
        print('saving 3_%d'%ip3)
        # plt.imsave('output/3_%d' % (ip3), p33_rt_image, cmap = 'jet')
        # plt.imshow(mark_boundaries(p33_rt_image, mask).reshape((48, 48, 3)), cmap = 'jet')
        # plt.axis('off') 
        # plt.savefig('output/fig3_%1d' % (ip3) + '.jpg')
        # plt.clf()

        scipy.misc.imsave(output_path+'fig3_%d.jpg' % (ip3), mark_boundaries(p33_rt_image, mask).reshape((48, 48, 3)))
        
    print('done!')
'''
    for i in range(len(images)):
        x_adv = images[i]
        # x_adv = x_adv.mul(torch.FloatTensor(std).view(1, 1, 1)).add(torch.FloatTensor(mean).view(1, 1, 1)).numpy()
        print(x_adv.shape)
        # x_adv = np.transpose(x_adv, (1, 2, 0))
        scipy.misc.imsave('output/%02d' % (i) + '.png', x_adv)
'''    

def model_predict(np_in):
    global model
    
    img = np_in[0,:,:, 0]
    img_tnesor = torch.FloatTensor(img)
    img_tnesor = img_tnesor.unsqueeze(0)
    img_tnesor = img_tnesor.unsqueeze(0)
    
    m = nn.Softmax(dim=1)
    ans = m(model(img_tnesor)).detach().numpy().reshape(7)
    rt = np.zeros((np_in.shape[0], 7))
    for i in range(np_in.shape[0]):
        rt[i] = ans
    return rt
def segmentation(img):
	segments = slic(img, n_segments=100, compactness=10)
	return segments
    
if __name__ == '__main__':
    model = Classifier()
    _dict = torch.load('models/p0.68013_model.pth', map_location='cpu')
    # _dict = torch.load('/content/drive/My Drive/ML2019/hw4/models/p0.68013_model.pth', map_location='cpu')
    model.load_state_dict(_dict)
    model.eval()
    main()