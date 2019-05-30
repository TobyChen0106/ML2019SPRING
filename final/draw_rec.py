import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from PIL import Image
import sys
import pandas as pd
import gc
import sys
from PIL import Image, ImageDraw, ImageFont


if __name__ == '__main__':
    # images = pd.read_csv('data/train_labels.csv').values[:100]
    images = pd.read_csv('train_output.csv').values
    n = len(images)
    
    i=0
    while(i<n):
        gc.collect()
        msg = "solving [%06d/%06d]" % (i+1, n)
        print(msg, flush=True)
        back = '\b'*len(msg)

        img = images[i]
        file_name = 'train%05d.png'%int(img[0])

        if int(img[5]) == 1:
            # print('tumor')
            t_x1 = int(img[1])
            t_y1 = int(img[2])
            t_w = int(img[3])
            t_h = int(img[4])
            t_x2 = t_x1+t_w
            t_y2 = t_y1+t_h

            # draw rect
            tumor_image = Image.open('data/train/'+file_name)
            print(t_x1, t_y1, t_x2, t_y2)
            d = ImageDraw.Draw(tumor_image)
            d.rectangle(xy=[t_x1, t_y1, t_x2, t_y2], fill=None, outline=(255))
            #d.rectangle(xy=[t_x1+10, t_y1+10, t_x2+10, t_y2+10], fill=None, outline=(255))
            

            if(i+1 < n ):
                next_img = images[i+1]
                next_file_name = 'train%05d.png'%int(next_img[0])
                
                if next_file_name == file_name and int(next_img[5]) == 1:
                    print('findnext', next_img[0])
                    t_x1 = int(next_img[1])
                    t_y1 = int(next_img[2])
                    t_w = int(next_img[3])
                    t_h = int(next_img[4])
                    t_x2 = t_x1+t_w
                    t_y2 = t_y1+t_h
                    print(t_x1, t_y1, t_x2, t_y2)
                    
                    d.rectangle(xy=[t_x1, t_y1, t_x2, t_y2], fill=None, outline=(255))
                    i+=1
  
            tumor_image.save('train_rec_output/'+'rec_'+file_name)
        i+=1          
        
        gc.collect()
        #print(back, end='', flush=True)
