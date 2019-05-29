import numpy as np
# from sklearn.manifold import TSNE
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader
# from skimage import io
import numpy as np
from PIL import Image
import sys
# from sklearn.cluster import KMeans
import pandas as pd
import gc
import sys
from PIL import Image, ImageDraw, ImageFont

# class MyDataset(Dataset):
#     def __init__(self, t_data, augmentation=1):
#         # self.train_data = np.genfromtxt(file_path, dtype=bytes, delimiter=' ')
#         # self.label = np.genfromtxt(file_path, dtype=bytes, delimiter=' ')
#         self.train_data = t_data
#         self.aug_size = augmentation

#         # self.label = pd.read_csv(label_file_path, delimiter=' ', header = -1)
#         # print(self.train_data.shape)
#         # print(self.label.shape)
#     def __len__(self):
#         return len(self.train_data) * self.aug_size

#     def __getitem__(self, idx):

#         # imread: a function that reads an image from path

#         i = int(idx/self.aug_size)
#         if (idx % self.aug_size != 0):
#             #             _img = Image.fromarray(self.train_data[i])
#             img = data_transformations(self.train_data[i])
#             # img = np.array(new_img)
# #             img = self.train_data[i]

#         else:
#             img = self.train_data[i]

#         # some operations/transformations
#         return img


def read_label(train_path):
    images = pd.read_csv(train_path).values
    # for line in images:
    #     file_name = line[0][10:]
    #     zero_count = 0
    #     ptr = 0
    #     while(file_name[ptr] == 0):

    return images


def corp_tumor_in_cancer_body(c_count, file_name, t_x1, t_y1, t_x2, t_y2, num_imgs=10):
    tumor_image = Image.open(file_name)
    tumor_image = np.array(tumor_image, dtype='uint8')

    new_images = []
    # x
    if(t_x2-t_x1-128 >= 0):
        tumor_x1_max = t_x2-t_x1-128
        tumor_x_length = 128
    else:
        tumor_x1_max = 0
        tumor_x_length = t_x2-t_x1
    # y
    if(t_y2-t_y1-128 >= 0):
        tumor_y1_max = t_y2-t_y1-128
        tumor_y_length = 128
    else:
        tumor_y1_max = 0
        tumor_y_length = t_y2-t_y1

    for i in range(num_imgs):
        new_image = np.zeros((128, 128), dtype='uint8')

        if tumor_x1_max == 0:
            tumor_x1 = t_x1
        else:
            tumor_x1 = np.random.randint(t_x1, t_x1+tumor_x1_max)

        if tumor_y1_max == 0:
            tumor_y1 = t_y1
        else:
            tumor_y1 = np.random.randint(t_y1, t_y1+tumor_y1_max)

        new_image[:tumor_x_length, :tumor_y_length] \
            = tumor_image[tumor_x1:tumor_x1+tumor_x_length, tumor_y1:tumor_y1+tumor_y_length]

        new_images.append(new_image)
        # img = Image.fromarray(new_image)
        # img.save('output/cancer/%06d.jpg' % (c_count))
        c_count+=1

    return c_count, np.array(new_images, dtype='uint8')


def corp_no_tumor_in_cancer_body(n_count, file_name, t_x1, t_y1, t_x2, t_y2, num_imgs=5):
    tumor_image = Image.open(file_name)
    tumor_image = np.array(tumor_image, dtype='uint8')

    new_images = []

    success_count = 0
    while(True):
        new_image = np.zeros((1, 128, 128), dtype='uint8')

        new_image_x1 = np.random.randint(0, 896)
        new_image_y1 = np.random.randint(0, 896)
        new_image_x2 = new_image_x1+128
        new_image_y2 = new_image_y1+128

        if(new_image_x1 >= t_x1 and new_image_x1 <= t_x2 and new_image_y1 >= t_y1 and new_image_y1 <= t_y2):
            continue
        if(new_image_x2-1 >= t_x1 and new_image_x2-1 <= t_x2 and new_image_y2-1 >= t_y1 and new_image_y2-1 <= t_y2):
            continue

        if(t_x1 >= new_image_x1 and t_x1 <= new_image_x2-1 and t_y1 >= new_image_y1 and t_y1 <= new_image_y2-1):
            continue
        if(t_x2 >= new_image_x1 and t_x2 <= new_image_x2-1 and t_y2 >= new_image_y1 and t_y2 <= new_image_y2-1):
            continue

        success_count += 1

        new_image = tumor_image[new_image_x1:new_image_x2,
                                new_image_y1:new_image_y2]
        new_images.append(new_image)
        # img = Image.fromarray(new_image)
        # img.save('output/no_cancer/%06d.jpg' %(n_count))
        n_count+=1
        if(success_count >= num_imgs):
            break
    return n_count, np.array(new_images, dtype='uint8')


def corp_no_tumor_in_no_cancer_body(n_count, file_name, num_imgs=5):
    tumor_image = Image.open(file_name)
    tumor_image = np.array(tumor_image, dtype='uint8')

    new_images = []

    for i in range(num_imgs):
        new_image = np.zeros((1, 128, 128), dtype='uint8')

        new_image_x1 = np.random.randint(0, 896)
        new_image_y1 = np.random.randint(0, 896)
        new_image_x2 = new_image_x1+128
        new_image_y2 = new_image_y1+128

        new_image = tumor_image[new_image_x1:new_image_x2,
                                new_image_y1:new_image_y2]
        new_images.append(new_image)
        # img = Image.fromarray(new_image)
        # img.save('output/no_cancer/%06d.jpg' %(n_count))
        n_count+=1

    return n_count, np.array(new_images, dtype='uint8')


if __name__ == '__main__':
    images = read_label('data/train.csv')

    result_csv = []
    t = 0
    nt = 0
    n = len(images)

    labels = []
    train_data_flag = 0
    c_count = 0
    n_count = 0
    for i, img in enumerate(images):
        gc.collect()
        msg = "solving [%06d/%06d]" % (i+1, n)
        print(msg, end='', flush=True)
        back = '\b'*len(msg)
        print(back, end='', flush=True)

        file_name = img[0]

        if img[5] == "tumor":
            # print('tumor')
            t_x1 = int(img[1])
            t_y1 = int(img[2])
            t_x2 = int(img[3])
            t_y2 = int(img[4])

            # # draw rect
            # tumor_image = Image.open('data/'+file_name)
            # d = ImageDraw.Draw(tumor_image)
            # d.rectangle(xy = [t_x1, t_y1, t_x2, t_y2], fill=None, outline=(255))
            # tumor_image.save('output/'+'rec'+file_name[15:])
            
            # corp tumor part

            # new_c_count, new_tumor_images = corp_tumor_in_cancer_body(
            #     c_count, 'data/'+file_name, t_x1, t_y1, t_x2, t_y2, 10)
            # c_count = new_c_count
            for i in range(10):
                labels.append([int(1)])
            
            # new_n_count, new_no_tumor_images = corp_no_tumor_in_cancer_body(
            #     n_count, 'data/'+file_name, t_x1, t_y1, t_x2, t_y2)
            # n_count = new_n_count
            for i in range(5):
                labels.append([int(0)])

            # if(train_data_flag):
            #     train_data = np.append(train_data, new_tumor_images, axis=0)
            #     train_data = np.append(train_data, new_no_tumor_images, axis=0)
            #     # labels.append(int(1))
            #     # labels.append(int(1))
            # else:
            #     train_data = new_tumor_images
            #     train_data = np.append(train_data, new_no_tumor_images, axis=0)
            #     train_data_flag = 1
                # labels.append(int(1))
                # labels.append(int(1))
        else:
            # print("no tumor")
            for i in range(5):
                labels.append([int(0)])
            # new_n_count, new_no_tumor_images = corp_no_tumor_in_no_cancer_body(
            #     n_count, 'data/'+file_name)
            # n_count = new_n_count

            # if(train_data_flag):
            #     train_data = np.append(train_data, new_no_tumor_images, axis=0)
            #     labels.append(int(0))
            # else:
            #     train_data = new_no_tumor_images
            #     train_data_flag = 1
            #     labels.append(int(0))
        gc.collect()
    # print('')
    # print(train_data.shape)
    # np.save('train_data', train_data)
    np.save('train_label', np.array(labels))
    # result_csv = np.array(result_csv)
    # print(result_csv.shape)
    # np.savetxt(sys.argv[3], result_csv, delimiter=",", fmt="%s")
    # print('\ntumor count', t)
    # print('no umor count', nt)
