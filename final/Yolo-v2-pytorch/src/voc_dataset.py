"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from src.data_augmentation import *
import pandas as pd
from random import uniform


class VOCDataset(Dataset):
    def __init__(self, train_image_path, train_label_path, data_augmentation=False, image_size=512):

        self.train_image_path = train_image_path
        train_labels = pd.read_csv(train_label_path).values

        self.data_augmentation = data_augmentation
        self.classes = ['tumor']

        self.image_size = image_size
        self.num_classes = len(self.classes)

        self.new_train_labels = []
        self.image_name = []

        num_train_labels = len(train_labels)
        i = 0
        while i < num_train_labels:
            objects = []
            l = int(train_labels[i, 5])
            if l == 1:
                img_name = train_labels[i, 0]
                self.image_name.append(img_name)
                while i < num_train_labels and train_labels[i, 0] == img_name:

                    x1 = int(train_labels[i, 1]/2)
                    y1 = int(train_labels[i, 2]/2)
                    w1 = int(train_labels[i, 3]/2)
                    h1 = int(train_labels[i, 4]/2)
                    l = int(train_labels[i, 5])

                    bbox_1 = [x1, y1, x1+w1, y1+h1, 0]
                    objects.append(bbox_1)
                    
                    if data_augmentation:
                        for ob in range(5):
                            x_0 = uniform(0,w1/2)
                            y_0 = uniform(0,h1/2)
                            bbox_2 = [x1+x_0, y1+y_0, x1+w1/2+x_0, y1+h1/2+y_0, 0]

                            objects.append(bbox_2)


                    i += 1
                self.new_train_labels.append(objects)

                # self.image_name.append(img_name)
            else:
                i += 1
            # self.new_train_labels.append(objects)

        self.num_images = len(self.new_train_labels)
        # print('num_images', self.num_images)
        # print('num_images2', len(self.image_name))

    def __len__(self):
        return self.num_images

    def __getitem__(self, id):

        image_path = os.path.join(self.train_image_path, self.image_name[id])
        image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        objects = self.new_train_labels[id]
        
        if self.data_augmentation:
            transformations = Compose([VerticalFlip(), Crop(), Resize(512)])
            image, objects = transformations((image, objects))
        else:
            transformations = Compose([Resize(512)])
            image, objects = transformations((image, objects))
        image = np.transpose(np.array(image, dtype=np.float32), (2, 0, 1))
        # image = image[0]
        # image = np.expand_dims(image, axis=0)

        return image, np.array(objects, dtype=np.float32)
