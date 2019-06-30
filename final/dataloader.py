from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
from math import floor
import csv
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

#import pycocotools

import skimage.io
import skimage.transform
import skimage.color
import skimage

from PIL import Image

class CSVDataset(Dataset):
    """CSV dataset."""

    def __init__(self, train_file, class_list, transform=None, predict=False):
        """
        Args:
            train_file (string): CSV file with training annotations
            annotations (string): CSV file with class list
            test_file (string, optional): CSV file with testing annotations
        """
        self.train_file = train_file
        self.class_list = class_list
        self.transform = transform
        #predicter dataloader or not
        self.predict = predict
        # parse the provided class file
        try:
            with self._open_for_csv(self.class_list) as file:
                self.classes = self.load_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise(ValueError('invalid CSV class file: {}: {}'.format(self.class_list, e)), None)

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key
        # csv with img_path, x1, y1, x2, y2, class_name
        try:
            with self._open_for_csv(self.train_file) as file:
                self.image_data = self._read_annotations(csv.reader(file, delimiter=','), self.classes)
        except ValueError as e:
            raise(ValueError('invalid CSV annotations file: {}: {}'.format(self.train_file, e)), None)
        self.image_names = list(self.image_data.keys())

    def _parse(self, value, function, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        try:
            return function(value)
        except ValueError as e:
            raise(ValueError(fmt.format(e)), None)

    def _open_for_csv(self, path):
        """
        Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')


    def load_classes(self, csv_reader):
        result = {}

        for line, row in enumerate(csv_reader):
            line += 1

            try:
                class_name, class_id = row
            except ValueError:
                raise(ValueError('line {}: format should be \'class_name,class_id\''.format(line)), None)
            class_id = self._parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

            if class_name in result:
                raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
            result[class_name] = class_id
        return result


    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        
        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        name = self.image_names[idx]
        sample = {'img': img, 'annot': annot, 'name': name, 'scale': 1}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        img = skimage.io.imread(self.image_names[image_index])

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32)/255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotation_list = self.image_data[self.image_names[image_index]]
        annotations     = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotation_list) == 0:
            return annotations

        # parse annotations
        if self.predict == True:
            return annotations
        for idx, a in enumerate(annotation_list):
            # some annotations have basically no width / height, skip them
            x1 = a['x1']
            x2 = a['x2']
            y1 = a['y1']
            y2 = a['y2']

            if (x2-x1) < 1 or (y2-y1) < 1:
                continue

            annotation        = np.zeros((1, 5))
            
            annotation[0, 0] = x1
            annotation[0, 1] = y1
            annotation[0, 2] = x2
            annotation[0, 3] = y2

            annotation[0, 4]  = self.name_to_label(a['class'])
            annotations       = np.append(annotations, annotation, axis=0)

        return annotations

    def _read_annotations(self, csv_reader, classes):
        result = OrderedDict()
        for line, row in enumerate(csv_reader):
            line += 1
            if self.predict == False:
                try:
                    img_file, x1, y1, x2, y2, class_name = row[:6]
                except ValueError:
                    raise(ValueError('line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line)), None)
                if img_file not in result:
                    result[img_file] = []
                # If a row contains only an image path, it's an image without annotations.
                if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
                    continue

                x1 = self._parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
                y1 = self._parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
                x2 = self._parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
                y2 = self._parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))

                # Check that the bounding box is valid.
                if x2 <= x1:
                    raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
                if y2 <= y1:
                    raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

                # check if the current class name is correctly present
                if class_name not in classes:
                    raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

                result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})

            else:
                try:
                    img_file, sex, ap, age = row[:4]   
                except ValueError:
                    raise(ValueError('line {}: format should be \'img_file,sex,ap,age\''.format(line)), None)
                result[img_file] = []                 
                result[img_file].append({'sex': sex, 'ap': ap, 'age': age})

        return result

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def num_classes(self):
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)



def collater(data):

    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]
    names = [s['name'] for s in data]
        
    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)
    
    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                #print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1


    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'annot': annot_padded, 'name': names, 'scale': scales}

#add random zoom in and out(customized by Yu Han Huang)
class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, zoom_range=(-0.1,0.1), side=512):
        '''
        Args: zoom_range: The range for zoom in and out. (-0.1,0.1) means 90%~110%
              side: The length of the side of pictures
        '''
        self.zoom_range=zoom_range
        self.side = side

    def __call__(self, sample):
        zoom_range = self.zoom_range
        side = self.side
        image, annots, name = sample['img'], sample['annot'], sample['name']
        #resize the image to 512, and then resize the image by the range of scale (-0.1,0.1)
        random_scale = np.random.rand()*(zoom_range[1]-zoom_range[0])+1+zoom_range[0]
        rows, cols, cns = image.shape
        if rows < cols:
            scale = side/cols
        else:
            scale = side/rows
        new_rows = int(floor(side*random_scale))
        new_cols = int(floor(side*random_scale))
        # rescale the image so the smallest side is min_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (new_rows, new_cols))
        if new_rows >= new_cols:
            new_image = np.zeros((int(floor(side*new_rows/new_cols)), side, cns)).astype(np.float32)
        elif new_rows < new_cols:
            new_image = np.zeros((side, int(floor(side*new_cols/new_rows)), cns)).astype(np.float32)
        #original_annots = np.copy(annots)
        if random_scale <= 1:
            x = np.random.randint(low=0, high=new_image.shape[0]-new_rows+1)
            y = np.random.randint(low=0, high=new_image.shape[1]-new_cols+1)
            new_image[x:new_rows+x, y:new_cols+y, :] = image.astype(np.float32)
            annots[:, :4] *=random_scale*scale
            annots[:,0] += x
            annots[:,2] += x
            annots[:,1] += y
            annots[:,3] += y
        elif random_scale > 1:
            x = np.random.randint(low=0, high=-new_image.shape[0]+new_rows+1)
            y = np.random.randint(low=0, high=-new_image.shape[1]+new_cols+1)
            new_image[:, :, :] = image[x:x+new_image.shape[0], y:y+new_image.shape[1], :].astype(np.float32)
            annots[:, :4] *=random_scale*scale
            annots[:,0] -= x
            annots[:,2] -= x
            annots[:,1] -= y
            annots[:,3] -= y
            for i in range(len(annots)):
                for j in range(4):

                    if annots[i][j] < 0:
                        annots[i][j] = 0

                    elif j%2 == 0 and annots[i][j] > new_image.shape[0] - 1:
                        annots[i][j] = new_image.shape[0] - 1

                    elif j%2 == 1 and annots[i][j] > new_image.shape[1] - 1:
                        annots[i][j] = new_image.shape[1] - 1

        #print(new_image.shape)
        #print("original_annots :", original_annots, "\n", "new_annots :", annots, "\n", "x, y :", x, ",", y, "\n", "random_scale, scale :", random_scale, scale) 
        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots),'name': name, 'scale': scale}

class TensorTranformer(object):
    def __call__(self, sample):
        img, annots, name = sample['img'], sample['annot'], sample['name']
        #print(img.shape)
        return {'img': torch.from_numpy(img), 'annot': torch.from_numpy(annots),'name': name, 'scale': 1}



class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, angle=6):
        '''
        Args: angle: rotation angle
        '''
        self.angle = angle

    def __call__(self, sample, flip_x=0.5):
        #horizontally flip the image
        image, annots, name, scale = sample['img'], sample['annot'], sample['name'], sample['scale']
        if np.random.rand() < flip_x:
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()
            
            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots, 'name': name, 'scale': scale}
        
        #rotate the image(customized by Yu Han Huang)
        angle = np.random.rand()*2*self.angle - self.angle
        skimage.transform.rotate(image, angle)
        
        x1 = annots[:, 0].copy()
        x2 = annots[:, 2].copy()
        y1 = annots[:, 1].copy()
        y2 = annots[:, 3].copy()
        rows, cols, channels = image.shape
        x1 = x1 - rows/2 + 0.5
        x2 = x2 - rows/2 + 0.5
        y1 = y1 - cols/2 + 0.5
        y2 = y2 - cols/2 + 0.5
        new_x1 = x1*np.cos(np.radians(angle)) - y1*np.sin(np.radians(angle))
        new_x2 = x2*np.cos(np.radians(angle)) - y2*np.sin(np.radians(angle))
        new_y1 = y1*np.cos(np.radians(angle)) + x1*np.sin(np.radians(angle))
        new_y2 = y2*np.cos(np.radians(angle)) + x2*np.sin(np.radians(angle))
        new_x1 = new_x1 + rows/2 + 0.5
        new_x2 = new_x2 + rows/2 + 0.5
        new_y1 = new_y1 + cols/2 + 0.5
        new_y2 = new_y2 + cols/2 + 0.5
        annots[:, 0] = np.clip(new_x1, 0, rows)
        annots[:, 1] = np.clip(new_y1, 0, cols)
        annots[:, 2] = np.clip(new_x2, 0, rows)
        annots[:, 3] = np.clip(new_y2, 0, cols)

        return sample


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):

        image, annots, name, scale = sample['img'], sample['annot'], sample['name'], sample['scale']

        return {'img':((image.astype(np.float32)-self.mean)/self.std), 'annot': annots, "name": name, "scale": scale}

class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]

#original Resizer(used for validation and prediction)
class ValResizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, min_side=512, max_side=1024):
        image, annots, name = sample['img'], sample['annot'], sample['name']

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - (rows-1)%32 - 1
        pad_h = 32 - (cols-1)%32 - 1

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'name': name, 'scale': scale}

