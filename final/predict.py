import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse
import pandas as pd
import sys
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer, TensorTranformer, ValResizer

print(torch.__version__.split('.')[1])
assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
	parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

	parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
	parser.add_argument('--coco_path', help='Path to COCO directory')
	parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
	parser.add_argument('--csv_test', help='Path to file containing validation annotations (optional, see readme)')

	parser.add_argument('--model', help='Path to model (.pt) file.')

	parser = parser.parse_args(args)
	'''
	if parser.dataset == 'coco':
		dataset_val = CocoDataset(parser.coco_path, set_name='val2017', transform=transforms.Compose([Normalizer(), Resizer()]))
	'''
	if parser.dataset == 'csv':
		dataset_test = CSVDataset(train_file=parser.csv_test, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), ValResizer()]), predict=True)
	else:
		raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

	#sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
	dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0, collate_fn=collater)

	retinanet = torch.load(parser.model)

	use_gpu = True

	if use_gpu:
		retinanet = retinanet.cuda()

	retinanet.eval()

	unnormalize = UnNormalizer()

	def draw_caption(image, box, caption):

		b = np.array(box).astype(int)
		cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
		cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


	image_list = []
	x1_list = []
	width = []
	y1_list = []
	height = []
	label_list = []
	for idx, data in enumerate(dataloader_test):

		with torch.no_grad():
			st = time.time()
			scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
			#print(data['name'][0])
			if (idx+1)%100 == 0:
				print(idx+1)
			#print('Elapsed time: {}'.format(time.time()-st))
			idxs = np.where(scores>0.5)
			img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

			img[img<0] = 0
			img[img>255] = 255

			img = np.transpose(img, (1, 2, 0))

			img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

			for j in range(idxs[0].shape[0]):
				bbox = transformed_anchors[idxs[0][j], :]
				image_list += [data['name'][0]][36:]
				x1 = int(bbox[0])*2
				y1 = int(bbox[1])*2
				x2 = int(bbox[2])*2
				y2 = int(bbox[3])*2
				x1_list += [str(x1)]
				y1_list += [str(y1)]
				width += [str(x2-x1)]
				height += [str(y2-y1)]
				label_list += [1]
				label_name = dataset_test.labels[int(classification[idxs[0][j]])]
			if idxs[0].shape[0] == 0:
				image_list += [data['name'][0]][36:]
				x1_list += ['']
				y1_list += ['']
				width += ['']
				height += ['']
				label_list += [0]
		if (idx+1)%50 == 0:
			print(len(image_list), len(x1_list), len(y1_list), len(width), len(height), len(label_list))
	data = np.array([image_list])
	data = np.append(data, [x1_list], axis=0)
	data = np.append(data, [y1_list], axis=0)
	data = np.append(data, [width], axis=0)
	data = np.append(data, [height], axis=0)
	data = np.append(data, [label_list], axis=0)
	dataframe = pd.DataFrame(data = data.T)
	dataframe.to_csv("prediction.csv",index=False,sep=',')



if __name__ == '__main__':
 main()
