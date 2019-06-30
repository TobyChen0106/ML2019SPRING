import time
import os
import copy
import argparse
import pdb
import collections
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision
from torchsummary import summary
import model
from anchors import Anchors
import losses
from dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer, ValResizer
from torch.utils.data import Dataset, DataLoader

import csv_eval

assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):

	parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
	#add a bunch of arguments(customized by Yu Han Huang)
	parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
	parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
	parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
	parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
	parser.add_argument('--model', default='None')
	parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
	parser.add_argument('--resnext', help='change backbone to resnext101', action='store_true')
	parser.add_argument('--epochs', help='Number of Epochs', type=int, default=12)
	parser.add_argument('--batch_size', help='Batch Size', type=int, default=4)
	parser.add_argument('--workers', help='Number of Workers', type=int, default=4)
	parser.add_argument('--lr', help='Learning Rate for training', type=float, default=1e-5)
	parser.add_argument('--dropout1', help='Dropout Rate for layer dropout1 in ClassficationModel', type=float, default=0.25)
	parser.add_argument('--dropout2', help='Dropout Rate for layer dropout2 in ClassficationModel', type=float, default=0.25)
	parser.add_argument('--angle', help='Angle of pictures while implementing Data Augmentation', type=float, default=6)
	parser.add_argument('--size', help='The length of the side of pictures', type=int, default=512)
	parser.add_argument('--zoom_range', help='Zoom Range of pictures while implementing Data Augmentation. Please type two arguments for this one.',
						 nargs='+', type=float, default=[-0.1,0.1])
	parser.add_argument('--alpha', help='Alpha for focal loss', type=float, default=0.25)
	parser.add_argument('--gamma', help='Gamma for focal loss', type=float, default=2)
	parser.add_argument('--loss_with_no_bboxes', action='store_true')
	parser.add_argument('--no_bboxes_alpha', help='Alpha for focal loss', type=float, default=0.5)
	parser.add_argument('--no_bboxes_gamma', help='Gamma for focal loss', type=float, default=2)
	
	parser = parser.parse_args(args)

	# Create the data loaders
	if parser.dataset == 'csv':

		if parser.csv_train is None:
			raise ValueError('Must provide --csv_train when training on CSV,')

		if parser.csv_classes is None:
			raise ValueError('Must provide --csv_classes when training on CSV,')


		dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
		 transform=transforms.Compose([Normalizer(), Augmenter(angle=parser.angle), Resizer(zoom_range=parser.zoom_range, side=parser.side)]))

		if parser.csv_val is None:
			dataset_val = None
			print('No validation annotations provided.')
		else:
			dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), ValResizer()]))

	else:
		raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

	sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=False)
	dataloader_train = DataLoader(dataset_train, num_workers=parser.workers, collate_fn=collater, batch_sampler=sampler)

	if dataset_val is not None:
		sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
		dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

	# Create the model
	# add arguments dropout1, dropout2, alpha, gamma, loss_with_no_bboxes, no_bboxes_alpha, no_bboxes_gamma(customized by Yu Han Huang)
	if parser.resnext == False:
		if parser.depth == 18:
			retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True,
										 dropout1=parser.dropout1, dropout2=parser.dropout2,
										 alpha=parser.alpha, gamma=parser.gamma, loss_with_no_bboxes=parser.loss_with_no_bboxes,
										 no_bboxes_alpha=parser.no_bboxes_alpha, no_bboxes_gamma=parser.no_bboxes_gamma)
		elif parser.depth == 34:
			retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True,
										 dropout1=parser.dropout1, dropout2=parser.dropout2,
										 alpha=parser.alpha, gamma=parser.gamma, loss_with_no_bboxes=parser.loss_with_no_bboxes,
										 no_bboxes_alpha=parser.no_bboxes_alpha, no_bboxes_gamma=parser.no_bboxes_gamma)
		elif parser.depth == 50:
			retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True,
										 dropout1=parser.dropout1, dropout2=parser.dropout2,
										 alpha=parser.alpha, gamma=parser.gamma, loss_with_no_bboxes=parser.loss_with_no_bboxes,
										 no_bboxes_alpha=parser.no_bboxes_alpha, no_bboxes_gamma=parser.no_bboxes_gamma)
		elif parser.depth == 101:
			retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True,
										 dropout1=parser.dropout1, dropout2=parser.dropout2,
										 alpha=parser.alpha, gamma=parser.gamma, loss_with_no_bboxes=parser.loss_with_no_bboxes,
										 no_bboxes_alpha=parser.no_bboxes_alpha, no_bboxes_gamma=parser.no_bboxes_gamma)
		elif parser.depth == 152:
			retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True,
										 dropout1=parser.dropout1, dropout2=parser.dropout2,
										 alpha=parser.alpha, gamma=parser.gamma, loss_with_no_bboxes=parser.loss_with_no_bboxes,
										 no_bboxes_alpha=parser.no_bboxes_alpha, no_bboxes_gamma=parser.no_bboxes_gamma)
		else:
			raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
	else:
		if parser.depth == 101:
			retinanet = model.resnext101(num_classes=dataset_train.num_classes(), pretrained=True,
										 dropout1=parser.dropout1, dropout2=parser.dropout2,
										 alpha=parser.alpha, gamma=parser.gamma, loss_with_no_bboxes=parser.loss_with_no_bboxes,
										 no_bboxes_alpha=parser.no_bboxes_alpha, no_bboxes_gamma=parser.no_bboxes_gamma)	

	use_gpu = True

	if parser.model != 'None':
		retinanet = torch.load(parser.model)

	if use_gpu:
		retinanet = retinanet.cuda()

	retinanet = torch.nn.DataParallel(retinanet).cuda()

	retinanet.training = True

	optimizer = optim.Adam(retinanet.parameters(), lr=parser.lr)

	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

	loss_hist = collections.deque(maxlen=500)

	retinanet.train()
	retinanet.module.freeze_bn()


	print('Num training images: {}'.format(len(dataset_train)))

	for epoch_num in range(parser.epochs):

		retinanet.train()
		retinanet.module.freeze_bn()
		print_activate = 0
		epoch_loss = []
		for iter_num, data in enumerate(dataloader_train):
			try:
				optimizer.zero_grad()
				classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])

				classification_loss = classification_loss.mean()
				regression_loss = regression_loss.mean()

				loss = classification_loss + regression_loss
				#print(classification_loss, regression_loss)
				if bool(loss == 0):
					continue

				loss.backward()
				print_activate += 1
				torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

				optimizer.step()

				loss_hist.append(float(loss))

				epoch_loss.append(float(loss))
				if print_activate%15 == 0:
					print('Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))
				
				del loss
				del classification_loss
				del regression_loss
			except Exception as e:
				print(e)
				continue
		
		scheduler.step(np.mean(epoch_loss))	

		torch.save(retinanet.module, '{}_retinanet_resnext_v4_{}.pt'.format(parser.dataset, epoch_num))

		if parser.dataset == 'csv' and parser.csv_val is not None:

			print('Evaluating dataset')

			mAP = csv_eval.evaluate(dataset_val, retinanet)

	retinanet.eval()

	torch.save(retinanet, 'model_final.pt'.format(epoch_num))

if __name__ == '__main__':
 main()
