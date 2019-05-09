from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader
import sys
import time
import numpy as np
import pandas as pd
from PIL import Image
import io
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import jieba
import gc
import pandas as pd

def cut_raw(train_file, remove_space=False):
    print('[%s] reading csv...' % train_file)

    train_x = pd.read_csv(train_file).values[0:119018, 1]
    output = []
    for in_seq in train_x:
        out_seq = list(jieba.cut(in_seq, cut_all=False, HMM=False))
        if remove_space:
            while ' ' in out_seq:
                out_seq.remove(' ')
            output.append(out_seq)
    return output


def list_to_set(ls):
    s = set()
    for i in range(len(ls)):
        for x in ls[i]:
            s.add(x)
    return s


def encode(tokenized_samples, vocab, word_to_idx):
    features = []
    for sample in tokenized_samples:
        feature = []
        for token in sample:
            if token in word_to_idx:
                feature.append(word_to_idx[token])
            else:
                feature.append(0)
        features.append(feature)
    return features


class MyDataset(Dataset):
    def __init__(self, x_data, word_to_idx):
        self.x_data = x_data


        self.word_to_idx = word_to_idx
        self.vocab_size = len(word_to_idx)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        bow = torch.zeros([self.vocab_size], dtype=torch.float32)

        for word in self.x_data[idx]:
            bow[self.word_to_idx[word]-1] += 1

        return bow


def readfile_from_csv(train_file_path, test_file_path):
    data_x = cut_raw(train_file_path, remove_space=True)
    test_x = cut_raw(test_file_path, remove_space=True)

    print('constructing vocab set...')
    vocab = list_to_set(data_x+test_x)
    vocab_size = len(vocab)
    print('vocab_size = ', vocab_size)


    print('word_to_idx')
    word_to_idx = {word: i+1 for i, word in enumerate(vocab)}
    gc.collect()

    print('inupt data process done')
    return vocab_size, word_to_idx, test_x


class my_BOW_DNN_Net(nn.Module):
    def __init__(self, input_dim):
        super(my_BOW_DNN_Net, self).__init__()

        self.dense = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            # nn.Linear(1024, 1024),
            # nn.LeakyReLU(0.2),
            # nn.Dropout(p=0.5),
            nn.Linear(1024, 2),
        )

    def forward(self, inputs):
        return self.dense(inputs)

def main():

    vocab_size, word_to_idx, test_x = readfile_from_csv('/content/drive/My Drive/ML2019/hw6/hw6_data/train_x.csv',
                                       '/content/drive/My Drive/ML2019/hw6/hw6_data/test_x.csv')
    # weight, test_x = readfile_from_csv('hw6_data/train_x.csv',
    #                                    'hw6_data/test_x.csv',
    #                                    'word2vec_noHMM.wv')

    test_set = MyDataset(test_x,word_to_idx)

    batch_size = 1000

    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=8)
    models = [
        # '/content/drive/My Drive/ML2019/hw6/models/best_model_0436.pth',
        # '/content/drive/My Drive/ML2019/hw6/models/best_model_0436.pth',
        # '/content/drive/My Drive/ML2019/hw6/models/best_model_0436.pth',
        # '/content/drive/My Drive/ML2019/hw6/models/best_model_0436.pth',
        # '/content/drive/My Drive/ML2019/hw6/models/best_model_0436.pth',              
        # 'models/best_model_0307.pth'
        '/content/drive/My Drive/ML2019/hw6/models/my_BOW_DNN_Net_best_model.pth'
        ]

    # result_all = []
    for mi in range(len(models)):
        # model = my_RNN_Net().cuda()
        # model.load_state_dict(
        #     '/content/drive/My Drive/ML2019/hw6/models/best_model.pth')
        model = torch.load(models[mi])
        model.eval()

        # result_con = []
        for i, data in enumerate(test_loader):
            #########GPU#########
            # print('data', data.shape)
            test_pred = model(data.cuda())
            # result = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            result_raw = test_pred.cpu().data.numpy()
            # results.append(result)
            # print(result_raw)
            if (i == 0):
                    result_con = result_raw
            else:
                result_con = np.append(result_con, result_raw, axis = 0)
        print(result_con)
        
        if (mi == 0):
            result_all = result_con
        else:
            result_all = result_all + result_con

        print(result_all.shape)
        
        results = np.argmax(result_all, axis=1)

    # output_file = '/content/drive/My Drive/ML2019/hw6/result.csv'
    output_file = 'result.csv'
    result_csv = []
    result_csv.append(['id', 'label'])
    for i in range(len(results)):
        result_csv.append([i, int(results[i])])
    result_csv = np.array(result_csv)
    print(result_csv.shape)
    np.savetxt(output_file, result_csv, delimiter=",", fmt="%s")


if __name__ == '__main__':
    zero_vector = np.zeros(100)
    jieba.set_dictionary('/content/drive/My Drive/ML2019/hw6/dict.txt.big')
    main()
