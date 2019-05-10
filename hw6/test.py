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


def padding(features, maxlen=100, padding=0):
    padded_features = []
    for feature in features:
        if len(feature) >= maxlen:
            padded_feature = feature[:maxlen]
        else:
            padded_feature = feature
            while(len(padded_feature) < maxlen):
                padded_feature.append(padding)
        padded_features.append(padded_feature)
    return padded_features


def readfile_from_csv(train_file_path, test_file_path, word2vec_model_path):
    data_x = cut_raw(train_file_path, remove_space=True)
    test_x = cut_raw(test_file_path, remove_space=True)
    print('len data_x', len(data_x))
    print('len test_x', len(test_x))
    print('constructing vocab set...')
    vocab = list_to_set(data_x+test_x)
    vocab_size = len(vocab)
    print('vocab_size = ', vocab_size)

    vector_size = 250
    sequence_len = 100

    word_to_idx = {word: i+1 for i, word in enumerate(vocab)}
    idx_to_word = {i+1: word for i, word in enumerate(vocab)}

    test_x = encode(test_x, vocab, word_to_idx)
    test_x = padding(test_x, maxlen=sequence_len)

#     print('test0', test_x[0])
#     print('test1', test_x[1])
#     print('test2', test_x[2])
    weight = torch.zeros(vocab_size+1, vector_size)
    wv_model = pd.read_csv(word2vec_model_path).values[:, 1]
    test_x = torch.LongTensor(test_x)

    return weight, test_x, wv_model


class my_RNN_Net(nn.Module):
    def __init__(self, weight):
        super(my_RNN_Net, self).__init__()
        # self.embedding = nn.Embedding.from_pretrained(weight)
        # self.embedding.weight.requires_grad = False
        self.input_size = 250
        self.dropout = 0.95
        self.num_hiddens = 100
        self.num_layers = 2
        self.bidirectional = True

        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False

        self.encoder = nn.LSTM(self.input_size, hidden_size=self.num_hiddens,
                               num_layers=self.num_layers, bidirectional=self.bidirectional,
                               dropout=self.dropout)
        if self.bidirectional:
            self.decoder = nn.Sequential(
                nn.Linear(self.num_hiddens * 4, 2)
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(self.num_hiddens * 2, 2)
            )

    def my_init_weights(self, weight):
        print('init weight...')
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        states, hidden = self.encoder(embeddings.permute([1, 0, 2]))
        encoding = torch.cat([states[0], states[-1]], dim=1)
        outputs = self.decoder(encoding)
        return outputs


def main():

    # weight, test_x, new_result = readfile_from_csv(sys.argv[1],
    #                                                sys.argv[1],
    #                                                '/content/drive/My Drive/ML2019/hw6/models/HMMresult')
    weight, test_x = readfile_from_csv(sys.argv[1],
                                       sys.argv[1],
                                       'models/HMMresult')
    test_set = TensorDataset(test_x)

#     print(weight)
    num_epoch = 50
    batch_size = 1000

    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=8)
    # models = [
    #     '/content/drive/My Drive/ML2019/hw6/models/best_model_0252.pth',
    #     '/content/drive/My Drive/ML2019/hw6/models/best_model_0242.pth',
    #     '/content/drive/My Drive/ML2019/hw6/models/best_model_0307.pth']
    models = [
        'models/best_model_0252.pth',
        'models/best_model_0242.pth',
        'models/best_model_0307.pth']


    for mi in range(len(models)):
        # model = my_RNN_Net().cuda()
        # model.load_state_dict(
        #     '/content/drive/My Drive/ML2019/hw6/models/best_model.pth')
        model = torch.load(models[mi])
#         model.my_init_weights(weight)
#         model = model.cuda()
        print(model.embedding.weight)

        model.eval()
        print(model)
        # result_con = []
        for i, data in enumerate(test_loader):
            #########GPU#########
            test_pred = model(data[0].cuda())
            # result = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            result_raw = test_pred.cpu().data.numpy()
            # results.append(result)
            if (i == 0):
                result_con = result_raw
            else:
                result_con = np.append(result_con, result_raw, axis=0)

        if (mi == 0):
            result_all = result_con
        else:
            result_all = result_all + result_con

        results = np.argmax(result_all, axis=1)
        print(results.shape)

    output_file = sys.argv[3]
    # output_file = 'result.csv'
    result_csv = []
    result_csv.append(['id', 'label'])
    for i in range(len(new_result)):
        result_csv.append([i, int(new_result[i])])
    result_csv = np.array(result_csv)
    print(result_csv.shape)
    np.savetxt(output_file, result_csv, delimiter=",", fmt="%s")


if __name__ == '__main__':
    zero_vector = np.zeros(100)
    jieba.set_dictionary(sys.argv[2])
    main()
