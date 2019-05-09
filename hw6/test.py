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

#     train_x = np.genfromtxt(train_file, delimiter=',',
#                             skip_header=1, usecols=1, dtype='U20')
    train_x = pd.read_csv(train_file).values[0:119018, 1]
    output = []
    for in_seq in train_x:
        out_seq = list(jieba.cut(in_seq, cut_all=False, HMM=False))
        if remove_space:
            while ' ' in out_seq:
                out_seq.remove(' ')
            output.append(out_seq)
    return output


def word2vector(input_in_seqs, word2vec_model_path):
    global zero_vector

    model = KeyedVectors.load(word2vec_model_path, mmap='r')
    model.syn0[model.vocab[' '].index] = zero_vector
    print('padding')
    output_vectors = []
    count = 1
    for in_seq in input_in_seqs:
        out_vector = []
        in_seq_size = len(in_seq)
        for word_count in range(100):
            if(word_count < in_seq_size):

                in_vec = model[in_seq[word_count]]
#                 except:
#                   in_vec = zero_vector
                out_vector.append(in_vec)
            else:
                in_vec = model.get_vector(' ')
#                 print(in_vec)
                out_vector.append(in_vec)
        if count % 10000 == 0:
            gc.collect()
        count += 1
        assert(len(out_vector) == 100)
        output_vectors.append(out_vector)
    print('padding done')
    return output_vectors


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

    weight = torch.zeros(vocab_size+1, vector_size)

    test_x = torch.LongTensor(test_x)

    return weight, test_x


class my_RNN_Net(nn.Module):
    def __init__(self, weight):
        super(my_RNN_Net, self).__init__()
        # self.embedding = nn.Embedding.from_pretrained(weight)
        # self.embedding.weight.requires_grad = False
        self.input_size = 250
        self.dropout = 0.95
        self.num_hiddens = 200
        self.num_layers = 2
        self.bidirectional = True
#         self.pre_hidden
        
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False

        self.encoder = nn.LSTM(self.input_size, hidden_size=self.num_hiddens,
                               num_layers=self.num_layers, bidirectional=self.bidirectional,
                               dropout=self.dropout)
        if self.bidirectional:
            self.decoder = nn.Sequential(
#                 nn.Linear(self.num_hiddens * 4, self.num_hiddens * 4),
#                 nn.LeakyReLU(0.2),
#                 nn.Dropout(0.5),
#                 nn.Linear(self.num_hiddens * 4, self.num_hiddens * 4),
#                 nn.LeakyReLU(0.2),
#                 nn.Dropout(0.5),
                nn.Linear(self.num_hiddens * 4, 2)
            )
        else:
            self.decoder = nn.Sequential(
#                 nn.Linear(self.num_hiddens * 2, self.num_hiddens * 2),
#                 nn.LeakyReLU(0.2),
#                 nn.Dropout(0.5),
#                 nn.Linear(self.num_hiddens * 2, self.num_hiddens * 2),
#                 nn.LeakyReLU(0.2),
#                 nn.Dropout(0.5),
                nn.Linear(self.num_hiddens * 2, 2)
            )
    def init_weights(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
                        
    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        states, hidden = self.encoder(embeddings.permute([1, 0, 2]))
#         self.pre_hidden = hidden
        encoding = torch.cat([states[0], states[-1]], dim=1)
        outputs = self.decoder(encoding)
        return outputs

def main():

    # weight, test_x = readfile_from_csv('/content/drive/My Drive/ML2019/hw6/hw6_data/train_x.csv',
    #                                    '/content/drive/My Drive/ML2019/hw6/hw6_data/test_x.csv',
    #                                    '/content/drive/My Drive/ML2019/hw6/word2vec_noHMM.wv')
    weight, test_x = readfile_from_csv('hw6_data/train_x.csv',
                                       'hw6_data/test_x.csv',
                                       'word2vec_noHMM.wv')

    test_set = TensorDataset(test_x)

    num_epoch = 50
    batch_size = 100

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
        'models/best_model_0307.pth'
        # '/content/drive/My Drive/ML2019/hw6/models/best_model_0307.pth'
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
            test_pred = model(data[0].cuda())
            # result = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            result_raw = test_pred.cpu().data.numpy()
            # results.append(result)
            if (i == 0):
                    result_con = result_raw
            else:
                result_con = np.append(result_con, result_raw, axis = 0)
        
        if (mi == 0):
            result_all = result_con
        else:
            result_all = result_all + result_con
        
        results = np.argmax(result_all, axis=1)
        print(results.shape)

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
