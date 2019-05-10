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


def readfile_from_csv(train_file_path, test_file_path, label__file_path, word2vec_model_path, val_num=0.2):
    data_x = cut_raw(train_file_path, remove_space=True)
    test_x = cut_raw(test_file_path, remove_space=True)

    print('constructing vocab set...')
    vocab = list_to_set(data_x+test_x)
    vocab_size = len(vocab)
    print('vocab_size = ', vocab_size)

    vector_size = 250
    sequence_len = 100
    
    print('building word dependancies...')
    wv_model = Word2Vec(data_x+test_x, size=vector_size, iter = 10,
                        window=5, min_count=5, workers=4)

    wv_model = 
    data_y = pd.read_csv(label__file_path).values[0:119018, 1]
    gc.collect()

    word_to_idx = {word: i+1 for i, word in enumerate(vocab)}
    idx_to_word = {i+1: word for i, word in enumerate(vocab)}

    data_x = encode(data_x, vocab, word_to_idx)
    data_x = padding(data_x, maxlen=sequence_len)

    weight = torch.zeros(vocab_size+1, vector_size)

    for idx, word in idx_to_word.items():
        try:
            weight[idx, :] = torch.from_numpy(wv_model.wv.get_vector(word))
        except:
            continue        
            
    # assert (weight[0] == torch.zeros(vector_size))
    # print(weight[0])
    # print(torch.zeros(vector_size))

    print('to floattensor')
    data_x = torch.LongTensor(data_x)
    data_y = torch.LongTensor(data_y)

    print('first permutation')
    n = len(data_y)
    p = np.random.permutation(n)
    data_x = data_x[p]
    data_y = data_y[p]

    n_val = round(n * (1 - val_num))

    train_x = data_x[:n_val]
    gc.collect()
    train_y = data_y[:n_val]
    gc.collect()
    val_x = data_x[n_val:]
    gc.collect()
    val_y = data_y[n_val:]
    gc.collect()

    print('inupt data process done')

    return weight, train_x, train_y, val_x, val_y


class my_RNN_Net(nn.Module):
    def __init__(self, weight):
        super(my_RNN_Net, self).__init__()
        # self.embedding = nn.Embedding.from_pretrained(weight)
        # self.embedding.weight.requires_grad = False
        self.input_size = 250
        self.dropout = 0.5
        self.num_hiddens = 200
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

        states, hidden_out = self.encoder(embeddings.permute([1, 0, 2]))
        encoding = torch.cat([states[0], states[-1]], dim=1)
        outputs = self.decoder(encoding)

        return outputs


def main():

    weight, train_x, train_y, val_x, val_y = readfile_from_csv('/content/drive/My Drive/ML2019/hw6/hw6_data/train_x.csv',
                                                               '/content/drive/My Drive/ML2019/hw6/hw6_data/test_x.csv',
                                                               '/content/drive/My Drive/ML2019/hw6/hw6_data/train_y.csv',
                                                               '/content/drive/My Drive/ML2019/hw6/word2vec_noHMM.wv',
                                                               val_num=0.2)

    train_set = TensorDataset(train_x, train_y)
    val_set = TensorDataset(val_x, val_y)

    num_epoch = 20
    batch_size = 1024

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=8)
    val_loader = DataLoader(val_set,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=8)

    model = my_RNN_Net(weight).cuda()
    model.init_weights()
    train_loss_log = []
    train_acc_log = []
    val_loss_log = []
    val_acc_log = []

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     optimizer = torch.optim.SGD(model.parameters(), lr=1)
    best_acc = 0.0

    for epoch in range(num_epoch):
#         print(model.embedding.weight)
        
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        print('Epoch [%03d/%03d]' % (epoch+1, num_epoch))
        model.train()
        batch_num = len(train_loader)
        # msg = '...........'
        print('...........')
        for i, data in enumerate(train_loader):
            # print (msg,end = '', flush=True)
            optimizer.zero_grad()
#########GPU#########

            train_pred = model(data[0].cuda())
            batch_loss = loss(train_pred, data[1].cuda())
            batch_loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()

            train_acc += np.sum(
                np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            train_loss += batch_loss.item()

        model.eval()
        val_batch_num = len(val_loader)
        for i, data in enumerate(val_loader):
            #########GPU#########
            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].cuda())

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(),
                                        axis=1) == data[1].numpy())
            val_loss += batch_loss.item()

        val_acc = val_acc / val_set.__len__()
        train_acc = train_acc/train_set.__len__()
        print('[%0 3d/%03d] %2.2f sec(s) train_acc: %3.6f Loss: %3.6f | val_acc: %3.6f val_loss: %3.6f' %
              (batch_num, batch_num, time.time()-epoch_start_time,
               train_acc, train_loss/batch_num, val_acc, val_loss/val_batch_num))
        print('')

        # log
        train_loss_log.append(train_loss/batch_num)
        train_acc_log.append(train_acc)
        val_loss_log.append(val_loss/val_batch_num)
        val_acc_log.append(val_acc)

        if(val_acc > best_acc):
            with open('/content/drive/My Drive/ML2019/hw6/models/acc.txt', 'w') as f:
                f.write('-BEST MODEL -\nepoch: ' + str(epoch)+'/' +
                        str(num_epoch)+'\t'+'val_acc: '+str(val_acc)+'\n')
            torch.save(
                model, '/content/drive/My Drive/ML2019/hw6/models/best_model.pth')
            best_acc = val_acc
            print('** Best Model Updated! ***\n')

    torch.save(model,
               '/content/drive/My Drive/ML2019/hw6/models/final_model.pth')

    # log
    train_loss_log = np.array(train_loss_log)
    train_acc_log = np.array(train_acc_log)
    val_loss_log = np.array(val_loss_log)
    val_acc_log = np.array(val_acc_log)

    np.save('/content/drive/My Drive/ML2019/hw6/models/train_loss_log', train_loss_log)
    np.save('/content/drive/My Drive/ML2019/hw6/models/train_acc_log', train_acc_log)
    np.save('/content/drive/My Drive/ML2019/hw6/models/val_loss_log', val_loss_log)
    np.save('/content/drive/My Drive/ML2019/hw6/models/val_acc_log', val_acc_log)

if __name__ == '__main__':
    zero_vector = np.zeros(100)
    # jieba.set_dictionary('/content/drive/My Drive/ML2019/hw6/dict.txt.big')
    jieba.set_dictionary(sys.argv[])
    main()
