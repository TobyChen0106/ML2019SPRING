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


class MyDataset(Dataset):
    def __init__(self, x_data, y_data, word_to_idx):
        self.x_data = x_data
        self.y_data = y_data

        self.word_to_idx = word_to_idx
        self.vocab_size = len(word_to_idx)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        bow = torch.zeros([self.vocab_size], dtype=torch.float32)

        for word in self.x_data[idx]:
            bow[self.word_to_idx[word]-1] += 1

        return bow, self.y_data[idx]


def readfile_from_csv(train_file_path, test_file_path, label__file_path, word2vec_model_path, val_num=0.2):
    data_x = cut_raw(train_file_path, remove_space=True)
    test_x = cut_raw(test_file_path, remove_space=True)

    print('constructing vocab set...')
    vocab = list_to_set(data_x+test_x)
    vocab_size = len(vocab)
    print('vocab_size = ', vocab_size)

    data_y = pd.read_csv(label__file_path).values[0:119018, 1]
    gc.collect()

    print('word_to_idx')
    word_to_idx = {word: i+1 for i, word in enumerate(vocab)}
    gc.collect()

    # print('idx_to_word')
    # idx_to_word = {i+1: word for i, word in enumerate(vocab)}
    # gc.collect()

    # print('first permutation')
    n = len(data_y)
    # p = np.random.permutation(n)
    # data_x = data_x[p]
    # data_y = data_y[p]

    data_y = torch.LongTensor(data_y)

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
    return vocab_size, word_to_idx, train_x, train_y, val_x, val_y


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

    vocab_size, word_to_idx, \
        train_x, train_y, val_x, val_y = readfile_from_csv('/content/drive/My Drive/ML2019/hw6/hw6_data/train_x.csv',
                                                           '/content/drive/My Drive/ML2019/hw6/hw6_data/test_x.csv',
                                                           '/content/drive/My Drive/ML2019/hw6/hw6_data/train_y.csv',
                                                           '/content/drive/My Drive/ML2019/hw6/word2vec_noHMM.wv',
                                                           val_num=0.2)

    train_set = MyDataset(train_x, train_y, word_to_idx)
    val_set = MyDataset(val_x, val_y, word_to_idx)

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

    model = my_BOW_DNN_Net(vocab_size).cuda()
    train_loss_log = []
    train_acc_log = []
    val_loss_log = []
    val_acc_log = []

    # log
    np.save('/content/drive/My Drive/ML2019/hw6/models/train_loss_log', train_loss_log)
    np.save('/content/drive/My Drive/ML2019/hw6/models/train_acc_log', train_acc_log)
    np.save('/content/drive/My Drive/ML2019/hw6/models/val_loss_log', val_loss_log)
    np.save('/content/drive/My Drive/ML2019/hw6/models/val_acc_log', val_acc_log)

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
                model, '/content/drive/My Drive/ML2019/hw6/models/my_BOW_DNN_Net_best_model.pth')
            best_acc = val_acc
            print('** Best Model Updated! ***\n')

    torch.save(model,
               '/content/drive/My Drive/ML2019/hw6/models/my_BOW_DNN_Net_final_model.pth')

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
    jieba.set_dictionary('/content/drive/My Drive/ML2019/hw6/dict.txt.big')
    main()
