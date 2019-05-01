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


def cut_raw(train_file):
    print('reading_csv...')
    train_x = np.genfromtxt(train_file, delimiter=',',
                            skip_header=1, usecols=1, dtype='U20')
    output = []
    for in_seq in train_x:
        out_seq = list(jieba.cut(in_seq, cut_all=False, HMM=True))
        # while ' ' in out_seq:
        #     out_seq.remove(' ')
        output.append(out_seq)
    return output


def word2vector(input_in_seqs, word2vec_model_path):
    model = KeyedVectors.load(word2vec_model_path, mmap='r')

    output_vectors = []
    for in_seq in input_in_seqs:
        out_vector = []
        in_seq_size = len(in_seq)

        for word_count in range(100):
            if(word_count < in_seq_size):
                in_vec = model.get_vector(in_seq[word_count])
                out_vector.append(in_vec)
            else:
                in_vec = model.get_vector(' ')
                out_vector.append(in_vec)

        assert(len(out_vector) == 100)
        output_vectors.append(out_vector)
    return output_vectors


def readfile_from_csv(train_file_path, label__file_path, word2vec_model_path, val_num=0.2):
    print("Reading csv File...")

    train_x = cut_raw(train_file_path)
    train_y = np.genfromtxt(train_file_path, delimiter=',',
                            skip_header=1, usecols=1, dtype='int32')
    print(train_y.shape)

    train_x = np.array(word2vector(train_x, word2vec_model_path))
    print(train_x.shape)

    train_x = train_x.reshape(-1, 1, 100*100)

    n = len(train_y)
    p = np.random.permutation(n)
    train_x = train_x[p]
    train_y = train_y[p]

    n_val = round(n * (1 - val_num))

    train_x = train_x[:n_val]
    train_y = train_y[:n_val]
    val_x = train_x[n_val:]
    val_y = train_y[n_val:]

    train_x = torch.FloatTensor(train_x)
    val_x = torch.FloatTensor(val_x)

    train_y = torch.LongTensor(train_y)
    val_y = torch.LongTensor(val_y)

    return train_x, train_y, val_x, val_y


class my_RNN_Net(nn.Module):
    def __init__(self):
        super(my_RNN_Net, self).__init__()
        # self.embedding = nn.Embedding.from_pretrained(weight)
        # self.embedding.weight.requires_grad = False
        self.input_size = 100*100
        self.dropout = 0.5
        self.num_hiddens = 1024
        self.num_layers = 2
        self.bidirectional = True

        self.encoder = nn.LSTM(self.input_size, hidden_size=self.num_hiddens,
                               num_layers=self.num_layers, bidirectional=self.bidirectional,
                               dropout=self.dropout)
        if self.bidirectional:
            self.decoder = nn.Linear(self.num_hiddens * 4, 2)
        else:
            self.decoder = nn.Linear(self.num_hiddens * 2, 2)

    def forward(self, inputs):
        # embeddings = self.embedding(inputs)
        states, hidden = self.encoder(inputs)
        encoding = torch.cat([states[0], states[-1]], dim=1)
        outputs = self.decoder(encoding)
        return outputs


def main():

    train_x, train_y, val_x, val_y = readfile_from_csv('hw6_data/train_x.csv',
                                                       'hw6_data/train_y.csv',
                                                       'word2vec.wv',
                                                       val_num=0.2)

    train_set = TensorDataset(train_x, train_y)
    val_set = TensorDataset(val_x, val_y)

    num_epoch = 50
    batch_size = 128

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=8)
    val_loader = DataLoader(val_set,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=8)

#########GPU#########
    model = my_RNN_Net().cuda()
    train_loss_log = []
    train_acc_log = []
    val_loss_log = []
    val_acc_log = []

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_acc = 0.0

    for epoch in range(num_epoch):
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
            # train_pred = model(data[0])
            # batch_loss = loss(train_pred, data[1])

            batch_loss.backward()
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
        # train_loss_log.append(train_loss/batch_num)
        # train_acc_log.append(train_acc)
        # val_loss_log.append(val_loss/val_batch_num)
        # val_acc_log.append(val_acc)


'''
        if (val_acc > best_acc):
            with open('/content/drive/My Drive/ML2019/hw3_torch/save/acc.txt','w') as f:
                f.write('-BEST MODEL -\nepoch: ' + str(epoch)+'/'+str(num_epoch)+'\t'+'val_acc: '+str(val_acc)+'\n')
            torch.save(model.state_dict(), '/content/drive/My Drive/ML2019/hw3_torch/save/L_best_model.pth')
            best_acc = val_acc
            print('** Best Model Updated! ***\n')
'''
    if (val_acc > best_acc):
        with open('models/acc.txt','w') as f:
            f.write('-BEST MODEL -\nepoch: ' + str(epoch)+'/'+str(num_epoch)+'\t'+'val_acc: '+str(val_acc)+'\n')
        torch.save(model.state_dict(), 'models/best_model.pth')
        best_acc = val_acc
        print('** Best Model Updated! ***\n')

    # if (val_acc > 0.67):
    #     path = '/content/drive/My Drive/ML2019/hw3_torch/save/models/L_%1.4f_model.pth'%(val_acc)
    #     torch.save(model.state_dict(), path)
    #     print ('**  Model Saved! ***\n')

    # path = '/content/drive/My Drive/ML2019/hw3_torch/save/'
    # np.save(path+'train_loss_log', train_loss_log)
    # np.save(path+'train_acc_log', train_acc_log)
    # np.save(path+'val_loss_log', val_loss_log)
    # np.save(path+'val_acc_log', val_acc_log)

    # dataset = MyDataset("train_data.npy", "test.npy")
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
torch.save(model.state_dict(), 'models/final_model.pth')

if __name__ == '__main__':
    main()
