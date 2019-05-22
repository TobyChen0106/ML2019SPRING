import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader
from skimage import io
import numpy as np
from PIL import Image
import time


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),  # b, 64, 32, 32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),  # b, 64, 32, 32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.pool_1 = nn.MaxPool2d(2, stride=2, return_indices=True)  # b, 64, 16, 16
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, stride=1, padding=1),  # b, 32, 16, 16
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),  # b, 32, 16, 16
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )
        self.pool_2 = nn.MaxPool2d(2, stride=2, return_indices=True)  # b, 32, 8, 8

        # code = (batch, 512)
        self.encode_fc = nn.Sequential(
            nn.Linear(32*8*8, 32*8*8),
            nn.Linear(32*8*8, 8*8*8)
        )
        self.decode_fc = nn.Sequential(
            nn.Linear(8*8*8, 32*8*8),
            nn.Linear(32*8*8, 32*8*8)
        )

        self.decoder_1 = nn.Sequential(

            nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1),  # b, 32, 8, 8
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1),  # b, 32, 8, 8
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )
        self.unpool_1 = nn.MaxUnpool2d(2, stride=2)  # b, 32, 16, 16

        self.decoder_2 = nn.Sequential(
            nn.ConvTranspose2d(32, 64, 3, stride=1,
                               padding=1),  # b, 64, 16, 16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 64, 3, stride=1,
                               padding=1),  # b, 64, 16, 16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )

        self.unpool_2 = nn.MaxUnpool2d(2, stride=2)  # b, 64, 32, 32

        self.decoder_3 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1),  # b, 3, 32, 32
            nn.Tanh()
        )

    def forward(self, x):
        batchsize = x.shape[0]

        # print('encoder')
        x = self.encoder_1(x)
        x, indice_1 = self.pool_1(x)
        x = self.encoder_2(x)
        x, indice_2 = self.pool_2(x)

        # print('view1')
        x = x.view(batchsize, 32*8*8)
        # print('encode_fc')
        x = self.encode_fc(x)
        # print('decode_fc')
        x = self.decode_fc(x)
        # print('view2')
        x = x.view(batchsize, 32, 8, 8)
        # print('decoder')
        x = self.decoder_1(x)
        x = self.unpool_1(x, indice_2)
        x = self.decoder_2(x)
        x = self.unpool_2(x, indice_1)
        x = self.decoder_3(x)
        return x

    def encode(self, images):
        batchsize = images.shape[0]
        x = self.encoder_1(images)
        x, indice_1 = self.pool_1(x)
        x = self.encoder_2(x)
        x, indice_2 = self.pool_2(x)
        x = x.view(batchsize, 32*8*8)
        x = self.encode_fc(x)
        return indice_1, indice_2, x

    def decode(self, indice_1, indice_2, code):
        batchsize = code.shape[0]
        x = self.decode_fc(code)
        x = x.view(batchsize, 32, 8, 8)
        x = self.decoder_1(x)
        x = self.unpool_1(x, indice_2)
        x = self.decoder_2(x)
        x = self.unpool_2(x, indice_1)
        x = self.decoder_3(x)
        return x


def read_image(input_dir, val=0):
    print('reading images...')
#     images = []
#     for i in range(40000):
#         image = Image.open(input_dir+'%06d.jpg' % (i+1))
#         image = np.array(image)
#         image = (image - 127.5)/127.5   # (-1 ~ 1)
#         images.append(image)
#     images = np.array(images)
#     np.save('images',images)
    images = np.load('/content/drive/My Drive/ML2019/hw7/data/images.npy')
    print('iamges shape: ', images.shape)
    print('iamges dtype: ', images.dtype)
    # print(images[0])

    if(val != 0):
        n = images.shape[0]
        p = np.random.permutation(n)
        images = images[p]

        n_val = round(n * (1 - val))

        train_x = images[:n_val]
        val_x = images[n_val:]
        return torch.FloatTensor(train_x).permute(0, 3, 1, 2), torch.FloatTensor(val_x).permute(0, 3, 1, 2)
    else:
        return torch.FloatTensor(images).permute(0, 3, 1, 2), []


if __name__ == '__main__':
    train_x, val_x = read_image('data/images/', val=0.2)

    train_set = TensorDataset(train_x)
    val_set = TensorDataset(val_x)

    num_epoch = 500
    batch_size = 1000

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=8)
    val_loader = DataLoader(val_set,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=8)

    model = autoencoder().cuda()
#     model = autoencoder()

    train_loss_log = []
    train_acc_log = []
    val_loss_log = []
    val_acc_log = []

    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_loss = 100.0

    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        print('Epoch [%03d/%03d]' % (epoch+1, num_epoch))
        model.train()
        batch_num = len(train_loader)
        print('...........')
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            # print(data[0].shape)
            train_pred = model(data[0].cuda())
            batch_loss = loss(train_pred, data[0].cuda())
#             train_pred = model(data[0])
#             batch_loss = loss(train_pred, data[0])
            batch_loss.backward()
            optimizer.step()

            train_loss += batch_loss.item()

        model.eval()
        val_batch_num = len(val_loader)
        for i, data in enumerate(val_loader):

            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[0].cuda())
#             val_pred = model(data[0])
#             batch_loss = loss(val_pred, data[0])
            val_loss += batch_loss.item()

        print('[%0 3d/%03d] %2.2f sec(s)  train_Loss: %3.6f |  val_Loss: %3.6f' %
              (batch_num, batch_num, time.time()-epoch_start_time,
               train_loss/batch_num, val_loss/val_batch_num))
        print('')

        # log
        # train_loss_log.append(train_loss/batch_num)
        # train_acc_log.append(train_acc)
        # val_loss_log.append(val_loss/val_batch_num)
        # val_acc_log.append(val_acc)

        if(val_loss/val_batch_num < best_loss):
            # with open('/content/drive/My Drive/ML2019/hw6/models/acc.txt', 'w') as f:
            with open('/content/drive/My Drive/ML2019/hw7/models/acc.txt', 'w') as f:
                f.write('-BEST MODEL -\nepoch: ' + str(epoch)+'/' +
                        str(num_epoch)+'\t'+'val_loss: '+str(val_loss/val_batch_num)+'\n')
            torch.save(
                model.state_dict(), '/content/drive/My Drive/ML2019/hw7/models/best_model.pth')
            best_loss = val_loss/val_batch_num
            print('** Best Model Updated! ***\n')

    torch.save(model.state_dict(),
               '/content/drive/My Drive/ML2019/hw7/models/final_model.pth')

    # log
    # train_loss_log = np.array(train_loss_log)
    # train_acc_log = np.array(train_acc_log)
    # val_loss_log = np.array(val_loss_log)
    # val_acc_log = np.array(val_acc_log)

    # np.save('/content/drive/My Drive/ML2019/hw6/models/train_loss_log', train_loss_log)
    # np.save('/content/drive/My Drive/ML2019/hw6/models/train_acc_log', train_acc_log)
    # np.save('/content/drive/My Drive/ML2019/hw6/models/val_loss_log', val_loss_log)
    # np.save('/content/drive/My Drive/ML2019/hw6/models/val_acc_log', val_acc_log)
