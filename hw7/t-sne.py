import numpy as np
# from sklearn.manifold import TSNE
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
from sklearn.cluster import KMeans
from MulticoreTSNE import MulticoreTSNE as TSNE

# class autoencoder(nn.Module):
#     def __init__(self):
#         super(autoencoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 64, 3, stride=1, padding=1),  # b, 64, 32, 32
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(64, 32, 3, stride=2, padding=1),  # b, 32, 16, 16
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(32, 16, 3, stride=2, padding=1),  # b, 16, 8, 8
#             nn.BatchNorm2d(16),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 4, 4
#             nn.BatchNorm2d(8),
#             nn.LeakyReLU(0.2)
#         )


#         # code = (batch, 512)
#         self.encode_fc = nn.Sequential(
# #             nn.Linear(8*8*8, 32*8*8),
#             nn.Linear(8*8*8, 2*8*8)
#         )
#         self.decode_fc = nn.Sequential(
# #             nn.Linear(8*8*8, 32*8*8),
#             nn.Linear(2*8*8, 8*8*8)
#         )

#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1, output_padding = 1),  # b, 16, 8, 8
#             nn.BatchNorm2d(16),
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(16, 32, 3, stride=2, padding=1, output_padding = 1),  # b, 32, 16, 16
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(32, 64, 3, stride=2,
#                                padding=1, output_padding = 1),  # b, 64, 32, 32
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2),
# #             nn.ConvTranspose2d(64, 64, 3, stride=1,
# #                                padding=1),  # b, 64, 16, 16
# #             nn.BatchNorm2d(64),
# #             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1),  # b, 3, 32, 32
#             nn.Tanh()
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x

#     def encode(self, images):
#         return self.encoder(images)

#     def decode(self, indice_1, indice_2, code):
#         return self.decoder(code)
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),  # b, 64, 32, 32
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),  # b, 64, 32, 32
            nn.LeakyReLU(0.2)
        )
        self.pool_1 = nn.MaxPool2d(2, stride=2, return_indices=True)  # b, 64, 16, 16
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1),  # b, 32, 16, 16
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),  # b, 32, 16, 16
            nn.LeakyReLU(0.2)
        )
        self.pool_2 = nn.MaxPool2d(2, stride=2, return_indices=True)  # b, 32, 8, 8

        # code = (batch, 512)
        self.encode_fc = nn.Sequential(
#             nn.Linear(32*8*8, 32*8*8),
            nn.Linear(256*8*8, 256),
            nn.LeakyReLU(0.2)
        )
        self.decode_fc = nn.Sequential(
            nn.Linear(256, 256*8*8),
            nn.LeakyReLU(0.2)
#             nn.Linear(32*8*8, 32*8*8)
        )
        
        self.unpool_1 = nn.MaxUnpool2d(2, stride=2)  # b, 256, 16, 16
        self.decoder_1 = nn.Sequential(

            nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),  # b, 32, 16, 16
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),  # b, 32, 8, 8
            nn.LeakyReLU(0.2)
        )
        self.unpool_2 = nn.MaxUnpool2d(2, stride=2)  # b, 64, 32, 32
        
        self.decoder_2 = nn.Sequential(
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
        x = x.view(batchsize, 256*8*8)
        # print('encode_fc')
        x = self.encode_fc(x)
        # print('decode_fc')
        x = self.decode_fc(x)
        # print('view2')
        x = x.view(batchsize, 256, 8, 8)
        # print('decoder')
        x = self.unpool_1(x, indice_2)
        x = self.decoder_1(x)
        x = self.unpool_2(x, indice_1)
        x = self.decoder_2(x)

        return x

    def encode(self, images):
        batchsize = images.shape[0]
        x = self.encoder_1(images)
        x, indice_1 = self.pool_1(x)
        x = self.encoder_2(x)
        x, indice_2 = self.pool_2(x)
        x = x.view(batchsize, 256*8*8)
        x = self.encode_fc(x)
        return x

    def decode(self, indice_1, indice_2, code):
        batchsize = code.shape[0]
        x = self.decode_fc(code)
        x = x.view(batchsize, 256, 8, 8)
        x = self.decoder_1(x)
        x = self.unpool_1(x, indice_2)
        x = self.decoder_2(x)
        x = self.unpool_2(x, indice_1)
        x = self.decoder_3(x)
        return x
def read_image(input_dir):
    print('reading images...')
    # images = []
    # for i in range(40000):
    #     image = Image.open(input_dir+'%06d.jpg' % (i+1))
    #     image = np.array(image)
    #     image = (image - 127.5)/127.5   # (-1 ~ 1)
    #     images.append(image)
    # images = np.array(images)
    # np.save('images',images)
    images = np.load('data/images.npy')
    print('iamges shape: ', images.shape)
    print('iamges dtype: ', images.dtype)
    # print(images[0])

    return torch.FloatTensor(images).permute(0, 3, 1, 2)

if __name__ == '__main__':
    batch_size = 300
    train_x = read_image('data/images/')
    train_set = TensorDataset(train_x)
    test_loader = DataLoader(train_set,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=8)
    
    encoded_x = []

    model = autoencoder()
    model.load_state_dict(torch.load('models/best_model_m3.pth', map_location={'cuda:0': 'cpu'}))
    model.eval()

    test_loader_num = len(test_loader)

    for i, data in enumerate(test_loader):
        # test_pred = model(data[0])
        # for i, img in enumerate(test_pred):
        #     img = img.permute(1, 2, 0)
        #     img = img.cpu().detach().numpy()
        #     img = (img+1)*127.5

        #     img = np.array(img, dtype = 'uint8')
        #     img = Image.fromarray(img)
        #     img.save('output/%03d.jpg'%(i+1))
        # break
        encoded_x_batch = model.encode(data[0])
        encoded_x_batch = encoded_x_batch.view(-1,256).cpu().detach().numpy()

        if (i == 0):
            encoded_x = encoded_x_batch
        else:
            encoded_x = np.append(encoded_x, encoded_x_batch, axis=0)
        msg = 'solving [%04d/%04d]'%(i+1,test_loader_num)
        print(msg, end = '',flush = True)
        back = '\b'*len(msg)
        print(back, end = '',flush = True)

    print('encoded_x shape', encoded_x.shape)
    print('tsne encoded_x...')
    
    # tsne = TSNE(n_components=2)
    # x_tsne = tsne.fit_transform(encoded_x)

    tsne = TSNE(n_jobs=2, n_components=8)
    x_tsne = tsne.fit_transform(encoded_x)

    np.array(x_tsne)
    print(x_tsne.shape)
    np.save('data/x_tsne_8_m3',x_tsne)

    