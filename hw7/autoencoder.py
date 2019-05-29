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

data_transformations = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(size=32,
                                    scale=(0.7, 1.0),
                                    ratio=(0.75, 1.3333333333333333),
                                    interpolation=2),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=30,
#                                     resample=Image.BILINEAR,
                                    resample=False,
                                    expand=False,
                                    center=None),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.ToTensor()
])


class MyDataset(Dataset):
    def __init__(self, t_data, augmentation = 1):
        # self.train_data = np.genfromtxt(file_path, dtype=bytes, delimiter=' ')
        # self.label = np.genfromtxt(file_path, dtype=bytes, delimiter=' ')
        self.train_data = t_data
        self.aug_size = augmentation
  
        # self.label = pd.read_csv(label_file_path, delimiter=' ', header = -1)
        # print(self.train_data.shape)
        # print(self.label.shape)
    def __len__(self):
        return len(self.train_data) * self.aug_size
    
    def __getitem__(self, idx):
       	
        # imread: a function that reads an image from path
        
        i  = int(idx/self.aug_size)
        if (idx % self.aug_size != 0):
#             _img = Image.fromarray(self.train_data[i])
            img = data_transformations(self.train_data[i])
            # img = np.array(new_img)
#             img = self.train_data[i]
  
            

        else:
            img = self.train_data[i]

        
        # some operations/transformations
        return img
      
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),  # b, 64, 32, 32
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # b, 64, 16, 16
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, 3, stride=1, padding=1),  # b, 128, 16, 16
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # b, 256, 8, 8
            nn.LeakyReLU(0.2)
            
        )
        self.encoder_fc = nn.Sequential(
            nn.Linear(256*8*8, 256),
            nn.LeakyReLU(0.2)
        )
        self.decoder_fc = nn.Sequential(
            nn.Linear(256, 256*8*8),
            nn.LeakyReLU(0.2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding = 1),  # b, 128, 16, 16
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding = 1),  # b, 64, 32, 32
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),  # b, 64, 32, 32
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1),  # b, 3, 32, 32
            nn.Tanh()
        )

        self.encoder.apply(gaussian_weights_init)
        self.encoder_fc.apply(gaussian_weights_init)
        self.decoder_fc.apply(gaussian_weights_init)
        self.decoder.apply(gaussian_weights_init)

    def forward(self, x):
        x=self.encoder(x)
        x= x.view(x.shape[0], 256*8*8)
        x= self.encoder_fc(x)
        x= self.decoder_fc(x)
        x = x.view(x.shape[0],256,8,8)
        x=self.decoder(x)

        return x

    def encode(self, images):
        x=self.encoder(images)
        x= x.view(images.shape[0], 256*8*8)
        x= self.encoder_fc(x)
        return x

    def decode(self, code):
        x= self.decoder_fc(code)
        x = x.view(code.shape[0],256,8,8)
        x=self.decoder(x)
        return x

def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)
        
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
    images = np.load('data/images.npy')
    
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
    train_x, val_x = read_image('data/images/', val=0.1)

    print(train_x.shape)
    train_set = MyDataset(train_x, augmentation=5)
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
            train_pred = model(data.cuda())
            batch_loss = loss(train_pred, data.cuda())
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
            with open('models/acc.txt', 'w') as f:
                f.write('-BEST MODEL -\nepoch: ' + str(epoch)+'/' +
                        str(num_epoch)+'\t'+'val_loss: '+str(val_loss/val_batch_num)+'\n')
            torch.save(
                model.state_dict(), 'models/best_model.pth')
            best_loss = val_loss/val_batch_num
            print('** Best Model Updated! ***\n')

    torch.save(model.state_dict(),
               'models/final_model.pth')

    # log
    # train_loss_log = np.array(train_loss_log)
    # train_acc_log = np.array(train_acc_log)
    # val_loss_log = np.array(val_loss_log)
    # val_acc_log = np.array(val_acc_log)

    # np.save('/content/drive/My Drive/ML2019/hw6/models/train_loss_log', train_loss_log)
    # np.save('/content/drive/My Drive/ML2019/hw6/models/train_acc_log', train_acc_log)
    # np.save('/content/drive/My Drive/ML2019/hw6/models/val_loss_log', val_loss_log)
    # np.save('/content/drive/My Drive/ML2019/hw6/models/val_acc_log', val_acc_log)
