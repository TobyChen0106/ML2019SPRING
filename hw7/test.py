import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.io import imread, imsave

test_pred = np.load('data/train_pred (1).npy')[:32]

for i, img in enumerate(test_pred):
    img = img.transpose(1, 2, 0)
    # print(img)
    img = (img+1)*127.5
    # print(img)

    img = np.array(img, dtype = 'uint8')
    img = Image.fromarray(img)
    # img.save('output/%03d.jpg'%(i))
    

    plt.figure(num='filters', figsize=(4, 8))
    plt.subplot(4, 8, i + 1)
    plt.axis('off') 
    plt.imshow(img)
plt.show()

for i in range(32):
    img=imread('data/images/%06d.jpg'%(i+1))
    plt.figure(num='filters', figsize=(4, 8))
    plt.subplot(4, 8, i + 1)
    plt.axis('off') 
    plt.imshow(img)
plt.show()
