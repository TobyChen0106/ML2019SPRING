import os
import sys
import numpy as np
from skimage.io import imread, imsave
import gc

IMAGE_PATH = 'data/Aberdeen'

# Images for compression & reconstruction
test_image = ['30.jpg', '50.jpg', '99.jpg', '137.jpg', '272.jpg']

# Number of principal components used
k = 5


def process(MM):
    M = np.array(MM)
    M -= np.min(M)
    M /= np.max(M)
    M = (M * 255).astype(np.uint8)
    return M


if __name__ == '__main__':
    # filelist = os.listdir(IMAGE_PATH)
    filelist = [str(i)+'.jpg' for i in range(415)]

    # Record the shape of images
    img_shape = imread(os.path.join(IMAGE_PATH, filelist[0])).shape

    img_data = []
    for filename in filelist:
        tmp = imread(os.path.join(IMAGE_PATH, filename))
        img_data.append(tmp.flatten())
        gc.collect()

    training_data = np.array(img_data, dtype='float32')
    # print(training_data)
    # np.save('training_data',training_data)
    gc.collect()

    # Calculate mean & Normalize

    print("p1.a")
    mean = np.mean(training_data, axis=0)
    imsave('output/mean.jpg', process(mean.reshape(img_shape)))
    training_data -= mean
    gc.collect()

    # Use SVD to find the eigenvectors
    print("p1.b")
    # u, s, v = np.linalg.svd(training_data.T, full_matrices=False)
    u = np.load("models/u.npy")
    s = np.load("models/s.npy")
    v = np.load("models/v.npy")
    # np.save('u',u)
    # np.save('s',s)
    # np.save('v',v)
    print('u', u.shape)
    print('s', s.shape)
    print('v', v.shape)

    for i in range(5):
        eigenface = process(u.T[i])
        imsave('eigenface_%d.jpg' % (i+1), eigenface.reshape(img_shape))

    print("p1.c")
    for x in test_image:
        # Load image & Normalize
        picked_img = imread(os.path.join(IMAGE_PATH, x))
        X = X = np.array(picked_img, dtype='float32').flatten()
        X -= mean

        # Compression
        weight = np.array([X.dot(u.T[i]) for i in range(5)])

        # Reconstruction
        reconstruct = process(weight.dot(u.T[:5]) + mean)
        imsave(x[:-4] + '_reconstruction.jpg', reconstruct.reshape(img_shape))

    # problem 1.d
    print("problem 1.d")
    for i in range(5):
        number = s[i] * 100 / sum(s)
        print(number)
