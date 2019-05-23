import os
import sys
import numpy as np
from skimage.io import imread, imsave
import gc

IMAGE_PATH = sys.argv[1]

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

    mean = np.mean(training_data, axis=0)
    training_data -= mean
    gc.collect()

    # Use SVD to find the eigenvectors
    u, s, v = np.linalg.svd(training_data.T, full_matrices=False)
    # u = np.load("models/u.npy")
    # s = np.load("models/s.npy")
    # v = np.load("models/v.npy")

    print('u', u.shape)
    print('s', s.shape)
    print('v', v.shape)

    
    # Load image & Normalize
    picked_img = imread(sys.argv[2])
    X = X = np.array(picked_img, dtype='float32').flatten()
    X -= mean

    # Compression
    weight = np.array([X.dot(u.T[i]) for i in range(5)])

    # Reconstruction
    reconstruct = process(weight.dot(u.T[:5]) + mean)
    imsave(sys.argv[3], reconstruct.reshape(img_shape))
