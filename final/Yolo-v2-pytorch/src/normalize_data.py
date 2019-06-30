import cv2
import numpy as np

test_image_path = 'data/test_image'
normalize_image_path = 'data/norm_test_image'
# 4998
# 21764
for i in range(4998):
    msg = "solving [%06d/%06d]" % (i+1, 4998)
    print(msg,  end='', flush=True)
    back = '\b'*len(msg)
    print(back, end='', flush=True)
    # print('***%d***'%i)
    image_path = '%s/test%04d.png'%(test_image_path,i)
    image = cv2.imread(image_path)
    # im = np.array(image)
    # print(im.max())
    # print(im.min())
    
    normalizedImg = np.zeros((1024, 1024))
    normalizedImg = cv2.normalize(image,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
    normalizedImg = cv2.resize(normalizedImg, (512,512))


    # im = np.array(normalizedImg)

    # print(im.max())
    # print(im.min())

    cv2.imwrite('%s/test%04d.png'%(normalize_image_path, i), normalizedImg)