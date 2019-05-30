import cv2
import numpy as np
import io
import sys
import gc

test_img_num = 100  # 4998
score_limit = 0.97
n = 57
cv2.setUseOptimized(True)
cv2.setNumThreads(4)
label = []
output = []


print('start doing selective search...')

cv2.setUseOptimized(True)
cv2.setNumThreads(16)

for i in range(test_img_num):
    # msg = "solving [%06d/%06d]" % (i, test_img_num)
    # print(msg, end='', flush=True)
    # back = '\b'*len(msg)
    # print(back, end='', flush=True)

    la = False
    # img_path = 'test_map/test_img%d.png' % i
    # im_path_ori = 'data/test/test%04d.png' % (i)

    img_path = 'train_map/train_img%d.png' % i
    im_path_ori = 'data/train/train%05d.png' % (i)

    im = cv2.imread(img_path)
    # print(im.shape)
    im_ori = cv2.imread(im_path_ori)
    imOut = im.copy()
    imOut_ori = im_ori.copy()
    # print('im max',np.array(im).max())
    # print('im min',np.array(im).min())
    # left half plane
    im_left = im[:, 0:n//2]

    # im_left = cv2.resize(im_ori[:, :512], (57, 28),
    #                      interpolation=cv2.INTER_CUBIC)

    # print('im_left shape = ', im_left.shape)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    # ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyFill()
    ss.setBaseImage(im_left)

    ss.switchToSelectiveSearchQuality()
    rects_left = ss.process()
    # print('[%d] num left rects: %d' % (i, len(rects_left)))
    new_rects_left = []
    score_left = []
    l = 0
    r = 0
    # print('initial number of rest left = ', rects_left.shape[0])
    for j in range(rects_left.shape[0]):
        x, y, w, h = rects_left[j]
        # x1 = np.round(x/17.965)
        # y1 = np.round(y/17.965)
        # x2 = np.round((x+w)/17.965)
        # y2 = np.round((y+h)/17.965)
        x1 = x
        y1 = y
        x2 = x+w
        y2 = y+h
        cx = x1 + w/2
        cy = y1 + h/2
        area = w * h
        # if w > (64*n)/1024 and w < (500*n)/1024 and h > (80*n)/1024 and h < (900*n)/1024 and x > (32*n)/1024 and y > (80*n)/1024 and cy>(120*n)/1024 and  cy<(800*n)/1024:
        # if x1 > 2 and x2 > 2 and y1 > 3 and y1 < 55 and y2 > 3 and y2 < 55:
        if x1 > 5 and x2 > 5:
        # if True:
            # if x1>2  and x2>2 and y1 > 3 and y1 < 52 and y2 > 3 and y2 < 52:
            l += 1
            new_rects_left.append(rects_left[j])
            # score_left.append(np.sum(im_left[x:x+w, y:y+h])/(area*255) + area/(57*57*10))
            score_left.append(np.sum(im_left[y:y+h, x:x+w, 0])/(area*255)+0.1*h/w)

    if l > 0:
        new_rects_left = np.array(new_rects_left).reshape(-1, 4)
        max_id_left = np.argmax(score_left)

        print('[%d] best score of left = ' % i, score_left[max_id_left])

        if score_left[max_id_left] > score_limit:
            # print('!!!!left tumer detected!!!!!')
            max_rect_left = new_rects_left[max_id_left]
            # print('best rect left = ', max_rect_left)
            la = True
            # cv2.rectangle(imOut, (max_rect_left[0], max_rect_left[1]), (
            #     max_rect_left[0]+max_rect_left[2], max_rect_left[1]+max_rect_left[3]), (0, 255, 0), 1, cv2.LINE_AA)
            # cv2.rectangle(imOut_ori, (max_rect_left[0]*16 + 56, max_rect_left[1]*16 + 56), (max_rect_left[0]*16 +
            #                                                                                 56 + max_rect_left[2]*16, max_rect_left[1]*16 + 56 + max_rect_left[3]*16), (0, 255, 0), 1, cv2.LINE_AA)
            output.append(i)
            output.append(max_rect_left[0]*16 + 56)
            output.append(max_rect_left[1]*16 + 56)
            output.append(max_rect_left[2]*16)
            output.append(max_rect_left[3]*16)
            output.append(1)

    # right half plane
    im_right = im[:, n//2:]
    # im_right = cv2.resize(im_ori[:, 512:], (57, 29),
    #                       interpolation=cv2.INTER_CUBIC)

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(im_right)
    ss.switchToSelectiveSearchQuality()
    rects_right = ss.process()
    # print('[%d] num right rects: %d' % (i, len(rects_right)))
    new_rects_right = []
    score_right = []
    for j in range(rects_right.shape[0]):
        x, y, w, h = rects_right[j]  # 0 28 22 29
        x1 = x
        y1 = y
        x2 = x+w
        y2 = y+h
        cx = x1 + w/2
        cy = y1 + h/2
        area = w * h
        # if w > (64*n)/1024 and w < (500*n)/1024 and h > (80*n)/1024 and h < (900*n)/1024 and x+w+n//2 < (988*n)/1024 and y > (80*n)/1024 and cy>(120*n)/1024 and  cy<(800*n)/1024:
        # if x1 < 26 and x2 < 26 and y1 > 3 and y1 < 55 and y2 > 3 and y2 < 55:
        if x1 < 24 and x2 < 24 :
        # if True:
            r = r+1

            new_rects_right.append(rects_right[j])
            # score_right.append(np.sum(im_right[x:x+w, y:y+h])/(area*255) + area/(57*57*10))
            score_right.append(np.sum(im_right[y:y+h, x:x+w, 0])/(area*255)+0.1*h/w)

    if r > 0:
        new_rects_right = np.array(new_rects_right).reshape(-1, 4)
        max_id_right = np.argmax(score_right)

        print('[%d] best score of right = ' % i, score_right[max_id_right])

        if score_right[max_id_right] > score_limit:
            # print('!!!!right tumer detected!!!!!')
            max_rect_right = new_rects_right[max_id_right]
            la = True
            # cv2.rectangle(imOut, (max_rect_right[0] + n//2, max_rect_right[1]), (max_rect_right[0] + n //
            #                                                                      2 + max_rect_right[2], max_rect_right[1]+max_rect_right[3]), (0, 255, 0), 1, cv2.LINE_AA)
            # cv2.rectangle(imOut_ori, ((max_rect_right[0] + n//2)*16 + 56, max_rect_right[1]*16 + 56), ((max_rect_right[0] + n//2)
            #                                                                                            * 16 + 56 + max_rect_right[2]*16, max_rect_right[1]*16 + 56 + max_rect_right[3]*16), (0, 255, 0), 1, cv2.LINE_AA)
            output.append(i)
            output.append((max_rect_right[0] + n//2)*16 + 56)
            output.append(max_rect_right[1]*16 + 56)
            output.append(max_rect_right[2]*16)
            output.append(max_rect_right[3]*16)
            output.append(1)
    if la == False:
        output.append(i)
        output.append(-1)
        output.append(-1)
        output.append(-1)
        output.append(-1)
        output.append(0)

output = np.array(output).reshape(-1, 6)
np.savetxt("train_output.csv", output, delimiter=",")
# np.save('output.npy', output)
print('output shape = ', output.shape)
