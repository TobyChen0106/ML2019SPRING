import cv2
import numpy as np
import sys
import gc

cnn_model_path = 'models/best_model_lily.pth'
img_path = 'test_map/'
test_img_num = 10  # 4998
# test_img_num_start = int(sys.argv[1])  # 4998
# test_img_num_end = int(sys.argv[1])+test_img_num  # 4998
batch_size = 19
map_stride = 16
n = 896//map_stride + 1

def load_img(path):
    print('start loading image...')
    img_ar = []
    for i in range(0, test_img_num):
        # if i%1 == 0:
            # print('loading img', i)
        img_path = path + 'test_img' + str(i) + '.png'
        img = cv2.imread(img_path)
        img = np.array(img)
        img_ar.append(img)

        gc.collect()
        msg = "loading image [%06d/%06d]" % (i+1, test_img_num)
        print(msg, end='', flush=True)
        back = '\b'*len(msg)
        print(back, end='', flush=True)
    img_ar = np.array(img_ar)
    print('\nimg_size = ', img_ar.shape)
    # np.save('/content/drive/My Drive/ML/hw7/data/images.npy', img_ar)
    print("finish loading image!\n")
    return img_ar


def selective_search(all_img):
    print('start doing selective search...')
    cv2.setUseOptimized(True)
    cv2.setNumThreads(16)
    for img in all_img:
        # left half plane
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        print('img shape = ', img[0:n//2].shape)
        ss.setBaseImage(img[0:n//2])
        # ss.switchToSelectiveSearchFast()
        ss.switchToSelectiveSearchQuality()
        rects_left = ss.process()
        new_rects_left = []
        score_left = []
        for j in range(rects_left.shape[0]):
            x, y, w, h = rects_left[j]
            area = w * h
            if w > (32/1024)*n and w < (512/1024)*n and h > (128/1024)*n and h < (512/1024)*n and x > (16/1024)*n and y > (16/1024)*n:
                new_rects_left.append(rects_left[j])
                score_left.append(np.sum(img[x:x+w, y:y+h])/area)
                print('rect = [{} {} {} {}]'.format(x, y, w, h))
                print('score = ', np.sum(
                    img[x:x+w, y:y+h])/area + area/(57*57))
        new_rects_left = np.array(new_rects_left).reshape(-1, 4)
        max_id_left = np.argmax(score_left)
        max_rect_left = new_rects_left[max_id_left]
        print('number: {}'.format(len(new_rects_left[0])))

        # right half plane
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(map1[512:, :, :])
        # ss.switchToSelectiveSearchFast()
        ss.switchToSelectiveSearchQuality()
        rects_right = ss.process()
        new_rects_right = []
        score_right = []
        for j in range(rects_right.shape[0]):
            x, y, w, h = rects_right[j]
            area = w * h
            if w > (32/1024)*n and w < (512/1024)*n and h > (128/1024)*n and h < (512/1024)*n and x > (16/1024)*n and y > (16/1024)*n:
                new_rects_right.append(rects_right[j])
                score_right.append(np.sum(map1[i, x:x+w, y:y+h])/area)
                print('rect = [{} {} {} {}]'.format(x, y, w, h))
                print('score = ', np.sum(map1[i, x:x+w, y:y+h])/area)
        new_rects_right = np.array(new_rects_right).reshape(-1, 4)
        max_id_right = np.argmax(score_right)
        max_rect_right = new_rects_right[max_id_right]
        imOut = map1.copy()
        cv2.rectangle(imOut, (max_rect_left[0], max_rect_left[1]), (
            max_rect_left[0]+max_rect_left[2], max_rect_left[1]+max_rect_left[3]), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.rectangle(imOut, (max_rect_right[0], max_rect_lright[1]), (
            max_rect_right[0]+max_rect_right[2], max_rect_right[1]+max_rect_right[3]), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow("Output", imOut)
    return imOut


if __name__ == '__main__':
    all_img = load_img(img_path)
    imgOut = selective_search(all_img)
