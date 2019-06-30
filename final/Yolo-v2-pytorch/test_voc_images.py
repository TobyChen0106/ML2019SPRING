"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
import glob
import argparse
import pickle
import cv2
import numpy as np
from src.utils import *
from src.yolo_net import Yolo

CLASSES = ['tumor']


def get_args():
    parser = argparse.ArgumentParser(
        "You Only Look Once: Unified, Real-Time Object Detection")
    parser.add_argument("--image_size", type=int, default=512,
                        help="The common width and height for all images")
    parser.add_argument("--conf_threshold", type=float, default=0.6)
    parser.add_argument("--nms_threshold", type=float, default=0.5)
    parser.add_argument("--pre_trained_model_type", type=str,
                        choices=["model", "params"], default="model")
    parser.add_argument("--pre_trained_model_path", type=str,
                        default="trained_models/only_params_trained_yolo_voc")
    parser.add_argument("--input", type=str, default="test_images")
    parser.add_argument("--output", type=str, default="test_images")

    args = parser.parse_args()
    return args


def test(opt):

    model = Yolo(1).cuda()
    model.load_state_dict(torch.load(opt.pre_trained_model_path))

    model.eval()
    colors = pickle.load(open("src/pallete", "rb"))

    result_csv = []
    result_csv.append(['patientId','x','y','width','height','Target'])

    # for id in range(4998):
    for id in range(20000,21764):
        msg = "solving [%06d/%06d]" % (id+1, 4998)
        print(msg, end='', flush=True)
        back = '\b'*len(msg)
        print(back, end='', flush=True)
        
        image_path = 'data/norm_train_image/train%05d.png' % id
        image = cv2.imread(image_path)

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        image = cv2.resize(image, (512, 512))
        image = np.transpose(np.array(image, dtype=np.float32), (2, 0, 1))
        # image = image[0]
        # image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=0)

        # print(image.shape)


        width_ratio = float(opt.image_size) / width
        height_ratio = float(opt.image_size) / height
        data = Variable(torch.FloatTensor(image))

        if (width_ratio != 1 or height_ratio != 1):
            print('\nNOT ratioa 1!!!\n')
        if torch.cuda.is_available():
            data = data.cuda()
        with torch.no_grad():
            logits = model(data)
            predictions = post_processing(logits, opt.image_size, CLASSES, model.anchors, opt.conf_threshold,
                                          opt.nms_threshold)
        if len(predictions) != 0:
            predictions = predictions[0]
            output_image = cv2.imread(image_path)
            for pred in predictions:
                xmin = int(max(pred[0] / width_ratio, 0))
                ymin = int(max(pred[1] / height_ratio, 0))

                xmax = int(min((pred[0]+pred[2]) / width_ratio, width))
                ymax = int(min(pred[1]+(pred[3]) / height_ratio, height))

                w = int(min((pred[2]) / width_ratio, width))
                h = int(min((pred[3]) / height_ratio, height))

                if(xmin+w > 511):
                    w = 511-xmin
                if(ymin+h > 511):
                    h = 511-ymin
                    
                output = ['train%05d.png' % id,xmin*2,ymin*2,w*2,h*2,1]
                result_csv.append(output)

                color = colors[CLASSES.index(pred[5])]
                cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), color, 2)
                text_size = cv2.getTextSize(pred[5] + ' : %.2f' % pred[4], cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                cv2.rectangle(output_image, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4), color, -1)
                cv2.putText(
                    output_image, pred[5] + ' : %.2f' % pred[4],
                    (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 255, 255), 1)
                print("Image: {} Object: {}, Bounding box: ({},{}) ({},{})".format(image_path, pred[5], xmin, xmax, ymin, ymax))
            cv2.imwrite('predictions/'+str(id) + "_prediction.jpg", output_image)
            # test_images/
        else:
            output = ['test%04d.png' % id,'','','','',0]
            result_csv.append(output)

    # result_csv = np.array(result_csv)
    # print(result_csv.shape)
    # np.savetxt('result.csv', result_csv, delimiter=",", fmt="%s")


if __name__ == "__main__":
    opt = get_args()
    test(opt)
