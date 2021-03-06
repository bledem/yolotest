import time
import cv2
import glob
import os
import sys
import numpy as np
from chainer import serializers, Variable
import chainer.functions as F
import argparse
from yolov2 import *


class drinkPredictor:
    def __init__(self):
        # hyper parameters
        weight_file = "./backup/10500.model"
        self.n_classes = 3
        self.n_boxes = 5
        self.detection_thresh = 0.6
        self.iou_thresh = 0.6
        #self.label_file = "./data/label.txt"
        self.label_file = "./XmlToTxt/classes.txt"
        with open(self.label_file, "r") as f:
            self.labels = f.read().strip().split("\n")

        # load model
        print("loading drink model...")
        yolov2 = YOLOv2(n_classes=self.n_classes, n_boxes=self.n_boxes)
        model = YOLOv2Predictor(yolov2)
        serializers.load_hdf5(weight_file, model) # load saved model
        model.predictor.train = False
        model.predictor.finetune = False
        self.model = model

    def __call__(self, orig_img):
        orig_input_height, orig_input_width, _ = orig_img.shape
        #img = cv2.resize(orig_img, (640, 640))
        img = reshape_to_yolo_size(orig_img)
        input_height, input_width, _ = img.shape
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img[:,:,:3]
        img = np.asarray(img, dtype=np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        x_data = img[np.newaxis, :, :, :]
        #x = np.asarray(x_data, dtype=np.float32)
        x_data = Variable(x_data)

        if hasattr(cuda, "cupy"):
            cuda.get_device(0).use()
            self.model.to_gpu()
            x_data.to_gpu()
        # forward

        x, y, w, h, conf, prob = self.model.predict(x_data)

        # parse results
        _, _, _, grid_h, grid_w = x.shape
        x = F.reshape(x, (self.n_boxes, grid_h, grid_w)).data
        y = F.reshape(y, (self.n_boxes, grid_h, grid_w)).data
        w = F.reshape(w, (self.n_boxes, grid_h, grid_w)).data
        h = F.reshape(h, (self.n_boxes, grid_h, grid_w)).data
        conf = F.reshape(conf, (self.n_boxes, grid_h, grid_w)).data
        prob = F.transpose(F.reshape(prob, (self.n_boxes, self.n_classes, grid_h, grid_w)), (1, 0, 2, 3)).data
        print("conf shape", conf.shape, "prob shape", prob.shape)

        detected_indices = (conf * prob).max(axis=0) > self.detection_thresh
        selected = []
        break_p = False
        for i in range(detected_indices.shape[0]):

            for j in range(detected_indices[i].shape[0]):

                for k in range(detected_indices[i][j].shape[0]):
                    if (detected_indices[i][j][k] == True):
                        #print('detected indice', i, " ", j, detected_indices[i], detected_indices[i][j] )
                        selected.append(detected_indices[i])
                        break_p = True
                        break
                if (break_p==True):
                    break_p=False
                    break
        selected = np.asarray(selected, dtype=np.int32)
        #selected=Variable(selected)
        #print('detected indice', prob.transpose(1, 2, 3, 0)[detected_indices])
        #print('detected argmax', prob.transpose(1, 2, 3, 0)[detected_indices][0].argmax())
        #print('prob transpose', prob.transpose(1, 2, 3, 0).shape)
        #print('prob in', prob.transpose(1, 2, 3, 0)[0])
        #print('prob 3', prob.transpose(1, 2, 3, 0)[0][0].argmax())


        results = []
        #sum is the number of true, so for every true we do add the results
        # we use a Boolean mask to extract all the lines corresponding to True in /prob.transpose(1, 2, 3, 0)[detected_indices]/
        for i in range(int(detected_indices.sum())):
            results.append({
                "label_nb":int(prob.transpose(1, 2, 3, 0)[detected_indices][i].argmax()),
                "label": self.labels[int(prob.transpose(1, 2, 3, 0)[detected_indices][i].argmax())],
                "probs": prob.transpose(1, 2, 3, 0)[detected_indices][i],
                "conf" : conf[detected_indices][i],
                "objectness": conf[detected_indices][i] * prob.transpose(1, 2, 3, 0)[detected_indices][i].max(),
                "box"  : Box(
                            x[detected_indices][i],
                            y[detected_indices][i],
                            w[detected_indices][i],
                            h[detected_indices][i])
            })

        # nms
        nms_results = nms(results, self.iou_thresh)
        return nms_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="give folder for mAP")
    parser.add_argument('-list', '--list', help="give path of img list" )
    parser.add_argument('-img', '--img', help="give path of img folder" )


    args = vars(parser.parse_args())
    # argument parse
    print("loading ImageNet generator")
    predictor = drinkPredictor()

    with open(args["list"]) as f:
        content = f.readlines()
        print("content", content)
    img_names = [args["img"]+"/"+x.strip()+".JPEG" for x in content]


    for img in range(len(content)):
        image_name = content[img].split("\n",1)[0]
        print("content", image_name)


        file = open("XmlToTxt/predictions/"+image_name+".txt", "+w")

        image_file = img_names[img]
        print("img file", image_file)

        # read image
        #print("loading image...")
        #print("args", image_file)
        orig_img = cv2.imread(image_file)
        nms_results = predictor(orig_img)
        for result in nms_results :
            print('%s %.3f %.3f %.3f %.3f %.3f \n' % (result["label_nb"],result["probs"].max()*result["conf"],result["box"].x, result["box"].y,result["box"].w,result["box"].h))
            file.write('%s %.3f %.3f %.3f %.3f %.3f \n' % (result["label_nb"],result["probs"].max()*result["conf"],result["box"].x, result["box"].y,result["box"].w,result["box"].h))
        file.close()


