import cv2
import numpy as np
from chainer import serializers, Variable, cuda
import chainer.functions as F
from yolov2_predict import drinkPredictor
from yolov2 import *
from chainer_sort.models import SORTMultiObjectTracking
import sys


cap = cv2.VideoCapture(1)
nakbot_predictor = drinkPredictor()
model = SORTMultiObjectTracking(
    detector, voc_bbox_label_names,
    sort_label_names)

while(True):
    ret, orig_img = cap.read()

    nms_results = nakbot_predictor(orig_img)

    # draw result
    for result in nms_results:
        left, top = result["box"].int_left_top()
        cv2.rectangle( orig_img, result["box"].int_left_top(), result["box"].int_right_bottom(),(255, 0, 255), 3)
        text = '%s(%2d%%)' % (result["label"], result["probs"].max()*result["conf"]*100)
        cv2.putText(orig_img, text, (left, top-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        print(text)
    cv2.imshow("Prediction result", orig_img)
    cv2.waitKey(1)




#    nms_results = coco_predictor(orig_img)

#    # draw result
#    for result in nms_results:
#        #x = cuda.to_cpu(result["box"].x)
#        box = Box(result["box"].x, result["box"].y, result["box"].w, result["box"].h)
#        left, top = box.int_left_top()
#        right, bottom = result["box"].int_right_bottom()
#        cv2.rectangle(orig_img, (left, top), (right, bottom), (255, 0, 255), 3)
#        text = '%s(%2d%%)' % (result["label"], result["probs"].max()*result["conf"]*100)
#        cv2.putText(orig_img, text, (left, top-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#        print(text)

#    cv2.imshow("w", orig_img)
#    cv2.waitKey(1)
