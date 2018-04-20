import time
import cv2
import numpy as np
import glob
from chainer import serializers, Variable
import chainer.functions as F
import argparse
from darknet19 import *

# argument parse
parser = argparse.ArgumentParser(description="photo needed")
parser.add_argument('path', help="Give the path of the image")
args = parser.parse_args()


#try to load from here
item_path = "./items"
item_files = glob.glob(item_path + "/*")
image_file = args.path

labels = []

input_height, input_width = (224, 224)
for i,item in enumerate(item_files):
    labels.append(item.split("/")[-1].split(".")[0])


weight_file_test = "./backup/101.model"
model_test = Darknet19Predictor(Darknet19())
serializers.load_hdf5(weight_file_test, model_test) # load saved model
model_test.predictor.train = False

# read image and process on it
img1 = cv2.imread(image_file)
img2 = cv2.imread("./items/schweps.png")
img3 = cv2.imread("./items/coca-zero.png")

# forward
x = []
img_list=[]

img_list.append(img1)
img_list.append(img2)
img_list.append(img3)
cv2.imshow('test feed', img1)
cv2.waitKey(1000)

#print('shape', img2.shape[:2])
for elt in img_list:
    elt = cv2.resize(elt, (input_height, input_width))

    elt = elt[:,:,:3]
    elt = np.asarray(elt, dtype=np.float32) / 255.0
    elt = elt.transpose(2, 0, 1)

    x.append(np.array(elt))

x = np.asarray(x, dtype=np.float32)
x = Variable(x)

if hasattr(cuda, "cupy"):
    cuda.get_device(0).use()
    model_test.to_gpu()
    x.to_gpu()


#y, loss, accuracy = model(x, one_hot_t)
result = model_test.predict(x).data

print('result raw', result)

result1= (result[0,:])
result2= (result[1,:])
result3= (result[2,:])


if hasattr(cuda, "cupy"):
    result1 = result1.get()
    result2= result2.get()
    result3= result3.get()



#print("final prediction", y, "accuracy", accuracy)
predicted_order = np.argsort(-result1.flatten())
predicted_order2 = np.argsort(-result2.flatten())
predicted_order3 = np.argsort(-result3.flatten())

for index in predicted_order:
    cls = labels[index]
    prob = result1.flatten()[index] * 100
    print("%16s : %.2f%%" % (cls, prob))
for index in predicted_order2:
    cls = labels[index]
    prob = result2.flatten()[index] * 100
    print("%16s : %.2f%%" % (cls, prob))
for index in predicted_order3:
    cls = labels[index]
    prob = result3.flatten()[index] * 100
    print("%16s : %.2f%%" % (cls, prob))
model_test.cleargrads()
