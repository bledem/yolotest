import time
import cv2
import numpy as np
import glob
from chainer import serializers, Variable
import chainer.functions as F
import argparse
from darknet19 import *

#try to load from here
item_path = "./items"
item_files = glob.glob(item_path + "/*")
labels = []

input_height, input_width = (224, 224)
for i,item in enumerate(item_files):
    labels.append(item.split("/")[-1].split(".")[0])


weight_file_test = "./backup/fourcans.model"
model_test = Darknet19Predictor(Darknet19())
serializers.load_hdf5(weight_file_test, model_test) # load saved model
model_test.predictor.train = False

# read image and process on it
img1 = cv2.imread("./items/schweps.png")
img2 = cv2.imread("./items/pet_water.png")

img1 = cv2.resize(img1, (input_height, input_width))
img2 = cv2.resize(img2, (input_height, input_width))

cv2.imshow('test feed', img2)
cv2.waitKey(50)
#print('shape', img2.shape[:2])

img1 = img1[:,:,:3]
img1 = np.asarray(img1, dtype=np.float32) / 255.0
img1 = img1.transpose(2, 0, 1)
img2 = img2[:,:,:3]
img2 = np.asarray(img2, dtype=np.float32) / 255.0
img2 = img2.transpose(2, 0, 1)
# load model
model_test.predictor.train = False

# forward
x = []

x.append(np.array(img1))
x.append(np.array((img2)))
x = np.asarray(x, dtype=np.float32)
x = Variable(x)

if hasattr(cuda, "cupy"):
    print('hasatt inside')
    cuda.get_device(0).use()
    model_test.to_gpu()
    x.to_gpu()

one_hot_t = []
first = [0,0,0,1]
secund = [0,0,1,0]
one_hot_t.append(np.array(first))
one_hot_t.append(np.array(secund))
one_hot_t = np.array(one_hot_t, dtype=np.float32)
one_hot_t = Variable(one_hot_t)
one_hot_t.to_gpu(0)

#y, loss, accuracy = model(x, one_hot_t)
result = model_test.predict(x).data
result1= (result[0,:])
result2= (result[1,:])

if hasattr(cuda, "cupy"):
    result1 = result1.get()
    result2= result2.get()


#print("final prediction", y, "accuracy", accuracy)
predicted_order = np.argsort(-result1.flatten())
predicted_order2 = np.argsort(-result2.flatten())

for index in predicted_order:
    cls = labels[index]
    prob = result1.flatten()[index] * 100
    print("%16s : %.2f%%" % (cls, prob))
for index in predicted_order2:
    cls = labels[index]
    prob = result2.flatten()[index] * 100
    print("%16s : %.2f%%" % (cls, prob))
model_test.zerograds()
