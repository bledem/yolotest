import time
import cv2
import numpy as np
import chainer
import glob
import os
from chainer import serializers, optimizers, Variable, cuda
import chainer.functions as F
from yolov2 import *
from lib.utils import *
from lib.image_generator import *
from darknet19 import *

import matplotlib
from laplotter import LossAccPlotter

def copy_conv_layer(src, dst, layers):
    for i in layers:
        src_layer = eval("src.conv%d" % i)
        dst_layer = eval("dst.conv%d" % i)
        dst_layer.W = src_layer.W
        dst_layer.b = src_layer.b

def copy_bias_layer(src, dst, layers):
    for i in layers:
        src_layer = eval("src.bias%d" % i)
        dst_layer = eval("dst.bias%d" % i)
        dst_layer.b = src_layer.b

def copy_bn_layer(src, dst, layers):
    for i in layers:
        src_layer = eval("src.bn%d" % i)
        dst_layer = eval("dst.bn%d" % i)
        dst_layer.N = src_layer.N
        dst_layer.avg_var = src_layer.avg_var
        dst_layer.avg_mean = src_layer.avg_mean
        dst_layer.gamma = src_layer.gamma
        dst_layer.eps = src_layer.eps



# hyper parameters
train_sizes = [320, 352, 384, 416, 448]
#train_sizes = [448]
#initial_weight_file = "./backup/92800img.model"
initial_weight_file = None
trained_weight_file = "./darknet19_448.model"

backup_path = "backup"
backup_file = "%s/backup.model" % (backup_path)
batch_size =10
max_batches = 20000
learning_rate = 1e-5
learning_schedules = { 
    "0"    : 1e-5,
    "500"  : 1e-4,
    "10000": 1e-5,
    "20000": 1e-6 
}

lr_decay_power = 4
momentum = 0.9
weight_decay = 0.005
n_classes = 3
n_boxes = 5
partial_layer = 18

start = time.time()
plotter = LossAccPlotter(title="YOLOv2 loss",
                         save_to_filepath="plot.png",
                         show_regressions=True,
                         show_averages=True,
                         show_loss_plot=True,
                         show_acc_plot=False,
                         show_plot_window=True,
                         x_label="Number of input images")

# load image generator
#print("loading image generator...")
#generator = ImageGenerator(item_path, background_path)
print("loading ImageNet generator")
imageNet_data = ImageNet_data("./XmlToTxt/water_bottle_img", "./XmlToTxt/water_bottle_bbox", "./XmlToTxt/images_list.txt", n_classes)

# load model
print("loading initial model...")
trained_model = Darknet19()
serializers.load_npz(trained_weight_file, trained_model) # load saved model
trained_model = Darknet19Predictor(trained_model)


if initial_weight_file is None:
    print("inititializing with the darknet19_448 weights")
    yolov2 = YOLOv2(n_classes=n_classes, n_boxes=n_boxes)
    copy_conv_layer(trained_model.predictor, yolov2, range(1, partial_layer+1))
    copy_bias_layer(trained_model.predictor, yolov2, range(1, partial_layer+1))
    copy_bn_layer(trained_model.predictor, yolov2, range(1, partial_layer+1))
    model = YOLOv2Predictor(yolov2)
else :
    print("initializing with serial_load backup weight")
    yolov2 = YOLOv2(n_classes=n_classes, n_boxes=n_boxes)
    model = YOLOv2Predictor(yolov2)
    #serializers.load_hdf5(initial_weight_file, model)



model.predictor.train = True
model.predictor.finetune = True
cuda.get_device(0).use()
model.to_gpu()

optimizer = optimizers.MomentumSGD(lr=learning_rate, momentum=momentum)
optimizer.use_cleargrads()
optimizer.setup(model)
model.predictor.conv1.disable_update()
model.predictor.conv2.disable_update()
model.predictor.conv3.disable_update()
model.predictor.conv4.disable_update()
model.predictor.conv5.disable_update()
model.predictor.conv6.disable_update()
model.predictor.conv7.disable_update()
model.predictor.conv8.disable_update()
model.predictor.conv9.disable_update()
model.predictor.conv10.disable_update()
model.predictor.conv11.disable_update()
model.predictor.conv12.disable_update()
model.predictor.conv13.disable_update()
model.predictor.conv14.disable_update()
model.predictor.conv16.disable_update()
#model.predictor.conv17.disable_update()
#model.predictor.conv18.disable_update()
#model.predictor.conv19.disable_update()


optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))

# start to train
print("start training")
for batch in range(max_batches):




    if str(batch) in learning_schedules:
        optimizer.lr = learning_schedules[str(batch)]
    if batch % 80 == 0:
        #we take a random squared size we can found in train size
        input_width = input_height = train_sizes[np.random.randint(len(train_sizes))]
    # generate sample
#    x, t = generator.generate_samples(
#        n_samples=batch_size,
#        n_items=1,
#        crop_width=input_width,
#        crop_height=input_height,
#        min_item_scale=0.1,
#        max_item_scale=0.6,
#        rand_angle=45,
#        minimum_crop=1,
#        delta_hue=0.01,
#        delta_sat_scale=0.5,
#        delta_val_scale=0.5
#    )
#    x, t = generator.generate_simple_dataset(n_samples=batch_size, w_in=input_width,
#        h_in=input_height)

    x, t = imageNet_data.imageNet_yolo(n_samples=batch_size, w_in=input_width, h_in=input_height)

    for i, image in enumerate(x):
        image = np.asarray(image, dtype=np.float32) * 255.0
        image = np.transpose(image, (1, 2, 0)).copy()
        cv2.imwrite( "data/training_yolo/img0"+str(batch)+".png" , image )
        image = np.asarray(image, dtype=np.float32) / 255.0

        width, height, _ = image.shape
#        for truth_box in t[i]:
#            box_x, box_y, box_w, box_h = truth_box['x']*width, truth_box['y']*height, truth_box['w']*width, truth_box['h']*height
#            image = cv2.rectangle(image.copy(), (int(box_x-box_w/2), int(box_y-box_h/2)), (int(box_x+box_w/2), int(box_y+box_h/2)), (0, 0, 255), 3)
#        cv2.imshow("feed images", image)
#        cv2.waitKey(200)


    x = Variable(x)
    x.to_gpu()


    # forward
    print("Computing the loss")
    loss, h1 = model(x, t)
    now = time.time() - start
    print("batch: %d     input size: %dx%d     learning rate: %f    loss: %f time: %f" % (batch, input_height, input_width, optimizer.lr, loss.data, now))
    print("/////////////////////////////////////")
    now = time.time() - start
    #print("[batch %d (%d images)] learning rate: %f, loss: %f, accuracy: %f, time: %f" % (batch+1, (batch+1) * batch_size, optimizer.lr, loss.data, accuracy.data, now))
    #to avoid the first loss very high errors (due to??)
    if batch*batch_size > 1000 :
        plotter.add_values(batch*batch_size,loss_train=loss.data)
    # backward and optimize
    model.cleargrads()
    h1.unchain_backward()
    loss.backward()
    print("Updating the weights")
    optimizer.update()


    if (batch+1) %1000 == 0:
        model_file = "%s/%s.model" % (backup_path, batch+1)
        print("saving model to %s" % (model_file))
        serializers.save_hdf5(model_file, model)
        serializers.save_hdf5(backup_file, model)

        print ("check darknetNet ", trained_model.predictor.conv1.W[0],"and b /n",
         trained_model.predictor.conv1.b,
         "are the same as YOLOv2", model.predictor.conv1.W.data[0],  "and b /n")

print("saving model to %s/yolov2_final.model" % (backup_path))
serializers.save_hdf5("%s/yolov2_final.model" % (backup_path), model)


model.to_cpu()
serializers.save_hdf5("%s/yolov2_final_cpu.model" % (backup_path), model)
