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
import matplotlib
from laplotter import LossAccPlotter

# hyper parameters
train_sizes = [320, 352, 384, 416, 448]
item_path = "./items"
background_path = "./backgrounds"
initial_weight_file = "./backup/200.model"
backup_path = "backup"
backup_file = "%s/backup.model" % (backup_path)
batch_size = 8
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
n_classes = 4
n_boxes = 5

start = time.time()
plotter = LossAccPlotter(title="YOLOv2 loss and accuracy",
                         save_to_filepath="plot.png",
                         show_regressions=True,
                         show_averages=True,
                         show_loss_plot=True,
                         show_acc_plot=True,
                         show_plot_window=True,
                         x_label="Batch")

# load image generator
print("loading image generator...")
generator = ImageGenerator(item_path, background_path)

# load model
print("loading initial model...")
yolov2 = YOLOv2(n_classes=n_classes, n_boxes=n_boxes)
model = YOLOv2Predictor(yolov2)
serializers.load_hdf5(initial_weight_file, model)

model.predictor.train = True
model.predictor.finetune = False
cuda.get_device(0).use()
model.to_gpu()

optimizer = optimizers.MomentumSGD(lr=learning_rate, momentum=momentum)
optimizer.use_cleargrads()
optimizer.setup(model)
#optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))

# start to train
print("start training")
for batch in range(max_batches):
    if str(batch) in learning_schedules:
        optimizer.lr = learning_schedules[str(batch)]
    if batch % 80 == 0:
        #we take a random squared size we can found in train size
        input_width = input_height = train_sizes[np.random.randint(len(train_sizes))]

    # generate sample
    x, t = generator.generate_samples(
        n_samples=batch_size,
        n_items=2,
        crop_width=input_width,
        crop_height=input_height,
        min_item_scale=0.05,
        max_item_scale=0.6,
        rand_angle=90,
        minimum_crop=0.8,
        delta_hue=0.01,
        delta_sat_scale=0.5,
        delta_val_scale=0.5
    )
    for i, image in enumerate(x):
        image = np.transpose(image, (1, 2, 0)).copy()
        #image = np.asarray(image, dtype=np.float32) * 255.0
        #for i, image in enumerate(x):
        cv2.imshow("feed images", image)
        cv2.waitKey(60)

    x = Variable(x)
    x.to_gpu()

    # forward
    loss = model(x, t)
    now = time.time() - start
    print("batch: %d     input size: %dx%d     learning rate: %f    loss: %f time: %f" % (batch, input_height, input_width, optimizer.lr, loss.data, now))
    print("/////////////////////////////////////")
    now = time.time() - start
    #print("[batch %d (%d images)] learning rate: %f, loss: %f, accuracy: %f, time: %f" % (batch+1, (batch+1) * batch_size, optimizer.lr, loss.data, accuracy.data, now))

    plotter.add_values(batch,loss_train=loss.data, acc_train=loss.data)
    # backward and optimize
    model.cleargrads()
    loss.backward()
    optimizer.update()

    # save model
    if (batch+1) %100 == 0:
        model_file = "%s/%s.model" % (backup_path, batch+1)
        print("saving model to %s" % (model_file))
        serializers.save_hdf5(model_file, model)
        serializers.save_hdf5(backup_file, model)

print("saving model to %s/yolov2_final.model" % (backup_path))
serializers.save_hdf5("%s/yolov2_final.model" % (backup_path), model)

model.to_cpu()
serializers.save_hdf5("%s/yolov2_final_cpu.model" % (backup_path), model)
