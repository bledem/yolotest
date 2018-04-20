import time
import cv2
import numpy as np
import chainer
import glob
import os
from chainer import serializers, optimizers, Variable, cuda, iterators
import chainer.functions as F
from darknet19 import *
from lib.image_generator import *
import matplotlib
from laplotter import LossAccPlotter

plotter = LossAccPlotter(title="YOLO loss and accuracy",
                         save_to_filepath="plot.png",
                         show_regressions=True,
                         show_averages=True,
                         show_loss_plot=True,
                         show_acc_plot=True,
                         show_plot_window=True,
                         x_label="Batch")
# hyper parameters
input_height, input_width = (224, 224)
#weight_file = "./darknet19.model"
item_path = "./items"
background_path = "./backgrounds"
label_file = "./data/label.txt"
backup_path = "./backup"
batch_size = 32
max_batches = 2000
learning_rate = 0.001
lr_decay_power = 4
momentum = 0.9
weight_decay = 0.0005

# load image generator
print("loading image generator...")
generator = ImageGenerator(item_path, background_path)
generator_debug = ImageGenerator(item_path, background_path)


with open(label_file, "r") as f:
    labels = f.read().strip().split("\n")

# load model
print("loading model...")
model = Darknet19Predictor(Darknet19())
backup_file = "%s/301.model" % (backup_path)
if os.path.isfile(backup_file):
    serializers.load_hdf5(backup_file, model) # load saved model
model.predictor.train = True
cuda.get_device(0).use()
model.to_gpu(0) # for gpu
start = time.time()

optimizer = optimizers.MomentumSGD(lr=learning_rate, momentum=momentum)
optimizer.use_cleargrads()
optimizer.setup(model)
#freeze the weights
#odel.conv1.disable_update()
optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))

item_files = glob.glob(item_path + "/*")
x_debug = []
t_debug = []


# start to train
print("start training")
for batch in range(max_batches):
    model.predictor.train = True

    #generate sample
    x, t = generator.generate_samples(
        n_samples=batch_size,
        n_items=1,
        crop_width=input_width,
        crop_height=input_height,
        min_item_scale=0.1,
        max_item_scale=1.3,
        rand_angle=45,
        minimum_crop=0.7,
        delta_hue=0.01,
        delta_sat_scale=0.5,
        delta_val_scale=0.5
    )
   # x_debug, t_debug = generator_debug.generate_simple_dataset(n_samples=batch_size, w_in=input_width,
    #    h_in=input_height)


    one_hot_t_special = [0.0,0.0,1.0,0.0]
    one_hot_t_special = np.array(one_hot_t_special, dtype=np.float32)

    x = Variable(x)
    #print('GPU infop', cuda.get_array_module(x))
    x.to_gpu(0)
    one_hot_t = []

    #t = t_debug
    for i in range(len(t)):
        one_hot_t.append(t[i][0]["one_hot_label"])
    #one_hot_t.append(one_hot_t_special)
    one_hot_t = np.array(one_hot_t, dtype=np.float32)
    one_hot_t = Variable(one_hot_t)
    one_hot_t.to_gpu(0)

    y, loss, accuracy = model(x, one_hot_t)
    print("y result", y)
    now = time.time() - start
    print("[batch %d (%d images)] learning rate: %f, loss: %f, accuracy: %f, time: %f" % (batch+1, (batch+1) * batch_size, optimizer.lr, loss.data, accuracy.data, now))

    plotter.add_values(batch,loss_train=loss.data, acc_train=accuracy.data)
    model.cleargrads()
    #model.zerograds()
    loss.backward()

    optimizer.lr = learning_rate * (1 - batch / max_batches) ** lr_decay_power # Polynomial decay learning rate
    optimizer.update()


    # save model
    if (batch) % 100== 0:
        model_file = "%s/%s.model" % (backup_path, batch+1)
        print("saving model to %s" % (model_file))
        serializers.save_hdf5(model_file, model)
        serializers.save_hdf5(backup_file, model)

        #try to load from here
#        weight_file_test = "./backup/backup.model"
#        model_test = Darknet19Predictor(Darknet19())
#        serializers.load_hdf5(weight_file_test, model_test) # load saved model
#        model_test.predictor.train = False

        # read image and process on it
        img1 = cv2.imread("./items/sprite.png")
        img2 = cv2.imread("./items/coca-zero.png")

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
        #model.predictor.train = False

        # forward
        x = []

        x.append(np.array(img1))
        x.append(np.array((img2)))
        x = np.asarray(x, dtype=np.float32)
        x = Variable(x)

        if hasattr(cuda, "cupy"):
            print('hasatt inside')
            cuda.get_device(0).use()
            model.to_gpu()
            x.to_gpu()

        one_hot_t = []
        first = [0,0,0,1]
        secund = [0,0,1,0]
        one_hot_t.append(np.array(first))
        one_hot_t.append(np.array(secund))
        one_hot_t = np.array(one_hot_t, dtype=np.float32)
        one_hot_t = Variable(one_hot_t)
        one_hot_t.to_gpu(0)

        y, loss, accuracy = model(x, one_hot_t)
        print("y result", y)
        result = model.predict(x).data
        result1= (result[0,:])
        result2= (result[1,:])

        print("result 1 ", result1, "and2", result2, labels)

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
        model.zerograds()

print("saving model to %s/darknet19_final.model" % (backup_path))
serializers.save_hdf5("%s/darknet19_final.model" % (backup_path), model)
