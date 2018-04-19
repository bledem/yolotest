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
batch_size = 10
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
backup_file = "%s/501.model" % (backup_path)
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
labels = []
x_debug = []
t_debug = []
result_compare = []
for i,item in enumerate(item_files):
    image = cv2.imread(item, cv2.IMREAD_UNCHANGED)
    #cv2.imshow("w", image)
    #cv2.waitKey(0)
    image = np.asarray(image, dtype=np.float32) / 255.0
    image = image.transpose(2, 0, 1)
    x_debug.append(image)
    labels.append(item.split("/")[-1].split(".")[0])

for i,item in enumerate(item_files):
    one_hot_label = np.zeros(len(labels))
    one_hot_label[i] = 1
    t_debug.append(one_hot_label)
    #print("corresponding label", labels)
    #x_debug = np.array(x_debug).reshape(4,730,300)
    #print("shape x incorrect", len(x_debug))
#train_iter = iterators.SerialIterator(train, batch_size)

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
        rand_angle=25,
        minimum_crop=0.7,
        delta_hue=0.01,
        delta_sat_scale=0.5,
        delta_val_scale=0.5
    )
    x_debug, t_debug = generator_debug.generate_simple_dataset(n_samples=batch_size, w_in=input_width,
        h_in=input_height)
    #for i, image in enumerate(x):
        #for truth_box in t[i]:
           # print(truth_box['label'])
        #image = np.transpose(image, (1, 2, 0)).copy()
        #cv2.imshow("w", image)
        #cv2.waitKey(0)

    #cv2.imshow("w", image)
    #cv2.waitKey(0)

    one_hot_t_special = [0.0,0.0,1.0,0.0]
    #one_hot_t_special = np.array(one_hot_t_special, dtype=np.float32)

    x = Variable(x_debug)
    #print('GPU infop', cuda.get_array_module(x))
    x.to_gpu(0)
    one_hot_t = []
    t = t_debug
    for i in range(len(t)):
        one_hot_t.append(t[0][i]["one_hot_label"])
    one_hot_t.append(one_hot_t_special)
    one_hot_t = np.array(one_hot_t, dtype=np.float32)
    one_hot_t = Variable(one_hot_t)
    one_hot_t.to_gpu(0)

    y, loss, accuracy = model(x, one_hot_t)
    print("y result", y)
    result_compare.append(y)
    now = time.time() - start
    print("[batch %d (%d images)] learning rate: %f, loss: %f, accuracy: %f, time: %f" % (batch+1, (batch+1) * batch_size, optimizer.lr, loss.data, accuracy.data, now))

    plotter.add_values(batch,loss_train=loss.data, acc_train=accuracy.data)

    model.zerograds()
    loss.backward()

    optimizer.lr = learning_rate * (1 - batch / max_batches) ** lr_decay_power # Polynomial decay learning rate
    optimizer.update()


    # save model
    if (batch) % 2 == 0:
        model_file = "%s/%s.model" % (backup_path, batch+1)
        print("saving model to %s" % (model_file))
        serializers.save_hdf5(model_file, model)
        serializers.save_hdf5(backup_file, model)

        # read image and process on it
        print("final results", result_compare)
        img1 = cv2.imread("./items/sprite.png")
        img2 = cv2.imread("./items/coca-zero.png")

        img1 = cv2.resize(img1, (input_height, input_width))
        img2 = cv2.resize(img2, (input_height, input_width))

        cv2.imshow('test feed', img2)
        cv2.waitKey(500)
        #print('shape', img2.shape[:2])

        img1 = img1[:,:,:3]
        img1 = np.asarray(img1, dtype=np.float32) / 255.0
        img1 = img1.transpose(2, 0, 1)
        img2 = img2[:,:,:3]
        img2 = np.asarray(img2, dtype=np.float32) / 255.0
        img2 = img2.transpose(2, 0, 1)
        # load model
        model.predictor.train = False

        # forward
        x = []

        x.append(np.array(img1))
        x.append(np.array((img2)))
        x = np.asarray(x, dtype=np.float32)
        x = Variable(x)

        if hasattr(cuda, "cupy"):
            cuda.get_device(0).use()
            model.to_gpu()
            x.to_gpu()

        one_hot_t = []
        first = [1,0,0,0]
        secund = [0,1,0,0]
        one_hot_t.append(np.array(first))
        one_hot_t.append(np.array(secund))
        one_hot_t = np.array(one_hot_t, dtype=np.float32)
        one_hot_t = Variable(one_hot_t)
        one_hot_t.to_gpu(0)

        y, loss, accuracy = model(x, one_hot_t)
        #if hasattr(cuda, "cupy"):
           # y = y.get()
        print("final prediction", y, "accuracy", accuracy)
        #predicted_order = np.argsort(-y.flatten())
        #for index in predicted_order:
            #cls = labels[index]
            #prob = y.flatten()[index] * 100
            #print("%16s : %.2f%%" % (cls, prob))
        model.zerograds()
        model.cleargrads()

print("saving model to %s/darknet19_final.model" % (backup_path))
serializers.save_hdf5("%s/darknet19_final.model" % (backup_path), model)
