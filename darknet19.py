import numpy as np
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils, initializers
from chainer import Link, Chain, ChainList
import chainer.links as L
import chainer.functions as F
from lib.utils import *
from lib.functions import *
import time

class Darknet19(Chain):

    """
    Darknet19
    - It takes (224, 224, 3) or (448, 448, 4) sized image as input
    """

    def __init__(self):
        initializer = initializers.HeNormal()
        super(Darknet19, self).__init__(
            ##### common layers for both pretrained layers and yolov2 #####

            conv1  = L.Convolution2D(3, 32, ksize=3, stride=1, pad=1, nobias=True),
            bn1    = L.BatchNormalization(32, use_beta=False),
            bias1  = L.Bias(shape=(32,)),
            conv2  = L.Convolution2D(32, 64, ksize=3, stride=1, pad=1, nobias=True),
            bn2    = L.BatchNormalization(64, use_beta=False),
            bias2  = L.Bias(shape=(64,)),
            conv3  = L.Convolution2D(64, 128, ksize=3, stride=1, pad=1, nobias=True),
            bn3    = L.BatchNormalization(128, use_beta=False),
            bias3  = L.Bias(shape=(128,)),
            conv4  = L.Convolution2D(128, 64, ksize=1, stride=1, pad=0, nobias=True),
            bn4    = L.BatchNormalization(64, use_beta=False),
            bias4  = L.Bias(shape=(64,)),
            conv5  = L.Convolution2D(64, 128, ksize=3, stride=1, pad=1, nobias=True),
            bn5    = L.BatchNormalization(128, use_beta=False),
            bias5  = L.Bias(shape=(128,)),
            conv6  = L.Convolution2D(128, 256, ksize=3, stride=1, pad=1, nobias=True),
            bn6    = L.BatchNormalization(256, use_beta=False),
            bias6  = L.Bias(shape=(256,)),
            conv7  = L.Convolution2D(256, 128, ksize=1, stride=1, pad=0, nobias=True),
            bn7    = L.BatchNormalization(128, use_beta=False),
            bias7  = L.Bias(shape=(128,)),
            conv8  = L.Convolution2D(128, 256, ksize=3, stride=1, pad=1, nobias=True),
            bn8    = L.BatchNormalization(256, use_beta=False),
            bias8  = L.Bias(shape=(256,)),
            conv9  = L.Convolution2D(256, 512, ksize=3, stride=1, pad=1, nobias=True),
            bn9    = L.BatchNormalization(512, use_beta=False),
            bias9  = L.Bias(shape=(512,)),
            conv10 = L.Convolution2D(512, 256, ksize=1, stride=1, pad=0, nobias=True),
            bn10   = L.BatchNormalization(256, use_beta=False),
            bias10 = L.Bias(shape=(256,)),
            conv11 = L.Convolution2D(256, 512, ksize=3, stride=1, pad=1, nobias=True),
            bn11   = L.BatchNormalization(512, use_beta=False),
            bias11 = L.Bias(shape=(512,)),
            conv12 = L.Convolution2D(512, 256, ksize=1, stride=1, pad=0, nobias=True),
            bn12   = L.BatchNormalization(256, use_beta=False),
            bias12 = L.Bias(shape=(256,)),
            conv13 = L.Convolution2D(256, 512, ksize=3, stride=1, pad=1, nobias=True),
            bn13   = L.BatchNormalization(512, use_beta=False),
            bias13 = L.Bias(shape=(512,)),
            conv14 = L.Convolution2D(512, 1024, ksize=3, stride=1, pad=1, nobias=True),
            bn14   = L.BatchNormalization(1024, use_beta=False),
            bias14 = L.Bias(shape=(1024,)),
            conv15 = L.Convolution2D(1024, 512, ksize=1, stride=1, pad=0, nobias=True),
            bn15   = L.BatchNormalization(512, use_beta=False),
            bias15 = L.Bias(shape=(512,)),
            conv16 = L.Convolution2D(512, 1024, ksize=3, stride=1, pad=1, nobias=True),
            bn16   = L.BatchNormalization(1024, use_beta=False),
            bias16 = L.Bias(shape=(1024,)),
            conv17 = L.Convolution2D(1024, 512, ksize=1, stride=1, pad=0, nobias=True),
            bn17   = L.BatchNormalization(512, use_beta=False),
            bias17 = L.Bias(shape=(512,)),
            conv18 = L.Convolution2D(512, 1024, ksize=3, stride=1, pad=1, nobias=True),
            bn18   = L.BatchNormalization(1024, use_beta=False),
            bias18 = L.Bias(shape=(1024,)),

            ###### new layer, be careful of output nb to change nb of item had changed
            conv19 = L.Convolution2D(1024, 4, ksize=1, stride=1, pad=0),
        )
        self.train = False
        self.finetune = False

    def __call__(self, x):
        batch_size = x.data.shape[0]
        #print('shape', x.data.shape)
        ##### common layer
        h = F.leaky_relu(self.bias1(self.bn1(self.conv1(x),  finetune=self.finetune)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bias2(self.bn2(self.conv2(h),  finetune=self.finetune)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        #print('h data first',h[0][0])
        h = F.leaky_relu(self.bias3(self.bn3(self.conv3(h),  finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias4(self.bn4(self.conv4(h),  finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias5(self.bn5(self.conv5(h),  finetune=self.finetune)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bias6(self.bn6(self.conv6(h),  finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias7(self.bn7(self.conv7(h),  finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias8(self.bn8(self.conv8(h),  finetune=self.finetune)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)

        h = F.leaky_relu(self.bias9(self.bn9(self.conv9(h),  finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias10(self.bn10(self.conv10(h),  finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias11(self.bn11(self.conv11(h),  finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias12(self.bn12(self.conv12(h),  finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias13(self.bn13(self.conv13(h),  finetune=self.finetune)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        #print('h data check',h[0][0])
        h = F.leaky_relu(self.bias14(self.bn14(self.conv14(h),  finetune=self.finetune)), slope=1)
        #print('h data check too ',h[0][0])
        h = F.leaky_relu(self.bias15(self.bn15(self.conv15(h),  finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias16(self.bn16(self.conv16(h),  finetune=self.finetune)), slope=0.1)

        h = F.leaky_relu(self.bias17(self.bn17(self.conv17(h),  finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias18(self.bn18(self.conv18(h),  finetune=self.finetune)), slope=0.1)

        ###### new layer
        h = self.conv19(h)
        h = F.average_pooling_2d(h, h.data.shape[-1], stride=1, pad=0)
        # reshape
        #print('y shape', h.data.shape)
        y = F.reshape(h, (batch_size, -1))
        return y

class Darknet19Predictor(Chain):
    def __init__(self, predictor):
        super(Darknet19Predictor, self).__init__(predictor=predictor)

    def __call__(self, x, t):
        y = self.predictor(x)

        #test = self.predict(x).data
        #predicted_order = np.argsort(-test.flatten())
        #for index in predicted_order:
         #   prob = test.flatten()[index] * 100
          #  print("clase: %.2f%%" % ( prob))
        #print("results of the operation",  F.softmax(y).data)

        if t.ndim == 2: # use squared error when label is one hot label
            y = F.softmax(y)
            #print('loss debug, y', y[0])
            #print('shapes', y.shape, t.shape)
            loss = F.mean_squared_error(y, t)
            #loss = sum_of_squared_error(y, t)
            #print('loss value in CNN', y, t)
            accuracy = F.accuracy(y, t.data.argmax(axis=1).astype(np.int32))
        else: # use softmax cross entropy when label is normal label
            #print("cross entropy debug", y, t)
            loss = F.softmax_cross_entropy(y, t)
            accuracy = F.accuracy(y, t)

        return y, loss, accuracy

    def predict(self, x):
        y = self.predictor(x)
        return F.softmax(y)

##################################################

class YOLOv2(Chain):

    """
    YOLOv2
    - It takes (416, 416, 3) sized image as input
    """

    def __init__(self, n_classes, n_boxes):
        initialW = initializers.HeNormal()
        super(YOLOv2, self).__init__(
            conv1  = L.Convolution2D(3, 32, ksize=3, stride=1, pad=1, nobias=True, initialW=initialW),
            bn1    = L.BatchNormalization(32, use_beta=False, eps=2e-5),
            bias1  = L.Bias(shape=(32,)),
            conv2  = L.Convolution2D(32, 64, ksize=3, stride=1, pad=1, nobias=True, initialW=initialW),
            bn2    = L.BatchNormalization(64, use_beta=False, eps=2e-5),
            bias2  = L.Bias(shape=(64,)),
            conv3  = L.Convolution2D(64, 128, ksize=3, stride=1, pad=1, nobias=True, initialW=initialW),
            bn3    = L.BatchNormalization(128, use_beta=False, eps=2e-5),
            bias3  = L.Bias(shape=(128,)),
            conv4  = L.Convolution2D(128, 64, ksize=1, stride=1, pad=0, nobias=True, initialW=initialW),
            bn4    = L.BatchNormalization(64, use_beta=False, eps=2e-5),
            bias4  = L.Bias(shape=(64,)),
            conv5  = L.Convolution2D(64, 128, ksize=3, stride=1, pad=1, nobias=True, initialW=initialW),
            bn5    = L.BatchNormalization(128, use_beta=False, eps=2e-5),
            bias5  = L.Bias(shape=(128,)),
            conv6  = L.Convolution2D(128, 256, ksize=3, stride=1, pad=1, nobias=True, initialW=initialW),
            bn6    = L.BatchNormalization(256, use_beta=False, eps=2e-5),
            bias6  = L.Bias(shape=(256,)),
            conv7  = L.Convolution2D(256, 128, ksize=1, stride=1, pad=0, nobias=True, initialW=initialW),
            bn7    = L.BatchNormalization(128, use_beta=False, eps=2e-5),
            bias7  = L.Bias(shape=(128,)),
            conv8  = L.Convolution2D(128, 256, ksize=3, stride=1, pad=1, nobias=True, initialW=initialW),
            bn8    = L.BatchNormalization(256, use_beta=False, eps=2e-5),
            bias8  = L.Bias(shape=(256,)),
            conv9  = L.Convolution2D(256, 512, ksize=3, stride=1, pad=1, nobias=True, initialW=initialW),
            bn9    = L.BatchNormalization(512, use_beta=False, eps=2e-5),
            bias9  = L.Bias(shape=(512,)),
            conv10 = L.Convolution2D(512, 256, ksize=1, stride=1, pad=0, nobias=True, initialW=initialW),
            bn10   = L.BatchNormalization(256, use_beta=False, eps=2e-5),
            bias10 = L.Bias(shape=(256,)),
            conv11 = L.Convolution2D(256, 512, ksize=3, stride=1, pad=1, nobias=True, initialW=initialW),
            bn11   = L.BatchNormalization(512, use_beta=False, eps=2e-5),
            bias11 = L.Bias(shape=(512,)),
            conv12 = L.Convolution2D(512, 256, ksize=1, stride=1, pad=0, nobias=True, initialW=initialW),
            bn12   = L.BatchNormalization(256, use_beta=False, eps=2e-5),
            bias12 = L.Bias(shape=(256,)),
            conv13 = L.Convolution2D(256, 512, ksize=3, stride=1, pad=1, nobias=True, initialW=initialW),
            bn13   = L.BatchNormalization(512, use_beta=False, eps=2e-5),
            bias13 = L.Bias(shape=(512,)),
            conv14 = L.Convolution2D(512, 1024, ksize=3, stride=1, pad=1, nobias=True, initialW=initialW),
            bn14   = L.BatchNormalization(1024, use_beta=False, eps=2e-5),
            bias14 = L.Bias(shape=(1024,)),
            conv15 = L.Convolution2D(1024, 512, ksize=1, stride=1, pad=0, nobias=True, initialW=initialW),
            bn15   = L.BatchNormalization(512, use_beta=False, eps=2e-5),
            bias15 = L.Bias(shape=(512,)),
            conv16 = L.Convolution2D(512, 1024, ksize=3, stride=1, pad=1, nobias=True, initialW=initialW),
            bn16   = L.BatchNormalization(1024, use_beta=False, eps=2e-5),
            bias16 = L.Bias(shape=(1024,)),
            conv17 = L.Convolution2D(1024, 512, ksize=1, stride=1, pad=0, nobias=True, initialW=initialW),
            bn17   = L.BatchNormalization(512, use_beta=False, eps=2e-5),
            bias17 = L.Bias(shape=(512,)),
            conv18 = L.Convolution2D(512, 1024, ksize=3, stride=1, pad=1, nobias=True, initialW=initialW),
            bn18   = L.BatchNormalization(1024, use_beta=False, eps=2e-5),
            bias18 = L.Bias(shape=(1024,)),

            conv19 = L.Convolution2D(1024, 1024, ksize=3, stride=1, pad=1, nobias=True, initialW=initialW),
            bn19   = L.BatchNormalization(1024, use_beta=False),
            bias19 = L.Bias(shape=(1024,)),
            conv20 = L.Convolution2D(1024, 1024, ksize=3, stride=1, pad=1, nobias=True, initialW=initialW),
            bn20   = L.BatchNormalization(1024, use_beta=False),
            bias20 = L.Bias(shape=(1024,)),
            conv21 = L.Convolution2D(512, 64, ksize=1, stride=1, pad=0, nobias=True, initialW=initialW),
            bn21   = L.BatchNormalization(64, use_beta=False),
            bias21 = L.Bias(shape=(64,)),
            conv22 = L.Convolution2D(1024 + 64 * 4, 1024, ksize=3, stride=1, pad=1, nobias=True, initialW=initialW),
            bn22   = L.BatchNormalization(1024, use_beta=False),
            bias22 = L.Bias(shape=(1024,)),
            conv23 = L.Convolution2D(1024, n_boxes * (5 + n_classes), ksize=1,
                stride=1, pad=0, nobias=True,
                initialW=initializers.Constant(0)),
            bias23 = L.Bias(shape=(n_boxes * (5 + n_classes),)),
        )
        self.finetune = False
        self.n_boxes = n_boxes
        self.n_classes = n_classes

    def __call__(self, x):
        h = F.leaky_relu(self.bias1(self.bn1(self.conv1(x), finetune=self.finetune)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bias2(self.bn2(self.conv2(h), finetune=self.finetune)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bias3(self.bn3(self.conv3(h), finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias4(self.bn4(self.conv4(h), finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias5(self.bn5(self.conv5(h), finetune=self.finetune)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bias6(self.bn6(self.conv6(h), finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias7(self.bn7(self.conv7(h), finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias8(self.bn8(self.conv8(h), finetune=self.finetune)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bias9(self.bn9(self.conv9(h), finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias10(self.bn10(self.conv10(h), finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias11(self.bn11(self.conv11(h), finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias12(self.bn12(self.conv12(h), finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias13(self.bn13(self.conv13(h), finetune=self.finetune)), slope=0.1)
        high_resolution_feature = h
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bias14(self.bn14(self.conv14(h), finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias15(self.bn15(self.conv15(h), finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias16(self.bn16(self.conv16(h), finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias17(self.bn17(self.conv17(h), finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias18(self.bn18(self.conv18(h), finetune=self.finetune)), slope=0.1)

        h = F.leaky_relu(self.bias19(self.bn19(self.conv19(h), finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias20(self.bn20(self.conv20(h), finetune=self.finetune)), slope=0.1)

        h2 = high_resolution_feature
        h2 = F.leaky_relu(self.bias21(self.bn21(self.conv21(h2), finetune=self.finetune)), slope=0.1)
        h2 = reorg(h2)

        h = F.concat((h2, h), axis=1)
        h = F.leaky_relu(self.bias22(self.bn22(self.conv22(h), finetune=self.finetune)), slope=0.1)

        h = self.bias23(self.conv23(h))

        return h

class YOLOv2Predictor(Chain):
    def __init__(self, predictor):
        super(YOLOv2Predictor, self).__init__(predictor=predictor)
        self.anchors = [[0.57273, 0.677385], [1.87446, 2.06253], [3.33843, 5.47434], [7.88282, 3.52778], [9.77052, 9.16828]]
        self.thresh = 0.7
        self.ignore_thresh = 0.1
        self.seen = 0
        self.unstable_seen = 15000

    def __call__(self, input_x, t, ignore_t):
        if isinstance(input_x, chainer.Variable):
            device = cuda.get_device(input_x.data)
        else:
            device = cuda.get_device(input_x)
        xp = self.predictor.xp
        with device:
            output = self.predictor(input_x)
            batch_size, _, grid_h, grid_w = output.shape
            self.seen += batch_size
            x, y, w, h, conf, prob = F.split_axis(F.reshape(output, (batch_size, self.predictor.n_boxes, self.predictor.n_classes+5, grid_h, grid_w)), (1, 2, 3, 4, 5), axis=2)
            x = F.sigmoid(x)
            y = F.sigmoid(y)
            conf = F.sigmoid(conf)
            prob = F.transpose(prob, (0, 2, 1, 3, 4))
            prob = F.softmax(prob)


            # training labels
            tw = np.zeros(w.shape, dtype=np.float32)
            th = np.zeros(h.shape, dtype=np.float32)
            tx = np.tile(0.5, x.shape).astype(np.float32)
            ty = np.tile(0.5, y.shape).astype(np.float32)

            # set low learning rate for bounding boxes that have no object
            if self.seen < self.unstable_seen:
                box_learning_scale = np.tile(0.1, x.shape).astype(np.float32)
            else:
                box_learning_scale = np.tile(0, x.shape).astype(np.float32)

            tconf = np.zeros(conf.shape, dtype=np.float32)
            conf_learning_scale = np.tile(0.1, conf.shape).astype(np.float32)

            tprob = prob.data.copy()

            x_shift = np.broadcast_to(np.arange(grid_w, dtype=np.float32), x.shape[1:])
            y_shift = np.broadcast_to(np.arange(grid_h, dtype=np.float32).reshape(grid_h, 1), y.shape[1:])
            w_anchor = np.broadcast_to(np.reshape(np.array(self.anchors, dtype=np.float32)[:, 0], (self.predictor.n_boxes, 1, 1, 1)), w.shape[1:])
            h_anchor = np.broadcast_to(np.reshape(np.array(self.anchors, dtype=np.float32)[:, 1], (self.predictor.n_boxes, 1, 1, 1)), h.shape[1:])
            x_data = cuda.to_cpu(x.data)
            y_data = cuda.to_cpu(y.data)
            w_data = cuda.to_cpu(w.data)
            h_data = cuda.to_cpu(h.data)
            best_ious = []
            for batch in range(batch_size):
                n_truth_boxes = len(t[batch])
                box_x = (x_data[batch] + x_shift) / grid_w
                box_y = (y_data[batch] + y_shift) / grid_h
                box_w = np.exp(w_data[batch]) * w_anchor / grid_w
                box_h = np.exp(h_data[batch]) * h_anchor / grid_h

                ious = []
                for truth_index in range(n_truth_boxes):
                    truth_box_x = np.broadcast_to(np.array(t[batch][truth_index]["x"], dtype=np.float32), box_x.shape)
                    truth_box_y = np.broadcast_to(np.array(t[batch][truth_index]["y"], dtype=np.float32), box_y.shape)
                    truth_box_w = np.broadcast_to(np.array(t[batch][truth_index]["w"], dtype=np.float32), box_w.shape)
                    truth_box_h = np.broadcast_to(np.array(t[batch][truth_index]["h"], dtype=np.float32), box_h.shape)
                    ious.append(multi_box_iou(Box(box_x, box_y, box_w, box_h), Box(truth_box_x, truth_box_y, truth_box_w, truth_box_h)))
                if len(ious) > 0:
                    ious = np.asarray(ious)
                    best_ious.append(np.max(ious, axis=0))
                else:
                    best_ious.append(np.zeros_like(x_data[0]))
            best_ious = np.array(best_ious)

            # keep confidence of anchor that has more confidence than threshold
            tconf[best_ious > self.thresh] = conf.data.get()[best_ious > self.thresh]
            conf_learning_scale[best_ious > self.thresh] = 0

            # ignored regions are not considered either positive or negative

            best_ious = []
            for batch in range(batch_size):
                n_truth_boxes = len(ignore_t[batch])
                box_x = (x_data[batch] + x_shift) / grid_w
                box_y = (y_data[batch] + y_shift) / grid_h
                box_w = np.exp(w_data[batch]) * w_anchor / grid_w
                box_h = np.exp(h_data[batch]) * h_anchor / grid_h

                ious = []
                for truth_index in range(n_truth_boxes):
                    truth_box_x = np.broadcast_to(np.array(ignore_t[batch][truth_index]["x"], dtype=np.float32), box_x.shape)
                    truth_box_y = np.broadcast_to(np.array(ignore_t[batch][truth_index]["y"], dtype=np.float32), box_y.shape)
                    truth_box_w = np.broadcast_to(np.array(ignore_t[batch][truth_index]["w"], dtype=np.float32), box_w.shape)
                    truth_box_h = np.broadcast_to(np.array(ignore_t[batch][truth_index]["h"], dtype=np.float32), box_h.shape)
                    ious.append(multi_box_iou(Box(box_x, box_y, box_w, box_h), Box(truth_box_x, truth_box_y, truth_box_w, truth_box_h)))
                if len(ious) > 0:
                    ious = np.asarray(ious)
                    best_ious.append(np.max(ious, axis=0))
                else:
                    best_ious.append(np.zeros_like(x_data[0]))
            best_ious = np.array(best_ious)

            # do not update confidence for ignored regions
            tconf[best_ious > self.ignore_thresh] = conf.data.get()[best_ious > self.ignore_thresh]
            conf_learning_scale[best_ious > self.ignore_thresh] = 0

            # adjust x, y, w, h, conf, prob of anchor boxes that have objects
            abs_anchors = self.anchors / np.array([grid_w, grid_h])
            for batch in range(batch_size):
                for truth_box in t[batch]:
                    truth_w = int(float(truth_box["x"]) * grid_w)
                    truth_h = int(float(truth_box["y"]) * grid_h)
                    truth_n = 0
                    best_iou = 0.0
                    for anchor_index, abs_anchor in enumerate(abs_anchors):
                        iou = box_iou(Box(0, 0, float(truth_box["w"]), float(truth_box["h"])), Box(0, 0, abs_anchor[0], abs_anchor[1]))
                        if best_iou < iou:
                            best_iou = iou
                            truth_n = anchor_index

                    box_learning_scale[batch, truth_n, :, truth_h, truth_w] = 1.0
                    tx[batch, truth_n, :, truth_h, truth_w] = float(truth_box["x"]) * grid_w - truth_w
                    ty[batch, truth_n, :, truth_h, truth_w] = float(truth_box["y"]) * grid_h - truth_h
                    tw[batch, truth_n, :, truth_h, truth_w] = np.log(float(truth_box["w"]) / abs_anchors[truth_n][0])
                    th[batch, truth_n, :, truth_h, truth_w] = np.log(float(truth_box["h"]) / abs_anchors[truth_n][1])
                    tprob[batch, :, truth_n, truth_h, truth_w] = 0
                    tprob[batch, int(truth_box["label"]), truth_n, truth_h, truth_w] = 1

                    full_truth_box = Box(float(truth_box["x"]), float(truth_box["y"]), float(truth_box["w"]), float(truth_box["h"]))
                    predicted_box = Box(
                        (x[batch][truth_n][0][truth_h][truth_w].data.get() + truth_w) / grid_w,
                        (y[batch][truth_n][0][truth_h][truth_w].data.get() + truth_h) / grid_h,
                        np.exp(w[batch][truth_n][0][truth_h][truth_w].data.get()) * abs_anchors[truth_n][0],
                        np.exp(h[batch][truth_n][0][truth_h][truth_w].data.get()) * abs_anchors[truth_n][1]
                    )
                    predicted_iou = box_iou(full_truth_box, predicted_box)
                    tconf[batch, truth_n, :, truth_h, truth_w] = predicted_iou
                    conf_learning_scale[batch, truth_n, :, truth_h, truth_w] = 10.0

            tx = cuda.to_gpu(tx)
            ty = cuda.to_gpu(ty)
            tw = cuda.to_gpu(tw)
            th = cuda.to_gpu(th)
            tconf = cuda.to_gpu(tconf)
            tprob = cuda.to_gpu(tprob)

            box_learning_scale = cuda.to_gpu(box_learning_scale)
            conf_learning_scale = cuda.to_gpu(conf_learning_scale)

            x_loss = F.sum((tx - x) ** 2 * box_learning_scale) / 2
            y_loss = F.sum((ty - y) ** 2 * box_learning_scale) / 2
            w_loss = F.sum((tw - w) ** 2 * box_learning_scale) / 2
            h_loss = F.sum((th - h) ** 2 * box_learning_scale) / 2
            c_loss = F.sum((tconf - conf) ** 2 * conf_learning_scale) / 2
            p_loss = F.sum((tprob - prob) ** 2) / 2
            return x_loss, y_loss, w_loss, h_loss, c_loss, p_loss

    def init_anchor(self, anchors):
        self.anchors = anchors

    def predict(self, input_x):
        if isinstance(input_x, chainer.Variable):
            device = cuda.get_device(input_x.data)
        else:
            device = cuda.get_device(input_x)
        xp = self.predictor.xp
        with device:
            output = self.predictor(input_x)
            batch_size, input_channel, input_h, input_w = input_x.shape
            batch_size, _, grid_h, grid_w = output.shape
            x, y, w, h, conf, prob = F.split_axis(F.reshape(output, (batch_size, self.predictor.n_boxes, self.predictor.n_classes+5, grid_h, grid_w)), (1, 2, 3, 4, 5), axis=2)
            x = F.sigmoid(x)
            y = F.sigmoid(y)
            conf = F.sigmoid(conf)
            prob = F.transpose(prob, (0, 2, 1, 3, 4))
            prob = F.softmax(prob)
            prob = F.transpose(prob, (0, 2, 1, 3, 4))


            # convert coordinates to those on the image
            x_shift = xp.asarray(np.broadcast_to(np.arange(grid_w, dtype=np.float32), x.shape))
            y_shift = xp.asarray(np.broadcast_to(np.arange(grid_h, dtype=np.float32).reshape(grid_h, 1), y.shape))
            w_anchor = xp.asarray(np.broadcast_to(np.reshape(np.array(self.anchors, dtype=np.float32)[:, 0], (self.predictor.n_boxes, 1, 1, 1)), w.shape))
            h_anchor = xp.asarray(np.broadcast_to(np.reshape(np.array(self.anchors, dtype=np.float32)[:, 1], (self.predictor.n_boxes, 1, 1, 1)), h.shape))
            box_x = (x + x_shift) / grid_w
            box_y = (y + y_shift) / grid_h
            box_w = F.exp(w) * w_anchor / grid_w
            box_h = F.exp(h) * h_anchor / grid_h

return box_x, box_y, box_w, box_h, conf, prob
