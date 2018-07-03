import cv2
import os
import glob
import numpy as np
from PIL import Image
from lib.utils import *
from itertools import product
from chainer import datasets
import re


def sorted_nicely( l ):
    """ Sorts the given iterable in the way that is expected.

    Required arguments:
    l -- The iterable to be sorted.

    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)

def resize(img, input_width, input_height):
    img = Image.fromarray(img.transpose(1, 2, 0))
    img = img.resize((input_width, input_height), Image.BICUBIC)
    return np.asarray(img).transpose(2, 0, 1)

#def transform(inputs):
#    ground_truths = []
#    img, label = inputs
#    img = img[:3, ...]
#    img = resize(img.astype(np.uint8))
#   # img = img - mean[:, None, None]
#    img = img.astype(np.float32)
#    # ランダムに左右反転
#    if np.random.rand() > 0.5:
#        img = img[..., ::-1]

#    #we have to find how many truth box is there
#    nb_truth_box = int((len(label)-1)/ 5)
#    print("original label", label)
#    #hot_label = label.copy
#    for i in range(nb_truth_box):
#        one_hot_label = int(label[len(label)-1])*[0]
#        one_hot_label[int(label[4+(5*i)])] = 1
#        hot_label = label[i*5:((i*5)+5)]
#        hot_label = np.concatenate((hot_label,one_hot_label))
#        #hot_label = np.append(hot_label, one_hot_label)
#        ground_truths = np.concatenate((ground_truths,hot_label))
#    ground_truths= np.array(ground_truths)
#    print(" debug in transform ", ground_truths )

#    return img, ground_truths


def transform(inputs):
    train_sizes = [320, 352, 384, 416, 448]
    #input_width = input_height = train_sizes[np.random.randint(len(train_sizes))]
    input_width = input_height = 448
    #print("image size for the batch", input_width, input_height)
    ground_truths = []
    img, label = inputs
    img = img[:3, ...]
    img = resize(img.astype(np.uint8), input_width, input_height)
   # img = img - mean[:, None, None]
    img = img.astype(np.float32)
    # ランダムに左右反転
#    if np.random.rand() > 0.5:
#        img = img[..., ::-1]

    #we have to find how many truth box is there
    nb_truth_box = int((len(label)-1)/ 5)
    #print("original label", label)
    #hot_label = label.copy
    for i in range(nb_truth_box):
        one_hot_label = int(label[len(label)-1])*[0]
        one_hot_label[int(label[4+(5*i)])] = 1
        hot_label = label[i*5:((i*5)+5)]
        hot_label = np.concatenate((hot_label,one_hot_label))
        #hot_label = np.append(hot_label, one_hot_label)
        ground_truths.append(hot_label)
    ground_truths= np.array(ground_truths)
#    img = Variable(img)
#    img.to_gpu()
    #print(" debug in transform ", ground_truths )

    return img, ground_truths



class ImageNet_data():
    def __init__(self, img_path, bbox_path, list_path, nb_class):
        self.img_files = glob.glob(img_path + "/*")
        self.bbox_files = glob.glob(bbox_path + "/*")
        self.img_list = []
        self.bbox = []
        self.label_list = []
        self.nb_class = nb_class
        self.train_set_iterator = 0
        intersection = 0
        self.train_ratio=0.7
        self.train_sizes = [320, 352, 384, 416, 448]



        with open(list_path) as f:
            content = f.readlines()
            #print("content", content)

            self.img_names = [img_path+"/"+x.strip()+".JPEG" for x in content]
            self.bbox_names = [bbox_path+"/"+x.strip()+".txt" for x in content]



        self.img_names = sorted_nicely(self.img_names)

        #print(self.img_names[:20], len(self.img_names))
        self.bbox_names = sorted_nicely(self.bbox_names)
        #print(self.bbox_names[:20], len(self.bbox_names))

       # print("Verification of bbox and images number", len(self.bbox_names) == len(self.img_names), len(self.bbox_names), len(self.img_names), len(content) )

        for item_file in self.bbox_names:
            #print(item_file)
            file =  [float(x) for x in open(item_file).read().split()]
            self.bbox.append(file)
           # print (file, "for", item_file)
        for i in range(len(self.bbox)):
            ground_truths = []
            nb_box = int(len(self.bbox[i])/5)
           # print("nb_box",nb_box)

#            dt = np.zeros(6, np.dtype({'names': ['x', 'y', 'w', 'h', 'label'],
#            'formats': ['f8','f8','f8','f8','f8'] } ))
            while (nb_box>=1) :
                nb_box-=1
                #print("nb_box",nb_box)
                ground_truths += [self.bbox[i][1+(nb_box*5)],
                self.bbox[i][2+(nb_box*5)],
                self.bbox[i][3+(nb_box*5)],
                self.bbox[i][4+(nb_box*5)],
                self.bbox[i][0+(nb_box*5)]]
            ground_truths += [self.nb_class]
            #print (ground_truths, "for", self.bbox[i])

            self.label_list.append(ground_truths)



    def train_val_test(self):
        #img_lbl has the size of the dataset, each element is the image and
        img_lbl=list(zip(self.img_names,self.label_list))

#        for i in range(10):
#            print("result of the pairing img, labl", img_lbl[i] )
        base = datasets.LabeledImageDataset(img_lbl, label_dtype= np.dtype('Float64'))
        tbase = datasets.TransformDataset(base, transform)

        image, lbl = tbase.get_example(5)
        image = np.asarray(image, dtype=np.float32) /255.0
#before (3, 448, 448)
#after (448, 448, 3)
        image = np.transpose(image, (1, 2, 0)).copy()

#        print("result of the pairing img, labl", image.shape, lbl  )
#        cv2.imshow("image", image)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
#        print("label", lbl)
        train_size=int(len(tbase)*self.train_ratio)
        train, test = datasets.split_dataset(tbase, train_size)
        return train, test

    def imageNet_yolo(self, n_samples, w_in, h_in):
        x = []
        t = []
        #random = np.random.randint(0, len(self.img_names), n_samples)
        for j in range(self.train_set_iterator, self.train_set_iterator+ n_samples):
            ground_truths = []
            #i = random[j]
            i=j
            img_path = self.img_names[i]
            #print("for img" , img_path)
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            sample_image = cv2.resize(image, (w_in, h_in))
            one_hot_label = np.zeros(self.nb_class)
            one_hot_label[int(self.bbox[i][0])] = 1
            ground_truths.append({
                "x": self.bbox[i][1],
                "y": self.bbox[i][2],
                "w": self.bbox[i][3],
                "h": self.bbox[i][4],
                "label": self.bbox[i][0],
                "one_hot_label": one_hot_label
            })
            box = Box(self.bbox[i][1]*w_in, self.bbox[i][2]*h_in, self.bbox[i][3]*w_in,self.bbox[i][4]*h_in)
            #print(sample_image.shape)
            #sample_image = sample_image[:, :, :3]
            #print("box", box.int_left_top(), box.int_right_bottom(), "for", self.bbox[i][1], self.bbox[i][2], self.bbox[i][3],self.bbox[i][4])
            if j%10==0:
                print("gtruth hot label", ground_truths[0]["one_hot_label"])
            t.append(ground_truths)
#            cv2.rectangle(
#                sample_image,
#                box.int_left_top(), box.int_right_bottom(),
#                (255, 0, 255),
#                3
#            )
#            cv2.imshow(img_path[10:], sample_image)
#            cv2.waitKey(70)
#            cv2.destroyAllWindows()
            #print("bbox", ground_truths)
            sample_image = np.asarray(sample_image, dtype=np.float32) / 255.0
            sample_image = sample_image.transpose(2, 0, 1)
            vec = np.asarray(sample_image).astype(np.float32)
            x.append(vec)

        if (self.train_set_iterator + n_samples <= len(self.img_names)- n_samples):
            self.train_set_iterator += n_samples
        elif (len(self.img_names)- (self.train_set_iterator + n_samples))>0 :
            self.train_set_iterator = len(self.img_names)-n_samples
        else:
            self.train_set_iterator = 0
            print("One epoch done, we compute the mAp")



        # load model
        #model.predictor.train = False

        # forward
        #x.append(vec2)
        x = np.array(x)
        #print(t)
        return x,t
