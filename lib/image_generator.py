import cv2
import os
import glob
import numpy as np
from PIL import Image
from lib.utils import *
from itertools import product
from chainer import datasets
import re

# src_imageの背景画像に対して、overlay_imageのalpha画像を貼り付ける。pos_xとpos_yは貼り付け時の左上の座標
def overlay(src_image, overlay_image, pos_x, pos_y):
    # オーバレイ画像のサイズを取得
    ol_height, ol_width = overlay_image.shape[:2]

    # OpenCVの画像データをPILに変換
    # BGRAからRGBAへ変換
    src_image_RGBA = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
    overlay_image_RGBA = cv2.cvtColor(overlay_image, cv2.COLOR_BGRA2RGBA)

    #　PILに変換
    src_image_PIL=Image.fromarray(src_image_RGBA)
    overlay_image_PIL=Image.fromarray(overlay_image_RGBA)

    # 合成のため、RGBAモードに変更
    src_image_PIL = src_image_PIL.convert('RGBA')
    overlay_image_PIL = overlay_image_PIL.convert('RGBA')

    # 同じ大きさの透過キャンパスを用意
    tmp = Image.new('RGBA', src_image_PIL.size, (255, 255,255, 0))
    # 用意したキャンパスに上書き
    tmp.paste(overlay_image_PIL, (pos_x, pos_y), overlay_image_PIL)
    # オリジナルとキャンパスを合成して保存
    result = Image.alpha_composite(src_image_PIL, tmp)

    return  cv2.cvtColor(np.asarray(result), cv2.COLOR_RGBA2BGRA)

# 画像周辺のパディングを削除
def delete_pad(image): 
    orig_h, orig_w = image.shape[:2]
    mask = np.argwhere(image[:, :, 3] > 128) # alphaチャンネルの条件、!= 0 や == 255に調整できる
    (min_y, min_x) = (max(min(mask[:, 0])-1, 0), max(min(mask[:, 1])-1, 0))
    (max_y, max_x) = (min(max(mask[:, 0])+1, orig_h), min(max(mask[:, 1])+1, orig_w))
    return image[min_y:max_y, min_x:max_x]

# 画像を指定した角度だけ回転させる
def rotate_image(image, angle):
    orig_h, orig_w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((orig_h/2, orig_w/2), angle, 1)
    return cv2.warpAffine(image, matrix, (orig_h, orig_w))

# 画像をスケーリングする
def scale_image(image, scale):
    orig_h, orig_w = image.shape[:2]
    return cv2.resize(image, (int(orig_w*scale), int(orig_h*scale)))

# we take off h to the original height and w to the original width, the smaller h and w are the less transformed
def random_sampling(image, h, w): 
    orig_h, orig_w = image.shape[:2]
    y = np.random.randint(orig_h-h+1)
    x = np.random.randint(orig_w-w+1)
    return image[y:y+h, x:x+w]

# 画像をランダムに回転、スケールしてから返す
def random_rotate_scale_image(image, min_scale, max_scale, rand_angle):
    image = rotate_image(image, np.random.randint(rand_angle*2)-rand_angle)
    image = scale_image(image, min_scale + np.random.rand() * (max_scale-min_scale)) # 1 ~ 3倍
    return delete_pad(image)

# overlay_imageを、src_imageのランダムな場所に合成して、そこのground_truthを返す。
def random_overlay_image(src_image, overlay_image, minimum_crop):
    src_h, src_w = src_image.shape[:2]
    overlay_h, overlay_w = overlay_image.shape[:2]
    shift_item_h, shift_item_w = overlay_h * (1-minimum_crop), overlay_w * (1-minimum_crop)
    scale_item_h, scale_item_w = overlay_h * (minimum_crop*2-1), overlay_w * (minimum_crop*2-1)

    a = src_h-scale_item_h
    b = src_w-scale_item_w
    if a <= 1:
      a = 100
    if b <= 1:
      b = 100
    c = int(np.random.randint(a) - shift_item_h)
    d = int(np.random.randint(b) - shift_item_w)
    if c < 0:
      c = int(np.random.randint(a))
    if d < 0:
      d = int(np.random.randint(a))

    y = (int(c))
    x = (int(d))

    image = overlay(src_image, overlay_image, x, y)
    bbox = ((np.maximum(x, 0), np.maximum(y, 0)), (np.minimum(x+overlay_w, src_w-1), np.minimum(y+overlay_h, src_h-1)))

    return image, bbox


# 4点座標のbboxをyoloフォーマットに変換
def yolo_format_bbox(image, bbox):
    orig_h, orig_w = image.shape[:2]
    center_x = (bbox[1][0] + bbox[0][0]) / 2 / orig_w
    center_y = (bbox[1][1] + bbox[0][1]) / 2 / orig_h
    w = (bbox[1][0] - bbox[0][0]) / orig_w
    h = (bbox[1][1] - bbox[0][1]) / orig_h
    return(center_x, center_y, w, h)

def maximum_iou(box, boxes):
    max_iou = 0
    for src_box in boxes:
        iou = box_iou(box, src_box)
        if iou > max_iou:
            max_iou = iou
    return max_iou

class ImageGenerator():
    def __init__(self, item_path, background_path):
        self.bg_files = glob.glob(background_path + "/*")
        self.item_files = glob.glob(item_path + "/*")
        self.items = []
        self.labels = []
        self.bgs = []
        for item_file in self.item_files:
            image = cv2.imread(item_file, cv2.IMREAD_UNCHANGED)
            center = np.maximum(image.shape[0], image.shape[1])
            pixels = np.zeros((center*2, center*2, image.shape[2]))
            y = int(center - image.shape[0]/2)
            x = int(center - image.shape[1]/2)
            pixels[y:y+image.shape[0], x:x+image.shape[1], :] = image
            self.items.append(pixels.astype(np.uint8))
            self.labels.append(item_file.split("/")[-1].split(".")[0])
            #print( item_file.split("/")[-1].split(".")[0])
            #cv2.imshow("w", image)
            #cv2.waitKey(0)

        for bg_file in self.bg_files:
            self.bgs.append(cv2.imread(bg_file))

    def generate_random_animation(self, loop, bg_index, crop_width, crop_height, min_item_scale, max_item_scale):
        frames = []
        sampled_background = random_sampling(self.bgs[bg_index], crop_height, crop_width)
        bg_height, bg_width, _ = sampled_background.shape
        for i in range(loop):
            #class_id = np.random.randint(len(self.labels))
            class_id = i % len(self.labels)
            item = self.items[class_id]
            item = scale_image(item, min_item_scale + np.random.rand() * (max_item_scale-min_item_scale))
            orig_item = item
            item_height, item_width, _ = item.shape
            edges = [-item_width, -item_height, bg_width, bg_height]
            r = np.random.randint(2)
            rand1 = np.random.randint(edges[r+2] - edges[r]) + edges[r]
            center = edges[r] + (edges[r+2] - edges[r]) / 2 
            edges[r+2] = int(center + (center - rand1))
            edges[r] = rand1
            print(edges)

            r = np.random.randint(2)
            start_point = (edges[r*2], edges[r*2+1])
            end_point = (edges[r*2-2], edges[r*2-1])
            w_distance = end_point[0] - start_point[0]
            h_distance = end_point[1] - start_point[1]
            animate_frames = np.random.randint(30) + 50
            angle = np.random.rand() * 10 - 5
            rotate_cnt = 0
            total_angle = 0
            for j in range(animate_frames):
                rotate_cnt += 1
                if rotate_cnt % 10 == 0:
                    angle *= -1
                total_angle += angle
                item = rotate_image(orig_item, total_angle)
                frame = overlay(sampled_background, item, start_point[0] + int(w_distance * j / animate_frames), start_point[1] + int(h_distance * j / animate_frames))
                frames.append(frame[:, :, :3])
        return frames

    def generate_samples(self, n_samples, n_items, crop_width, crop_height, min_item_scale, max_item_scale, rand_angle, minimum_crop, delta_hue, delta_sat_scale, delta_val_scale):
        x = []
        t = []
        for i in range(n_samples):
            bg = self.bgs[np.random.randint(len(self.bgs))]
            sample_image = random_sampling(bg, crop_height, crop_width)
 
            ground_truths = []
            boxes = []
            for j in range(np.random.randint(n_items)+1):
                class_id = np.random.randint(len(self.labels))
                item = self.items[class_id]
                item = random_rotate_scale_image(item, min_item_scale, max_item_scale, rand_angle)

                tmp_image, bbox = random_overlay_image(sample_image, item, minimum_crop)
                yolo_bbox = yolo_format_bbox(tmp_image, bbox)
                box = Box(yolo_bbox[0], yolo_bbox[1], yolo_bbox[2], yolo_bbox[3])
                if maximum_iou(box, boxes) < 0.3:
                    boxes.append(box)
                    one_hot_label = np.zeros(len(self.labels))
                    one_hot_label[class_id] = 1
                    ground_truths.append({     
                        "x": yolo_bbox[0],
                        "y": yolo_bbox[1],
                        "w": yolo_bbox[2],
                        "h": yolo_bbox[3],
                        "label": class_id,
                        "one_hot_label": one_hot_label
                    })
                    sample_image = tmp_image[:, :, :3]
            t.append(ground_truths)
            sample_image = random_hsv_image(sample_image, delta_hue, delta_sat_scale, delta_val_scale)


           # if i==2:
#                cv2.imshow("feed example", sample_image)
#                cv2.waitKey(500)
            #print("Saving images in training folder")
            #cv2.imwrite( "data/trainingYOLO/img%03i.png"  %i, sample_image )
            sample_image = np.asarray(sample_image, dtype=np.float32) / 255.0
            sample_image = sample_image.transpose(2, 0, 1)
            x.append(sample_image)

        x = np.array(x)
        #print("shape x correct", x.shape)
        return x, t
        
    def generate_simple_dataset(self, n_samples, w_in, h_in):
        x = []
        t = []
        ground_truths = []
        for i in range(n_samples):
            class_id = np.random.randint(len(self.labels))
            print("label", self.labels[class_id])

            item = self.items[class_id]
            #sample_image = random_sampling(item, crop_height, crop_width)
            sample_image = cv2.resize(item, (w_in, h_in))


            orig_h, orig_w = sample_image.shape[:2]
            bbox = ((0, 0), (int(orig_h), int(orig_w)))
            center_x = (bbox[1][0] + bbox[0][0]) / 2 / int(orig_w)
            center_y = (bbox[1][1] + bbox[0][1]) / 2 / int(orig_h)
            w = (bbox[1][0] - bbox[0][0]) / orig_w
            h = (bbox[1][1] - bbox[0][1]) / orig_h
            bbox = (center_x, center_y, w, h)
            box = Box(bbox[0], bbox[1], bbox[2], bbox[3])
            one_hot_label = np.zeros(len(self.labels))
            one_hot_label[class_id] = 1
            ground_truths.append({
                "x": bbox[0],
                "y": bbox[1],
                "w": bbox[2],
                "h": bbox[3],
                "label": class_id,
                "one_hot_label": one_hot_label
            })
            sample_image = sample_image[:, :, :3]
            print(one_hot_label)

            #print('shape', sample_image.shape[:2])

            t.append(ground_truths)
            #cv2.imshow("w", sample_image)
            #cv2.waitKey(100)
            sample_image = np.asarray(sample_image, dtype=np.float32) / 255.0
            sample_image = sample_image.transpose(2, 0, 1)
            vec = np.asarray(sample_image).astype(np.float32)

            x.append(vec)
        img = cv2.imread("./items/pet_water.png")
        img = cv2.resize(img, (w_in, h_in))
        #cv2.imshow('test feed', img)
        #cv2.waitKey(250)
        print('shape', img.shape[:2])
        img = img[:,:,:3]
        img = np.asarray(img, dtype=np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        vec2 = np.asarray(img).astype(np.float32)

        # load model
        #model.predictor.train = False

        # forward
        #x.append(vec2)
        x = np.array(x)
        return x,t



