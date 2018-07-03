import argparse
import urllib.request
import numpy as np
import cv2

def download_image(): 
    parser = argparse.ArgumentParser(description='Web上から画像をダウンロードし、処理を行う。')
    parser.add_argument('--url', '-u', default='http://blog-imgs-63.fc2.com/o/o/i/ooinarukuu/20140502104641477.jpg', help='ダウンロードするイメージのURLを指定する')
    args = parser.parse_args()

    print('Download Image From {0} ....'.format(args.url))
    image_file_path = './sample_images/sample.jpg'
    urllib.request.urlretrieve(args.url, image_file_path)

    return image_file_path

#box structure is x,y,h,w
def _offset_boxes(boxes, im_shape, scale, offs, flip):
    if len(boxes) == 0:
        return boxes
    boxes = np.asarray(boxes, dtype=np.float)
    boxes *= scale
    #From col indice 0 2by2 (on x)
    boxes[:, 0:3:2] -= offs[0]
    #From col indice 1 2by2 8on y)
    boxes[:, 1:3:2] -= offs[1]
    boxes = clip_boxes(boxes, im_shape)

    if flip:
        boxes_x = np.copy(boxes[:, 0])
        boxes[:, 0] = im_shape[1] - boxes[:, 2]
        boxes[:, 2] = im_shape[1] - boxes_x
    return boxes

def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    if boxes.shape[0] == 0:
        return boxes

    # x1 >= 0
    boxes[:, 0] = np.maximum(np.minimum(boxes[:, 0], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1] = np.maximum(np.minimum(boxes[:, 1], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2] = np.maximum(np.minimum(boxes[:, 2], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3] = np.maximum(np.minimum(boxes[:, 3], im_shape[0] - 1), 0)
    return boxes

def imcv2_recolor(im, a=.1):
    # t = [np.random.uniform()]
    # t += [np.random.uniform()]
    # t += [np.random.uniform()]
    # t = np.array(t) * 2. - 1.
    t = np.random.uniform(-1, 1, 3)

    # random amplify each channel
    im = im.astype(np.float)
    im *= (1 + t * a)
    mx = 255. * (1 + a)
    up = np.random.uniform(-1, 1)
    im = np.power(im / mx, 1. + up * .5)
    # return np.array(im * 255., np.uint8)
    return im


def imcv2_affine_trans(im):
    # Scale and translate
    h, w, c = im.shape
    scale = np.random.uniform() / 10. + 1.
    max_offx = (scale - 1.) * w
    max_offy = (scale - 1.) * h
    offx = int(np.random.uniform() * max_offx)
    offy = int(np.random.uniform() * max_offy)

    im = cv2.resize(im, (0, 0), fx=scale, fy=scale)
    im = im[offy: (offy + h), offx: (offx + w)]
    flip = np.random.uniform() > 0.5
    if flip:
        im = cv2.flip(im, 1)

    return im, [scale, [offx, offy], flip]
