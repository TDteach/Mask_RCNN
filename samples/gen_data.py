import os
import sys
import random
import math
import numpy as np
import skimage.io
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import keras.backend as K
tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
sess = tf.Session(config=tfconfig)
K.set_session(sess)
# K.set_session(sess)
import keras.layers as KL
import keras.engine as KE
import keras.models as KM

#


# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library


from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version import coco
import coco
#matplotlib inline

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 2
    IMAGES_PER_GPU = 1



config = InferenceConfig()
config.display()

BATCH_SIZE = config.GPU_COUNT*config.IMAGES_PER_GPU

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


import json
def read_from_json(fname):
  with open(fname,'r') as f:
    data = json.load(f)
  return data


import pickle
def read_from_bin(fname):
  with open(fname,'rb') as f:
    data = pickle.load(f)
    return data
def write_to_bin(data, fname):
  with open(fname,'wb') as f:
    bd = pickle.dumps(data)
    f.write(bd)

person_thr = 0.90
prefix = '/home/public/tangdi/yellowset/GT-neg'
#root='/home/tdteach/data/thumbnails_features_deduped_publish/'
root='/home/public/tangdi/ground_truth'
#folders = read_from_json('/home/tdteach/data/sexy.json')
folders = os.listdir(root)

from queue import Queue, Empty, Full
import threading
import time

DONE = False
image_buffer = Queue(maxsize=100)
out_buffer = Queue(maxsize=100)

class MyThread(threading.Thread):
    def __init__(self, qu):
        threading.Thread.__init__(self)
        self.qu = qu
        self.bf = []
        self.fn = []

    def load_image(self, filename):
      try:
        print('loading '+filename)
        img = skimage.io.imread(filename)
        if img is None:
            return
        shape = img.shape
        if len(shape) != 3 or shape[0] < 10 or shape[1] < 10 or shape[2] != 3:
            return
        #self.qu.put((filename,img))
        self.bf.append(img)
        self.fn.append(filename)
        if len(self.fn) == BATCH_SIZE:
          self.qu.put((self.fn, self.bf))
          self.bf = []
          self.fn = []
          print('len %d\n' % (self.qu.qsize()*BATCH_SIZE))
      except OSError:
        pass
      except ValueError:
        pass

    def run(self):
      global DONE
      DONE=False
      zz = 0
      for fo in folders:
        if '.py' in fo or '.tar.gz' in fo:
          continue
        if 'positive' in fo:
          continue
        try:
            fo_path = os.path.join(root, fo)
            #fo_path = os.path.join(fo_path, 'noisy')
            list_filenames = os.listdir(fo_path)
        except NotADirectoryError:
            print('Not a directory')
            continue
        for f in list_filenames:
            zz += 1
            #if zz < 380000:
            #    continue
            nn = os.path.join(fo_path, f)
            if '.gif' in nn:
                continue
            if '.jpeg' in nn or '.png' in nn or '.jpg' in nn:
                self.load_image(nn)
      DONE=True


class OutThread(threading.Thread):
    def __init__(self, qu):
        threading.Thread.__init__(self)
        self.qu = qu
    def run(self):
        nf = 0 # num of bin files
        data = dict()

        n_rd = 0 # num of record files
        rd_ty = []
        rd_sc = []
        rd_roi = []
        rd_fn = []

        kk = 0

        while (not DONE) or (not self.qu.empty()):
            try:
              rsts, imgs, fns = self.qu.get(timeout=0.001)
              kk += 1
              print('done %d images' % (kk*BATCH_SIZE))
              for rst, img, fn in zip(rsts, imgs, fns):
                #print('dealing '+fn)
                ty = rst['class_ids']
                sc = rst['scores']
                roi = rst['rois']
                masks = rst['masks']

                rd_ty.append(ty)
                rd_sc.append(sc)
                rd_roi.append(roi)
                rd_fn.append(fn)

                if len(rd_fn) >= 20000:
                    np.savez(prefix+('.list.%d' % n_rd), filenames=rd_fn, classes=rd_ty, scores=rd_sc, rois=rd_roi)
                    n_rd += 1
                    rd_ty = []
                    rd_sc = []
                    rd_roi = []
                    rd_fn = []


                z = 0
                for k in range(len(ty)):
                    if ty[k] == 1 and sc[k] > person_thr:
                        img_st = np.zeros((128, 128, 4), dtype=np.float32)
                        cc = img[roi[k][0]:roi[k][2], roi[k][1]:roi[k][3]]
                        dd = skimage.transform.resize(cc, (128, 128))
                        mm = masks[roi[k][0]:roi[k][2], roi[k][1]:roi[k][3], k]
                        rs_mm = skimage.transform.resize(mm, (128, 128))
                        img_st[:, :, :3] = dd
                        img_st[:, :, 3] = rs_mm
                        na = ('%d_' % z) + fn
                        data[na] = img_st
                        # skimage.io.imsave('/home/tdteach/data/yellowset/'+fo+('%d_' % z)+f,dd)
                    z = z + 1

                if len(data) >= 10000:
                    write_to_bin(data, (prefix + '.bin.%d' % (nf)))
                    nf += 1
                    data = dict()
            except Empty:
                print('out_buffer empty')
                time.sleep(120)
                pass
        if len(data) > 0:
            write_to_bin(data,(prefix+'.bin.%d' % (nf)))
        if len(rd_fn) > 0:
            np.savez(prefix + ('.list.%d' % n_rd), filenames=rd_fn, classes=rd_ty, scores=rd_sc, rois=rd_roi)




t1 = MyThread(image_buffer)
t1.start()

t2 = OutThread(out_buffer)
t2.start()


zz = 0
while not DONE or not image_buffer.empty():
    try:
        fd_fn, feed = image_buffer.get(timeout=0.001)
        print('deal %d-%d images' % (zz*BATCH_SIZE, (zz+1)*BATCH_SIZE-1))
        zz += 1
        results = model.detect(feed, verbose=0)
        out_buffer.put((results, feed, fd_fn))
    except Empty:
        print('image_bufer empty')
        time.sleep(10)
        pass



'''
# Load a random image from the images folder
# file_names = next(os.walk(IMAGE_DIR))[2]
#image = skimage.io.imread(os.path.join(IMAGE_DIR, file_names[3]))
image = skimage.io.imread(os.path.join(IMAGE_DIR, '757.jpg'))
image_cv2 = cv2.imread(os.path.join(IMAGE_DIR, '757.jpg'))

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]

print(r['masks'].shape)
print(results[0]['scores'])
print(results[0]['class_ids'])
print(results[0]['rois'])
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])
'''
