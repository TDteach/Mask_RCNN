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


BATCH_SIZE = 5

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
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
    GPU_COUNT = 1
    IMAGES_PER_GPU = BATCH_SIZE

config = InferenceConfig()
config.display()

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

person_thr = 0.8
prefix = 'porno_imgs.bin'
root='/home/tdteach/data/yellowset/images/porno/'
#folders = read_from_json('/home/tdteach/data/sexy.json')
folders = os.listdir(root)
nf = 2
zz = 0
data = dict()
feed = []
for fo in folders:
    if '.' in fo:
      continue
    #if folders[fo] == 'male':
    #    print('male')
    #    continue
    try:
      fo_path = os.path.join(root,fo)
    except NotADirectoryError:
      continue
    list_filenames = os.listdir(fo_path)
    for f in list_filenames:
        try:
            nn = os.path.join(fo_path, f)
            print('%d : %s' %(zz, nn))
            img = skimage.io.imread(nn)
            if img is None:
                continue
            shape = img.shape
            if len(shape) != 3 or shape[0] < 10 or shape[1] < 10 or shape[2] != 3:
                continue
            #print(shape)
            zz += 1
            if (zz <= 10000):
                continue
            feed.append(img)
            if len(feed) == BATCH_SIZE:
              results = model.detect(feed, verbose=0)
              z = 0
              for rst,img in zip(results,feed):
                ty = rst['class_ids']
                sc = rst['scores']
                roi = rst['rois']
                masks = rst['masks']
                for k in range(len(ty)):
                  if ty[k] == 1 and sc[k] > person_thr:
                    img_st = np.zeros((128,128,4),dtype=np.float32)
                    cc = img[roi[k][0]:roi[k][2], roi[k][1]:roi[k][3]]
                    dd = skimage.transform.resize(cc,(128,128))
                    mm = masks[roi[k][0]:roi[k][2], roi[k][1]:roi[k][3], k]
                    rs_mm = skimage.transform.resize(mm,(128,128))
                    img_st[:,:,:3] = dd
                    img_st[:,:,3] = rs_mm
                    na = fo+('/%d_'%z)+f
                    data[na] = img_st
                    #skimage.io.imsave('/home/tdteach/data/yellowset/'+fo+('%d_' % z)+f,dd)
                    z = z+1
              feed = []
              if (zz%5000) == 0:
                write_to_bin(data,(prefix+'.%d' % (nf)))
                nf += 1
                data = dict()
        except OSError:
            pass
        except ValueError:
            pass

if len(data) > 0:
    write_to_bin(data,(prefix+'.%d' % (nf)))


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
