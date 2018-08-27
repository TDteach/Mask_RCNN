import os
import sys
import numpy as np


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
# sys.path.append(ROOT_DIR)  # To find local version of the library



# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

MODEL_NAME = "resnet50_yellow_fp003_poor.h5"
# Local path to trained weights file
MODEL_PATH = os.path.join(ROOT_DIR, MODEL_NAME)

import yellow
config = yellow.XHTConfig()


root = '/home/public/tangdi/yellowset/'


class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1000

config = InferenceConfig()
config.display()




import pickle
def read_from_bin(fname):
  with open(fname,'rb') as f:
    data = pickle.load(f)
  return data


model = yellow.ResNet50(mode="inference", model_dir=MODEL_DIR,
                              config=config)

weights_path = MODEL_PATH

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)


fn_list = os.listdir(root)
feed = []
fd_gt = []

gt_files = ['GT-porn.bin.0','GT-neg.bin.0']
for f in gt_files:
    fname = os.path.join(root,f)
    print(fname)
    data = read_from_bin(fname)
    for na, d in data.items():
        feed.append(d)
        if 'porn' in na:
            fd_gt.append(1)
        else:
            fd_gt.append(0)


results = model.detect(feed, verbose=0)

from sklearn import metrics

fpr, tpr, thr = metrics.roc_curve(fd_gt, results)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(fpr,tpr)
plt.show()





