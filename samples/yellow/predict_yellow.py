import os
import sys
import numpy as np


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
# sys.path.append(ROOT_DIR)  # To find local version of the library



# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

MODEL_NAME = "resnet50_yellow_fp003.h5"
# Local path to trained weights file
MODEL_PATH = os.path.join(ROOT_DIR, MODEL_NAME)

import yellow
config = yellow.XHTConfig()


root = '/home/tangdi/data/yellow_predict/'
prefix = '/home/tangdi/data/yellow_predict/yw_score'


class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 2
    IMAGES_PER_GPU = 10

config = InferenceConfig()
config.display()

BATCH_SIZE = config.GPU_COUNT*config.IMAGES_PER_GPU



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
rd_yw = []
n_rd = 0
for fname in fn_list:
    data = read_from_bin(fname)
    for d in data:
        feed.append(d)
        if len(feed) == BATCH_SIZE:
            results = model.detect(feed, verbose=0)
            for sc in results:
                rd_yw.append(sc)
            feed=[]

            if len(rd_yw) >= 10000:
                np.save(prefix + ('.%d' % n_rd), rd_yw)
                rd_yw = []
                n_rd += 1



np.save(prefix + ('.%d' % n_rd), rd_yw)