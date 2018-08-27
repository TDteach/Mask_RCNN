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


root = '/home/public/tangdi/yellow_predict/mrcnn_rst/'
prefix = '/home/public/tangdi/yellow_predict/mrcnn_rst/tieba-all.sclist'


save_length = 100000
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 2
    IMAGES_PER_GPU = 1000

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
fd_na = []
rd_yw = []
rd_na = []
n_rd = 0
for k in range(405):
    fname = os.path.join(root,('tieba-all.bin.%d' % k))
    print(fname)
    data = read_from_bin(fname)
    for na, d in data.items():
        feed.append(d)
        fd_na.append(na)
        if len(feed) == BATCH_SIZE:
            results = model.detect(feed, verbose=0)
            for f_na, sc in zip(fd_na, results):
                rd_na.append(f_na)
                rd_yw.append(sc)
            feed=[]
            fd_na = []

            if len(rd_yw) >= save_length:
                np.savez(prefix + ('.%d' % n_rd), boxid=rd_na, yellow=rd_yw)
                rd_yw = []
                rd_na = []
                n_rd += 1



np.savez(prefix + ('.%d' % n_rd), boxid=rd_na, yellow=rd_yw)
