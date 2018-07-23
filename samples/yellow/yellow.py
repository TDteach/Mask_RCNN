"""
ResNet50 to detect yellow image based on the output of Mask R-CNN
Configurations and data loading code for yellowset.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by TDteach

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model
    python3 yellow.py train --dataset=/path/to/yellowset/ --model=new

    # Continue training a model that you had trained earlier
    python3 yellow.py train --dataset=/path/to/yellowset/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 yellow.py train --dataset=/path/to/yellowset/ --model=last

    # Run XHT evaluatoin on the last model you trained
    python3 yellow.py evaluate --dataset=/path/to/yellowset/ --model=last
"""

import os
import sys
import time
import numpy as np
import random
import imgaug

import zipfile
import urllib.request
import shutil

import logging
import tensorflow as tf
import keras
import keras.backend as K
tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
sess = tf.Session(config=tfconfig)
K.set_session(sess)
# K.set_session(sess)
import keras.layers as KL
import keras.engine as KE
import keras.models as KM

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
import multiprocessing

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################


class XHTConfig(Config):
    """Configuration for training on XHT.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """

    # Give the configuration a recognizable name
    NAME = "xiaohuangtu"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 64

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 2

    # Number of classes (including background)
    NUM_CLASSES = 2  # XHT has 2 classes: white & yellow

    IMAGE_DIM = 128
    IMAGE_MAX_DIM = IMAGE_DIM
    IMAGE_MIN_DIM = IMAGE_DIM

    LOSS_WEIGHTS = {
        "class_loss": 1.
    }
    """
    Note about training values:
        None: Train BN layers. This is the normal mode
        False: Freeze BN layers. Good when batch size is small
        True: (don't use). Set layer in training mode even when inferencing
    """
    TRAIN_BN = None
    VALIDATION_STEPS = 50


    BACKBONE = None
    BACKBONE_STRIDES = None
    BBOX_STD_DEV = None
    DETECTION_MAX_INSTANCES = None
    DETECTION_MIN_CONFIDENCE = None
    DETECTION_NMS_THRESHOLD = None
    IMAGE_MIN_SCALE = None
    MASK_POOL_SIZE = None
    MASK_SHAPE = None
    MAX_GT_INSTANCES = 2
    MEAN_PIXEL = None
    MINI_MASK_SHAPE = None
    POOL_SIZE = None
    POST_NMS_ROIS_INFERENCE = None
    POST_NMS_ROIS_TRAINING = None
    ROI_POSITIVE_RATIO = None
    RPN_ANCHOR_RATIOS = None
    RPN_ANCHOR_SCALES = None
    RPN_ANCHOR_STRIDE = None
    RPN_BBOX_STD_DEV = None
    RPN_NMS_THRESHOLD = None
    RPN_TRAIN_ANCHORS_PER_IMAGE = None
    TRAIN_ROIS_PER_IMAGE = None
    STEPS_PER_EPOCH = None
    USE_MINI_MASK = None
    USE_RPN_ROIS = None




############################################################
#  Utility Functions
############################################################

def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}  {}".format(
            str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else "",
            array.dtype))
    print(text)

import pickle
def read_from_bin(fname):
  print('read data from '+fname)
  with open(fname,'rb') as f:
    data = pickle.load(f)
  lb = 0
  if 'yellow_imgs' in fname or 'porno_imgs' in fname:
    lb = 1
  rst = []
  for n,d in data.items():
    rst.append((d,lb))
  return rst

############################################################
#  Dataset
############################################################
class XHTDataset(utils.Dataset):
    def load_xht(self, bin_list, subset):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        """

        self.b_bin_list = []
        self.y_bin_list = []
        for b in bin_list:
            if 'yellow_imgs' in b or 'porno_imgs' in b:
                self.y_bin_list.append(b)
            else:
                self.b_bin_list.append(b)

        self.len_b = len(self.b_bin_list)
        self.len_y = len(self.y_bin_list)

        random.shuffle(self.b_bin_list)
        random.shuffle(self.y_bin_list)
        self.k_bin = 0
        self._feed_buffer()


        # for fn in bin_list:
        #     data = read_from_bin(fn)
        #     self.image_info.extend(data)
    def _feed_buffer(self):
        self.image_info = []
        while len(self.image_info) < 30000:
            data = read_from_bin(self.b_bin_list[self.k_bin%self.len_b])
            self.image_info.extend(data)
            data = read_from_bin(self.y_bin_list[self.k_bin%self.len_y])
            self.image_info.extend(data)
            self.k_bin += 1
            if (self.k_bin%self.len_b) == 0:
                random.shuffle(self.b_bin_list)
            if (self.k_bin%self.len_y) == 0:
                random.shuffle(self.y_bin_list)
        self._k_image = 0
        self._idx = np.arange(len(self.image_info))
        random.shuffle(self._idx)


    def prepare(self, class_map=None):
        self.num_classes = 2
        self.class_ids = np.arange(self.num_classes)
        self.class_names = ['white','yellow']
        # self.num_images = len(self.image_info)
        self.num_images = (self.len_b+self.len_y) * 5000
        self._image_ids = np.arange(self.num_images)


    def load_image(self, image_id):
        img, c_id = self.image_info[self._idx[self._k_image]]
        self._k_image += 1
        if self._k_image >= len(self._idx):
            self._feed_buffer()
        return img, c_id


def mold_image(image, config=None):
    """ Normalize image into [-1,1]
    """
    if np.max(image) <= 1 and np.min(image) >= 0:
        image[:,:,:3] = image[:,:,:3]*2.0 - 1.0
    elif np.min(image) >= 0:
        image[:, :, :3] = image[:, :, :3] * (1.0/127.5) - 1.0
    return image.astype(np.float32)

def unmold_image(normalized_images, config=None):
    """Takes a image normalized with mold() and returns the original."""
    img = normalized_images[:, :, :3]
    mask = normalized_images[:, :, 3] > 0.5
    mask = np.expand_dims(mask, -1).astype(np.bool)
    if np.min(img) < 0:
        img = img*127.5 +127.5
    else:
        img = img * 255
    return img.astype(np.uint8), mask

def load_image_gt(dataset, config, image_id, augment=False, augmentation=None):
    """Load and return ground truth data for an image (image, mask, bounding boxes).

    augment: (Depricated. Use augmentation instead). If true, apply random
        image augmentation. Currently, only horizontal flipping is offered.
    augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
        For example, passing imgaug.augmenters.Fliplr(0.5) flips images
        right/left 50% of the time.

    Returns:
    image: [height, width, 3]
    shape: the original shape of the image before resizing and cropping.
    class_ids: [instance_count] Integer class IDs
    """
    # Load image and mask
    image, class_id = dataset.load_image(image_id)

    # Random horizontal flips.
    # TODO: will be removed in a future update in favor of augmentation
    if augment:
        logging.warning("'augment' is depricated. Use 'augmentation' instead.")
        if random.randint(0, 1):
            image = np.fliplr(image)

    # Augmentation
    # This requires the imgaug lib (https://github.com/aleju/imgaug)
    if augmentation:

        # Store shapes before augmentation to compare
        image_shape = image.shape
        # Make augmenters deterministic to apply similarly to images and masks
        det = augmentation.to_deterministic()
        image = det.augment_image(image)
        # Verify that shapes didn't change
        assert image.shape == image_shape, "Augmentation shouldn't change image size"

    return image, class_id

def data_generator(dataset, config, shuffle=True, augment=False, augmentation=None, batch_size=1):
    """A generator that returns images and corresponding target class ids,
    bounding box deltas, and masks.

    dataset: The Dataset object to pick data from
    config: The model config object
    shuffle: If True, shuffles the samples before every epoch
    augment: (Depricated. Use augmentation instead). If true, apply random
        image augmentation. Currently, only horizontal flipping is offered.
    augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
        For example, passing imgaug.augmenters.Fliplr(0.5) flips images
        right/left 50% of the time.
    batch_size: How many images to return in each call
    detection_targets: If True, generate detection targets (class IDs, bbox
        deltas, and masks). Typically for debugging or visualizations because
        in trainig detection targets are generated by DetectionTargetLayer.

    Returns a Python generator. Upon calling next() on it, the
    generator returns two lists, inputs and outputs. The containtes
    of the lists differs depending on the received arguments:
    inputs list:
    - images: [batch, H, W, C]
    - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs

    outputs list: Usually empty in regular training. But if detection_targets
        is True then the outputs list contains target class_ids, bbox deltas,
        and masks.
    """
    b = 0  # batch item index
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0

    # Keras requires a generator to run indefinately.
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            # Get GT bounding boxes and masks for image.
            image_id = image_ids[image_index]
            image, gt_class_ids = load_image_gt(dataset, config, image_id, augment=augment,
                              augmentation=augmentation)

            # Init batch arrays
            if b == 0:
                batch_images = np.zeros(
                    (batch_size,) + image.shape, dtype=np.float32)
                batch_gt_class_ids = np.zeros(
                    (batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)

            # Add to batch
            batch_images[b] = mold_image(image.astype(np.float32), config)
            batch_gt_class_ids[b, gt_class_ids] = 1
            b += 1

            # Batch full?
            if b >= batch_size:
                inputs = [batch_images, batch_gt_class_ids]
                outputs = []

                yield inputs, outputs

                # start a new batch
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(
                dataset.image_info[image_id]))
            error_count += 1
            if error_count > 5:
                raise
############################################################
#  Loss Functions
############################################################
def class_loss_graph(target_class_ids, pred_class_logits):
    """Loss for the classifier head of Mask RCNN.

    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    """
    # During model building, Keras calls this function with
    # target_class_ids of type float32. Unclear why. Cast it
    # to int to get around it.
    target_class_ids = tf.cast(target_class_ids, 'int64')

    # Find predictions of classes that are not in the dataset.
    # pred_class_ids = tf.argmax(pred_class_logits, axis=2)

    # Loss
    print(pred_class_logits)
    y = tf.layers.flatten(pred_class_logits)
    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=target_class_ids, logits=y)


    # Computer loss mean. Use only predictions that contribute
    # to the loss to get a correct mean.
    loss = tf.reduce_mean(loss)
    return loss

############################################################
#  ResNet50 Model
############################################################
class ResNet50(modellib.MaskRCNN):

    def build(self, mode, config):
        """Build ResNet-50 architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        """
        assert mode in ['training', 'inference']

        # Image size must be dividable by 2 multiple timesk, default value : 128x128
        h, w = config.IMAGE_SHAPE[:2]

        # Inputs
        input_image = KL.Input(
            shape=[config.IMAGE_DIM, config.IMAGE_DIM, 4], name="input_image")

        if mode == "training":
            # GoundTruth Class IDs (zero padded)
            input_gt_class_ids = KL.Input(
                shape=[config.MAX_GT_INSTANCES], name="input_gt_class_ids", dtype=tf.int32)


        # Build the shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the thead (stage 5), so we pick the 4th item in the list.
        _, _, _, _, C5 = modellib.resnet_graph(input_image, 'resnet50',
                                         stage5=True, train_bn=config.TRAIN_BN)
        # Top-down Layers
        # TODO: add assert to varify feature map sizes match what's in config
        P5 = KL.Conv2D(256, (1, 1), name='fpn_c5p5')(C5)
        #P5 with shape [4,4,256]
        P6 = KL.MaxPooling2D(pool_size=P5.shape[1:3], name="fpn_p6")(P5)

        if mode == "training":
            x = KL.Dropout(rate=0.5)(P6)
            # class_logits = KL.Dense(2, activation='softmax', name='class_logits')(x)
            class_logits = KL.Dense(2, name='class_logits')(x)
            class_loss = KL.Lambda(lambda x: class_loss_graph(*x), name="class_loss")(
                [input_gt_class_ids, class_logits])
            model = KM.Model([input_image, input_gt_class_ids], [class_loss], name='resnet50')
        else:
            y = KL.Dense(2, activation='softmax', name='class_logits')(P6)
            model = KM.Model([input_image], [y], name='resnet50')


        # Add multi-GPU support.
        if config.GPU_COUNT > 1:
            from mrcnn.parallel_model import ParallelModel
            model = ParallelModel(model, config.GPU_COUNT)

        return model

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers='all',
              augmentation=None):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heaads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        augmentation: Optional. An imgaug (https://github.com/aleju/imgaug)
            augmentation. For example, passing imgaug.augmenters.Fliplr(0.5)
            flips images right/left 50% of the time. You can pass complex
            augmentations as well. This augmentation applies 50% of the
            time, and when it does it flips images right/left half the time
            and adds a Gausssian blur with a random sigma in range 0 to 5.

                augmentation = imgaug.augmenters.Sometimes(0.5, [
                    imgaug.augmenters.Fliplr(0.5),
                    imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
                ])
        """
        assert self.mode == "training", "Create model in training mode."

        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_generator = data_generator(train_dataset, self.config, shuffle=True,
                                         augmentation=augmentation,
                                         batch_size=self.config.BATCH_SIZE)
        val_generator = data_generator(val_dataset, self.config, shuffle=True,
                                       batch_size=self.config.BATCH_SIZE)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True),
        ]
# Train
        modellib.log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        modellib.log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name is 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()
            workers = 2

        self.config.STEPS_PER_EPOCH = train_dataset.num_images / self.config.BATCH_SIZE

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=True,
        )
        self.epoch = max(self.epoch, epochs)

    def detect(self, images, verbose=0):
        """Runs the detection pipeline.
        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(images)))
            for image in images:
                log("image", image)

        # Mold inputs to format expected by the neural network
        molded_images = []
        for img in images:
            molded_images.append(mold_image(img))

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape,\
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

        molded_images = np.asarray(molded_images)
        if verbose:
            log("molded_images", molded_images)
        # Run object detection
        y = self.keras_model.predict([molded_images], verbose=0)
        # Process detections
        results = []
        for i, image in enumerate(images):
            results.append(y[i][0][0][1])
        return results

    def compile(self, learning_rate, momentum):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        # Optimizer object
        optimizer = keras.optimizers.SGD(
            lr=learning_rate, momentum=momentum,
            clipnorm=self.config.GRADIENT_CLIP_NORM)
        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = ["class_loss"]
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            loss = (
                tf.reduce_mean(layer.output, keepdims=True)
                * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.add_loss(loss)

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
            for w in self.keras_model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.keras_model.compile(
            optimizer=optimizer,
            loss=[None] * len(self.keras_model.outputs))

        # Add metrics for losses
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            loss = (
                tf.reduce_mean(layer.output, keepdims=True)
                * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.metrics_tensors.append(loss)



############################################################
#  XHT Evaluation
############################################################

def evaluate_xht(model, dataset, limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    t_prediction = 0
    t_start = time.time()

    results = []
    gt_id = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image, c_id = dataset.load_image(image_id)
        gt_id.append(c_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        # image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
        #                                    r["rois"], r["class_ids"],
        #                                    r["scores"],
        #                                    r["masks"].astype(np.uint8))
        results.extend(r)

    # Calculate the AUC of results
    from sklearn import metrics
    fpr, tpr, thr = metrics.roc_curve(gt_id, results)
    print("AUC: %.6f" % (metrics.auc(fpr,tpr)))

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on MS COCO")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/yellow/",
                        help='Directory of the yellow dataset')
    parser.add_argument('--model', required=False,
                        metavar="/path/to/weights.h5",
                        default='new',
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=0,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=0, no limit)')
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = XHTConfig()
    else:
        class InferenceConfig(XHTConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = ResNet50(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = ResNet50(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == 'new':
        model_path = None
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()[1]
    else:
        model_path = args.model

    # Load weights
    if model_path:
        print("Loading weights ", model_path)
        model.load_weights(model_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the 1% training set form the
        # validation set.
        fn_list = os.listdir(args.dataset)
        tr_list = []
        vl_list = []



        for fn in fn_list:
            tr_list.append(os.path.join(args.dataset,fn))
        vl_list.append(tr_list[3])
        vl_list.append(tr_list[0])

        '''
        for fn in fn_list:
            if 'train' in fn:
                tr_list.append(os.path.join(args.dataset,fn))
            elif 'val' in fn:
                vl_list.append(os.path.join(args.dataset,fn))
        '''
        dataset_train = XHTDataset()
        dataset_train.load_xht(tr_list, "train")
        dataset_train.prepare()

        # Validation dataset
        dataset_val = XHTDataset()
        dataset_val.load_xht(vl_list, "val")
        dataset_val.prepare()

        # Image Augmentation
        # Right/Left flip 50% of the time
        augmentation = imgaug.augmenters.Fliplr(0.5)

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network Stage 1")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    augmentation=augmentation)

        # Training - Stage 2
        # decrease the learning rate
        print("Fine tune Resnet Stage 2")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE*0.1,
                    epochs=80,
                    augmentation=augmentation)


    elif args.command == "evaluate":
        # Validation dataset
        fn_list = os.listdir(args.dataset)
        vl_list = []
        for fn in fn_list:
            if 'val' in fn:
                vl_list.append(os.path.join(args.dataset,fn))
        dataset_val = XHTDataset()
        dataset_val.load_xht(vl_list, "val")
        dataset_val.prepare()
        print("Running XHT evaluation on {} images.".format(args.limit))
        evaluate_xht(model, dataset_val, limit=int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))


# URL from which to download the latest COCO trained weights
XHT_MODEL_URL = "https://github.com/TDteach/Mask_RCNN/releases/download/yellow/resnet50_yellow_v2.h5"

def download_yellow_weights(xht_model_path, verbose=1):
    """Download COCO trained weights from Releases.

    coco_model_path: local path of COCO trained weights
    """
    if verbose > 0:
        print("Downloading pretrained model to " + xht_model_path + " ...")
    with urllib.request.urlopen(XHT_MODEL_URL) as resp, open(xht_model_path, 'wb') as out:
        shutil.copyfileobj(resp, out)
    if verbose > 0:
        print("... done downloading pretrained model!")
