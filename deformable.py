if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import os
import sys
import json
import time
import random
import skimage.draw
import numpy as np
import datetime
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)
ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR) 
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import zipfile
import urllib.request
import shutil
from mrcnn import visualize


# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_DIR = os.path.join(ROOT_DIR, "data")
DEFAULT_DATASET_YEAR = "2018"

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/deformable/")

# Configurations

class DeformableConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "deformable"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 13  # COCO has 80 classes
    
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 500

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


#  Dataset

# Dataset_dir: ROOT/data
# Subsets: deformable1. deformable2
# Year: 2018
class DeformableDataset(utils.Dataset):

    def load_deformable(self, dataset_dir, subset, year=DEFAULT_DATASET_YEAR, class_ids=None, 
        class_map=None, return_coco=False, auto_download=False): 

        coco = COCO("{}/train/all_deformable_masks.json".format(dataset_dir))
        if subset == "minival" or subset == "valminusminival":
            subset = "val"
        image_dir = "{}/{}".format(dataset_dir, subset)

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco

        
    def load_mask(self, image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_mask(image_id)


    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "deformable":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle


    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


############################################################
#  RLE Encoding
############################################################

def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)



def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = DeformableDataset()
    dataset_train.load_deformable(DEFAULT_DATASET_DIR, "images")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DeformableDataset()
    dataset_val.load_deformable(DEFAULT_DATASET_DIR, "images")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=300,
                layers='heads')


def detect(model):
    """Run detection on images in the given directory."""

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = DeformableDataset()
    dataset.load_deformable(DEFAULT_DATASET_DIR, "test")
    dataset.prepare()
    # Load over images
    submission = []
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        rle = mask_to_rle(source_id, r["masks"], r["scores"])
        submission.append(rle)
        # Save image with masks
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=False, show_mask=False,
            title="Predictions")
        plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))

    # Save to csv file
    submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
    file_path = os.path.join(submit_dir, "submit.csv")
    with open(file_path, "w") as f:
        f.write(submission)
    print("Saved to ", submit_dir)

#  Training
if __name__ == '__main__':
    import argparse
    config = DeformableConfig()
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=DEFAULT_LOGS_DIR)
    model_path = COCO_MODEL_PATH
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True, 
                      exclude=[ "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    train(model)

    # config = DeformableConfig()
    # model = modellib.MaskRCNN(mode="inference", config=config, model_dir=DEFAULT_LOGS_DIR)
    # # model_path = COCO_MODEL_PATH
    # weights_path = model.find_last()
    # model.load_weights(weights_path, by_name=True)
    # detect(model)



    # TEST_DATA_DIR = os.path.join(DEFAULT_DATASET_DIR, "test")
    # file_names = next(os.walk(TEST_DATA_DIR))[2]
    # image = skimage.io.imread(os.path.join(TEST_DATA_DIR, random.choice(file_names)))
    # #Run detection
    # results = model.detect([image], verbose=1)

    # submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    # dataset = DeformableDataset()
    # dataset.load_deformable(DEFAULT_DATASET_DIR, "images")
    # dataset.prepare()

    # # Visualize results
    # r = results[0]
    # visualize.display_instances(
    #         image, r['rois'], r['masks'], r['class_ids'],
    #         dataset.class_names, r['scores'],
    #         show_bbox=False, show_mask=False,
    #         title="Predictions")

    # plt.savefig("{}/{}.png".format(submit_dir, "result1"))


    # for image_id in TEST_DATA_DIR:
    #     # Load image and run detection
    #     image = skimage.io.imread(image_id)
    #     # Detect objects
    #     r = model.detect([image], verbose=0)[0]
    #     # Encode image to RLE. Returns a string of multiple lines
    #     source_id = dataset.image_info[image_id]["id"]
    #     rle = mask_to_rle(source_id, r["masks"], r["scores"])
    #     submission.append(rle)
    #     # Save image with masks
    #     visualize.display_instances(
    #         image, r['rois'], r['masks'], r['class_ids'],
    #         dataset.class_names, r['scores'],
    #         show_bbox=False, show_mask=False,
    #         title="Predictions")
    #     plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))





    # dataset_train = DeformableDataset()
    # dataset_train.load_deformable(DEFAULT_DATASET_DIR, "train", DEFAULT_DATASET_YEAR)
    
    # dataset_val = DeformableDataset()
    # val_type = "val"
    # dataset_val.load_deformable(DEFAULT_DATASET_DIR, val_type, DEFAULT_DATASET_YEAR)
    # dataset_val.prepare()
    
    # Training - Stage 1
    # print("Training network heads")
    # model.train(dataset_train, dataset_val,
    #                 learning_rate=config.LEARNING_RATE,
    #                 epochs=40,
    #                 layers='heads',
    #                 augmentation=augmentation)
    
    # # Training - Stage 2
    # # Finetune layers from ResNet stage 4 and up
    # print("Fine tune Resnet stage 4 and up")
    # model.train(dataset_train, dataset_val,
    #                 learning_rate=config.LEARNING_RATE,
    #                 epochs=120,
    #                 layers='4+',
    #                 augmentation=augmentation)
    
    # # Training - Stage 3
    # # Fine tune all layers
    # print("Fine tune all layers")
    # model.train(dataset_train, dataset_val,
    #                 learning_rate=config.LEARNING_RATE / 10,
    #                 epochs=160,
    #                 layers='all',
    #                 augmentation=augmentation)


