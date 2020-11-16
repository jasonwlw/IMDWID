"""
Mask R-CNN
Configurations and data loading code for the synthetic Shapes dataset.
This is a duplicate of the code in the noteobook train_rgbd.ipynb for easy
import into other notebooks, such as inspect_model.ipynb.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import sys
import math
import random
import numpy as np
import cv2
from scipy.io import loadmat

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils


class RGBDConfig(Config):
    """Configuration for training on the toy rgbd dataset.
    Derives from the base Config class and overrides values specific
    to the toy rgbd dataset.
    """
    # Give the configuration a recognizable name
    NAME = "rgbd"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 6  # background + 3 rgbd

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 480
    IMAGE_MAX_DIM = 640

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

    # RGB-D num channels
    IMAGE_CHANNEL_COUNT = 4

    #Mean pixel update
    MEAN_PIXEL = 4


class RGBDDataset(utils.Dataset):
    """Generates the rgbd synthetic dataset. The dataset consists of simple
    rgbd (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_images(self, data_dir, height, width):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """

        # Add classes
        with open(os.path.join(ROOT_DIR, '../classes.txt'), 'r') as f0:
            for i,line in enumerate(f0.readlines()):
                line = line.strip()
                self.add_class("rgbd", i+1, line) 

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of rgbd sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        ids = 0
        
        for im in os.listdir(data_dir):
            if 'combined' in im:
                mask_path = os.path.join(data_dir, im.replace('combined', 'masks'))
                class_path = os.path.join(data_dir, im.replace('combined', 'classes').replace('.npy','.txt'))
                self.add_image("rgbd", image_id = ids, path = os.path.join(data_dir, im), width = width, height = height, mask_path = mask_path, class_path = class_path)
                ids += 1

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        mask = np.load(info['mask_path'])
        classes = []
        with open(info['class_path'], 'r') as f0:
            for line in f0.readlines():
                classes.append(line.strip())
        # Handle occlusions
        count = mask.shape[2]
        if count > 1:
            occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
            for i in range(count - 2, -1, -1):
                mask[:, :, i] = mask[:, :, i] * occlusion
                occlusion = np.logical_and(
                    occlusion, np.logical_not(mask[:, :, i]))
            # Map class names to class IDs.
        elif count == 1:
            pass
        else:
            # Handle empty image?
            pass
        class_ids = np.array([self.class_names.index(s) for s in classes])
        return mask, class_ids.astype(np.int32)


    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        Returns numpy array: load arrays into memory
        """
        info = self.image_info[image_id]
        image = np.load(info['path'])
        return image

    def image_reference(self, image_id):
        """Return the rgbd data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "rgbd":
            return info["rgbd"]
        else:
            super(self.__class__).image_reference(self, image_id)
"""
    def load_mask(self, image_id):
        #Generate instance masks for rgbd of the given image ID.
        info = self.image_info[image_id]
        rgbd = info['rgbd']
        count = len(rgbd)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        for i, (shape, _, dims) in enumerate(info['rgbd']):
            mask[:, :, i:i + 1] = self.draw_shape(mask[:, :, i:i + 1].copy(),
                                                  shape, dims, 1)
        return mask, class_ids.astype(np.int32)
"""
