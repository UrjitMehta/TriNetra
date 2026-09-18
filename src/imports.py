import os
import re
import glob
import math
import json
import gc
import random

import numpy as np
import matplotlib.pyplot as plt
import cv2
import tifffile as tiff

from PIL import UnidentifiedImageError

from sklearn.model_selection import KFold

import tensorflow as tf
import keras

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, Callback
)
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K
from tensorflow.keras import mixed_precision

from keras.saving import register_keras_serializable
from keras import config

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

from skimage import measure, exposure
from skimage.measure import label, regionprops
from skimage.transform import resize as sk_resize
from skimage.filters import threshold_otsu
from skimage.morphology import opening, closing, footprint_rectangle


# Enable mixed-precision training for improved computational efficiency.
mixed_precision.set_global_policy("mixed_float16")


def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


set_global_seed(42)


# ========== CONFIG PATHS ==========

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dataset_dir = os.path.join(project_root, "dataset", "combined_dataset")

train_img_dir = os.path.join(dataset_dir, "train_images")
train_lbl_dir = os.path.join(dataset_dir, "train_all_TIFF_labels")
# train_lbl_dir = os.path.join(dataset_dir, "train_labels")

test_img_dir = os.path.join(dataset_dir, "test_images")
test_lbl_dir = os.path.join(dataset_dir, "test_all_TIFF_labels")
# test_lbl_dir = os.path.join(dataset_dir, "test_labels")

model_dir = os.path.join(project_root, "model_output", "model", "swin_cl_200_test")
loss_dir = os.path.join(project_root, "model_output", "loss_output", "swin_cl_200_test")
output_dir = os.path.join(project_root, "model_output", "epoch_output", "swin_cl_200_test")

log_dir = os.path.join(project_root, "model_output", "logs", "fits", "swin_cl_200_test")

log_path = os.path.join(project_root, "model_output", "evaluation_logs")

summary_dir = os.path.join(
    project_root, "model_output", "summary_output_folder", "swin_cl_200_test"
)

os.makedirs(summary_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(loss_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

use_input_guidance = True
