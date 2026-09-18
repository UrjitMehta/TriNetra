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
