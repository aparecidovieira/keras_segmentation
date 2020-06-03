#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import keras, os
from keras.models import *
from keras.layers import *
from tensorflow.python.keras import losses
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
import cv2, glob
import matplotlib.pyplot as plt
import itertools
# from util import losses, custom_data_generator, metrics
from keras.utils import multi_gpu_model
from . models import model_loader
import datetime
# from processing import abs_sobel_thresh
# from processing import mag_threshold
from . processing import *
# from util import custom_data_generator as data_util
from . models.common import lanenet_wavelet
from keras import backend as K

from keras.callbacks import TensorBoard,Callback
from PIL import Image
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
# set_session(tf.Session(config=config))

class LaneDetection(object):
    def __init__(self, dir_name, gpu, image_height=256, image_width=256):
        self.image_width, self.image_height = image_height, image_width
        self.channels = 3
        # checkpoint_name = "checkpoint_lanenet"
        self.one_hot_label= False
        self.data_aug = False
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        set_session(tf.Session(config=config))
        # self.bigImg = bigImg
        # self.bigMask = bigMask
        # In[44]:
        self.dir_name = dir_name

    def DetectLane(self):
        bigImg = cv2.imread(self.dir_name + '/sat_original.png', -1)
        bigMask = cv2.imread(self.dir_name + '/sat_mask.png', 0)
        index = bigMask == 0
        bigImg[index] = (0, 0, 0)
        h, w, _ =  bigImg.shape
        # checkpoint_name = "checkpoint_lanenet"
        checkpoint_name = "Mix_dataset_full_v2_BN_lanenet/"
        model_name = 'lanenet'
        # checkpoint_dir = "./checkpoints/%s/"%(checkpoint_name)
        checkpoint_dir_root = "/home/beemap/Documents/cesar-workspace/Segmentation/Keras_Code/keras3/checkpoints/"
        checkpoint_dir = checkpoint_dir_root + checkpoint_name
        print("Loading Model from.... ",checkpoint_dir)
        json_file = open(checkpoint_dir + model_name +".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights("%s/%s_weights_90.h5"%(checkpoint_dir, model_name))


        def binary_pipeline(img, mask):

            ind = (mask == 0)
            img[ind] = 0
            img_copy = cv.GaussianBlur(img, (3, 3), 0)
            #img_copy = np.copy(img)

            # color channels
            s_binary = hls_select(img_copy, sthresh=(140, 255), lthresh=(120, 255))
            #red_binary = red_select(img_copy, thresh=(200,255))

            # Sobel x
            x_binary = abs_sobel_thresh(img_copy,thresh=(25, 200))
            y_binary = abs_sobel_thresh(img_copy,thresh=(25, 200), orient='y')
            xy = cv.bitwise_and(x_binary, y_binary)

            #magnitude & direction
            mag_binary = mag_threshold(img_copy, sobel_kernel=3, thresh=(30,100))
            dir_binary = dir_threshold(img_copy, sobel_kernel=3, thresh=(0.8, 1.2))

            # Stack each channel
            gradient = np.zeros_like(s_binary)
            gradient[((x_binary == 1) & (y_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
            final_binary = cv.bitwise_or(s_binary, gradient)

            return final_binary

        bigLane = np.zeros_like((bigImg))
        for r in range(0, h, self.image_height):
            for c in range(0, w, self.image_width):

                img = bigImg[r:r+256, c:c+256, :3]
        #         h, w, _ = img.shape
                tmp = img[:]
                img = np.float32(img)/255.0

        #         input_image_gray = data_util.get_image(image,do_aug=[],gray=True, change=False)
                input_image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                w1, w2, w3, w4 = lanenet_wavelet(input_image_gray)
                w1 = np.expand_dims(w1, axis=0)
                w2 = np.expand_dims(w2, axis=0)
                w3 = np.expand_dims(w3, axis=0)
                w4 = np.expand_dims(w4, axis=0)

                mask = model.predict([np.expand_dims(img, axis=0), w1, w2, w3, w4], batch_size=None, verbose=0, steps=None)

                mask = np.round(mask[0, :, :, 0]).astype(int)
                seg = np.zeros((256, 256, 3))

                seg[:, :, 0] += ((mask[:, :] == 1) * (255)).astype('uint8')
                seg[:, :, 1] += ((mask[:, :] == 1) * ( 255)).astype('uint8')
                seg[:, :, 2] += ((mask[:, :] == 1) * ( 255)).astype('uint8')

                bigLane[r:r+256, c:c+256, :] = seg
        K.clear_session()
        name = '/final_marks.png'
        dest = self.dir_name + name
        cv2.imwrite(dest, bigLane)
