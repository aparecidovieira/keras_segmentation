import os, sys
# sys.path.append('.')
# print(os.path.abspath
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import keras
import tensorflow as tf
from keras.models import *
# from keras.layers import *
# from tensorflow.python.keras import losses
# from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
import cv2
import glob
import matplotlib.pyplot as plt
from keras import backend as K

import itertools
# from . util import losses, custom_data_generator, metrics
from keras.utils import multi_gpu_model
from . models import model_loader
import datetime
# from processing import abs_sobel_thresh
# from processing import mag_threshold
# from processing import *
# from util import custom_data_generator as data_util
from . models.common import lanenet_wavelet
sys.path.append('.')
from  helper import *
from cv2 import imread, imwrite, flip
from PIL import Image
import shutil
import matplotlib.pyplot as plt
from keras.backend.tensorflow_backend import set_session

# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
# set_session(tf.Session(config=config))

class LineDetection(object):
    def __init__(self, coords, lane_v2, gpu, folder, data=False, image_width=256, image_height=256):
        batch_size =30

        self.image_width, self.image_height = image_width, image_height
        self.channels = 3
        self.folder = folder
        self.one_hot_label= False
        self.data_aug = False
        self.coords = coords
        self.data = data
        self.gpu = gpu
        self.lane_v2 = lane_v2

    def download_data(self):
        if not self.data:
            fromlon, fromlat, tolon, tolat = self.coords
            ZOOM_LEVEL = 20
            fromx,fromy = lonlat2xy(fromlon, fromlat, ZOOM_LEVEL)
            tox,toy = lonlat2xy(tolon, tolat, ZOOM_LEVEL)
            fromx,fromy, tox,toy = list(map(int, [fromx,fromy, tox,toy]))
            sat_image = merge_tiles(get_sat_image(fromx,fromy, tox,toy,ZOOM_LEVEL, data_source),fromx,fromy, tox,toy)

    def DetectLines_v2():

        with closing(Pool(processes=30)) as pool:
            pool.map(getLines, (files))
            pool.terminate()


    def DetectLines(self):
#         os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu)


        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu)
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        set_session(tf.Session(config=config))

#         os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu)

        BigImg = self.folder + 'sat_original.png'
        BigLane = self.folder + 'final_marks.png'
        bigImg = imread(BigImg, -1)
        bigLane = imread(BigLane, -1)  + self.lane_v2
        bigLane_gray = imread(BigLane, 0)  + self.lane_v2[:, :, 0]

        bigLine = np.zeros_like((bigImg))

        checkpoint_name = "GAN_v4_label_normal_lanenet2/"
        model_name = 'lanenet2'
        checkpoint_dir_root = "/home/beemap/Documents/cesar-workspace/Segmentation/Keras_Code/keras3/checkpoints/"
        checkpoint_dir = checkpoint_dir_root + checkpoint_name
        print("Loading Model in .... ", checkpoint_dir)
        json_file = open(checkpoint_dir + model_name + ".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights("%s/%s_weights_90.h5"%(checkpoint_dir, model_name))


        for r in range(0, bigImg.shape[0], 256):
            for c in range(0, bigImg.shape[1], 256):
                img = bigImg[r:r+256, c:c+256, :3]
        #         h, w, _ = img.shape
                tmp = img[:]
                lane = bigLane[r:r+256, c:c+256, :3]
                img = np.float32(img)/255.0
                lane = np.float32(lane)/255.0

        #         input_image_gray = data_util.get_image(image,do_aug=[],gray=True, change=False)
                input_image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                w1, w2, w3, w4 = lanenet_wavelet(input_image_gray)
                w1 = np.expand_dims(w1, axis=0)
                w2 = np.expand_dims(w2, axis=0)
                w3 = np.expand_dims(w3, axis=0)
                w4 = np.expand_dims(w4, axis=0)
                lane_gray = bigLane_gray[r:r+256, c:c+256]

                hough = hough_lines(lane_gray, rho=1, min_line_len=10,threshold=40, max_line_gap=300)/255.0
                mask = model.predict([np.expand_dims(img, axis=0), np.expand_dims(lane, axis=0), w1, w2, w3, w4], batch_size=None, verbose=0, steps=None)

                mask = np.round(mask[0, :, :, 0]).astype(int)
                seg = np.zeros((256, 256, 3))

                seg[:, :, 0] += ((mask[:, :] == 1) * (255)).astype('uint8')
                seg[:, :, 1] += ((mask[:, :] == 1) * ( 255)).astype('uint8')
                seg[:, :, 2] += ((mask[:, :] == 1) * ( 255)).astype('uint8')

                bigLine[r:r+256, c:c+256, :] = seg
        K.clear_session()
        name = 'BigLine.png'
        new_dir = self.folder
        dest = new_dir + name
        cv2.imwrite(dest, bigLine)
