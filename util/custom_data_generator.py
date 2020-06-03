
import numpy as np
import cv2, sys
import itertools
#from pilutil import *
from scipy.misc import imread
# sys.path.append('..')
from  models.common import lanenet_wavelet
from keras.utils import to_categorical
#from scipy.misc import imread
#from matplotlib import imread

import os
import sys
import random


IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGE_CHANNEL = 3

def compute_class_weights(labels_dir, label_values=[(0,0,0),(0,255,0)]):
    '''
    Arguments:
        labels_dir(list): Directory where the image segmentation labels are
        num_classes(int): the number of classes of pixels in all images

    Returns:
        class_weights(list): a list of class weights where each index represents each class label and the element is the class weight for that label.

    '''
    image_files = [os.path.join(labels_dir, file) for file in os.listdir(labels_dir) if file.endswith('.png')]
    num_classes = len(label_values)
    class_pixels = np.zeros(num_classes)
    total_pixels = 0.0
    for n in range(len(image_files)):
        image = imread(image_files[n], mode="RGB")
        for index, colour in enumerate(label_values):
            class_map = np.all(np.equal(image, colour), axis = -1)
            class_map = class_map.astype(np.float32)
            class_pixels[index] += np.sum(class_map)

        print("\rProcessing image: " + str(n) + " / " + str(len(image_files)), end="")
        sys.stdout.flush()
    total_pixels = float(np.sum(class_pixels))
    index_to_delete = np.argwhere(class_pixels==0.0)
    class_pixels = np.delete(class_pixels, index_to_delete)
    class_weights = total_pixels / class_pixels
    class_weights = class_weights / np.sum(class_weights)
    return class_weights

def random_aug():
    brightness = random.randint(0, 1)
    flip = random.randint(0, 1)
    zoom = random.randint(0, 1)
    data_aug = {"brightness": brightness,"flip":flip,"zoom":zoom}
    return data_aug

def perform_aug(image,do_aug,is_mask=False):
    for key, value in do_aug.items():
        if key == "brightness" and value == 1:
            if not is_mask:
                image = random_brightness(image)
        if key == "flip" and value == 1:
            image = flip_image(image)
        if key == "zoom" and value == 1:
            image = zoom_image(image)
    return image

def flip_image(image):
    return cv2.flip(image, 1)

def random_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    rand = random.uniform(0.3, 1.0)
    hsv[:, :, 2] = rand*hsv[:, :, 2]
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return new_img

def zoom_image(image):
    zoom_pix = random.randint(0, 10)
    zoom_factor = 1 + (2*zoom_pix)/IMAGE_HEIGHT
    image = cv2.resize(image, None, fx=zoom_factor,fy=zoom_factor,interpolation=cv2.INTER_AREA)
    top_crop = (image.shape[0] - IMAGE_HEIGHT)//2
    left_crop = (image.shape[1] - IMAGE_WIDTH)//2
    image = image[top_crop: top_crop+IMAGE_HEIGHT,left_crop: left_crop+IMAGE_WIDTH]
    return image


def roundColor_2D(img):
    img[img > 70 ] = 255
    img[img <= 70 ] = 0
    return(img)

def changelabels(img,type):
    '''
    This fucntion used for the conversion from labels [0,1,2] to rgb colors.
    Takes img and type [(0,1,2)-> rgb or rgb->(0,1,2)]
    '''
    if type == "1d2rgb": # means [0,1,2] -> RGB
        colors = [  (0,0,0),(255,255,255) ,(0,255,0) ]
        seg_img = np.zeros( ( img.shape[0] ,img.shape[1] , 3  ), dtype=np.uint8 )

        for c in range(3):
            seg_img[:,:,0] += ( (img[:,: ] == c )*( colors[c][0] )).astype('uint8')
            seg_img[:,:,1] += ((img[:,: ] == c )*( colors[c][1] )).astype('uint8')
            seg_img[:,:,2] += ((img[:,: ] == c )*( colors[c][2] )).astype('uint8')

        #seg_img = cv2.resize(seg_img  , (256 , 256 ))
        seg_img = roundColor_2D(seg_img)
        return seg_img

    if type == "rgb21d":  # means RGB -> [0,1,2]
        #img = roundColor_2D(img)
        palette = {(0,0,0):0 ,(255,255,255):1, (0,255,0):1  }
        arr_2d = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for c, i in palette.items():
            m = np.all(img == np.array(c).reshape(1, 1, 3), axis=2)
            arr_2d[m] = i
        return arr_2d


def get_image(path, do_aug, gray=False, change=False):
    if gray:
        img =cv2.imread(path, 0) #cv2.COLOR_BGR2RGB
    else:
        img =cv2.imread(path,-1)[:, :, :3] #cv2.COLOR_BGR2RGB
    #print(img.shape)
    if change:
        img1 = img[:, :256, :]
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

        img2 = img[:, 256:, :]
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        new_img = np.zeros((256, 256, 6))
        new_img[:, :, :3] = img1
        new_img[:, :, 3:] = img2
        img = new_img
            #img_n = np.zeros((256, 256, 3))
    #img = cv2.resize(img, (256, 256))
    if not do_aug == [] and gray:
        img = perform_aug(img,do_aug)
    # img_n[:, :, :3] = img[:, 0:256, :]
    # img_n[:, :, 3:6] = img[:, 256:512, :]

    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # kernel_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    # low1 = cv2.filter2D(img, -1, kernel_sharp)
    # img = cv2.bilateralFilter(low1, 3, 75, 75)
    img = np.float32(img) / 255.0
    return img

def get_mask(path,one_hot_label,do_aug):
    mask = cv2.imread(path)
    if type(mask) != np.ndarray:
        mask = np.zeros((256, 256, 3))
    mask = roundColor_2D(mask)
    #mask = cv2.resize(mask, (256, 256))

    if not do_aug == []:
        mask = perform_aug(mask,do_aug,True)

    mask = changelabels(mask,"rgb21d")
    if one_hot_label:
        label = to_categorical(mask,2)
#         label = np.zeros(( 256 , 256 , 2 ))
#         for c in range(2):
#             label[: , : , c ] = (mask == c ).astype(int)
        label = np.reshape(label, (256 * 256, 2)).astype(int)
    else:
        label= np.expand_dims(mask,axis=-1)
    return label

def image_generator(files,images_path,masks_path, batch_size = 5, wavelet=True, one_hot_label = False, data_aug= False, change=False, wavelet_=True):
    zipped = itertools.cycle(zip(files))

    # Read in each input, perform preprocessing and get labels
    while True:
        batch_input = []
        batch_input2 = []

        batch_waves = []
        batch_output = []
        waves1 = []
        waves2 = []
        waves3 = []
        waves4 = []

        for i in range(batch_size):
            file_path = next(zipped)[0]

            #data augmentation
            do_aug = []
            if data_aug:
                do_aug = random_aug()

            input = get_image(images_path + file_path,do_aug, change=change)
            # print(input.shape)
            output = get_mask(masks_path + file_path,one_hot_label,do_aug)
            if wavelet_:
                input_gray = get_image(images_path + file_path, do_aug=do_aug, gray=True)
                # input_gray = cv2.cvtColor(input[:, :256, :], cv2.COLOR_BGR2GRAY)

                coeffs = lanenet_wavelet(input_gray[:, :256])
                w1, w2, w3, w4 = coeffs
                waves1.append(w1)
                waves2.append(w2)
                waves3.append(w3)
                waves4.append(w4)
            if change:
                batch_input.append(input[:, :, :3])
                batch_input2.append(input[:, :, 3:])
            else:
                batch_input.append(input)
            #batch_waves.append(coeffs)
            batch_output.append(output.astype(int))
        # print(batch_input)
        batch_x = np.array(batch_input)
        if change:
            batch_x2 = np.array(batch_input2)
        if wavelet_:
            waves1 = np.array( waves1)
            waves2 = np.array( waves2)
            waves3 = np.array( waves3)
            waves4 = np.array( waves4)

        # batch_waves = np.array( batch_waves)
        batch_y = np.array( batch_output)
        yield ([batch_x, waves1, waves2, waves3, waves4], batch_y)

        # yield ([batch_x, batch_x2, waves1, waves2, waves3, waves4], batch_y ) if change else ([batch_x, waves1, waves2, waves3, waves4], batch_y)
