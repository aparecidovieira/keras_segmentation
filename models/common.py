####
import keras
import keras.backend as K
from keras.layers import *
from keras.activations import *
from keras.models import *
from pywt import wavedec2
import pywt
import numpy as np

def wavelet(inputs, level=4):
    coeffs = wavedec2(inputs, 'db1', level=level)
    return coeffs

def multiply(tensor_a,tensor_b):
    return Multiply()([tensor_a,tensor_b])

def lanenet_wavelet(inputs):
    # if
    coeffs = wavelet(inputs)
    cA, C4, C3, C2, C1 = coeffs
    # cH4, cV4, cD4 = C4
    # cH3, cV3, cD3 = C3
    # cH2, cV2, cD2 = C2
    # cH1, cV1, cD1 = C1
    w1 = np.stack((C1), axis=-1)
    w2 = np.stack((C2), axis=-1)
    w3 = np.stack((C3), axis=-1)
    w4 = np.stack((C4), axis=-1)
    return w1, w2, w3, w4

def convBlock_v2(inputs, n_filters, kernel=3, strides=1):
    net = Conv2D(n_filters, kernel_size=[kernel, kernel], activation = 'relu', strides=1, padding ='same')(inputs)
    return net

def convBlock(inputs, n_filters, kernel=3, strides=1):
    net = Conv2D(n_filters, kernel_size=[kernel, kernel], activation = None, strides=1, padding ='same')(inputs)
    net = BatchNormalization()(net)
    net = Activation("relu")(net)
    return net

def conv_block(inputs, n_filters, blocks=3):
    for _ in range(blocks):
        inputs = convBlock(inputs, n_filters)
    return inputs

def conv_block_v2(inputs, n_filters, blocks=3):
    for _ in range(blocks):
        inputs = convBlock_v2(inputs, n_filters)
    return inputs

def conv1x1(inputs, n_filters, act=None):
    net = Conv2D(n_filters, kernel_size=1, activation = act, strides=1,padding ='same')(inputs)
    return net

def shortcut(net, res, n_filters, not_equal=False):
    if not_equal:
        res = conv1x1(res, n_filters)
    net = Add()([net, res])
    return net

def encoder_block(inputs, n_filters, blocks=3):
    for _ in range(blocks):
        inputs = convBlock(inputs, n_filters)
    return inputs

def encoder_block_v2(inputs, n_filters, blocks=3):
    for _ in range(blocks):
        inputs = convBlock_v2(inputs, n_filters)
    return inputs


def transBlock(inputs, n_filters):
    net = Conv2DTranspose(n_filters, kernel_size=[2, 2], strides=1, activation='relu')(inputs)
    return net

def sepConvBlock(inputs, n_filters):
    net = SeparableConv2D(n_filters, kernel=(3, 3), activation='relu')(inputs)
    net = conv1x1(net, n_filters, act='relu')
    net = SeparableConv2D(n_filters, kernel=(3, 3), dilation=2, activation='relu')(net)
    net = conv1x1(net, n_filters,act='relu')
    net = SeparableConv2D(n_filters, kernel=(3, 3), dilation=4, activation='relu')(net)
    net = conv1x1(net, n_filters, act='relu')
    return net

def pixelShuffle(inputs):
    inputs = x
    return inputs

def add(tensor_a, tensor_b):
    return Add()([tensor_a, tensor_b])

def pool(inputs, p=[2, 2], stride=[2, 2], pooling_type='MAX', padding='same'):
    pll = MaxPooling2D(pool_size=p, strides=stride, padding=padding)(inputs)
    return pll

def concat(input_A, input_B, axis=-1):
    net = Concatenate(axis)([input_A, input_B])
    return net


def desconv_v3(inputs, n_filters, rate=2):
    net = Conv2DTranspose(n_filters, kernel=[rate, rate], strides=[rate, rate])(inputs)
    return net

def desconv_v2(inputs, n_filters, rate=2):
    #def upBlock(inputs, n_filters, rate=2):
    up = UpSampling2D(rate)(inputs)
    net =  convBlock_v2(up, n_filters, kernel=2)
    return net

def desconv(inputs, n_filters, rate=2):
    #def upBlock(inputs, n_filters, rate=2):
    up = UpSampling2D(rate)(inputs)
    net =  convBlock(up, n_filters, kernel=2)
    return net
