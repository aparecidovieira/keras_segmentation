import keras
from keras.layers import *
from keras.activations import *
from keras.models import *
from keras import backend as K

def convBlock(inputs, n_filters, kernel=3, stride=1, activation='relu'):
    net = Conv2D(n_filters, kernel_size=[kernel, kernel], activation = activation, strides=1, padding ='same')(inputs)
    return net

def MultiResBlock(inputs, n_filters = 64, kernel=3, strides=[1, 1]):
    #alpha = 1.67
    #W = alpha * n_filters

    shortcut = inputs

    shortcut = convBlock(inputs, n_filters, kernel=1, stride=strides)

    n_filters1 = n_filters//3
    n_filters2 = n_filters - n_filters1 * 2


    conv3x3 = convBlock(inputs, n_filters1)
    conv5x5 = convBlock(conv3x3, n_filters1)
    conv7x7 = convBlock(conv5x5, n_filters2)

    out = Concatenate(-1)([conv3x3, conv5x5, conv7x7])
    out = add(shortcut, out)
    net = Activation('relu')(out)

    return net

def conv1x1(inputs, n_filters):
    net = Conv2D(n_filters, kernel_size=1, activation = None, strides=1,padding ='same')(inputs)
    return net

def resPath(inputs, n_filters, b=3):

    shortcut = inputs
    shortcut = convBlock(shortcut, n_filters, kernel=1)

    out = convBlock(inputs, n_filters)
    out = add(shortcut, out)
    out = Activation('relu')(out)

    for _ in range(b-1):
        shortcut = out
        shortcut = convBlock(shortcut, n_filters, kernel=1)

        out = convBlock(out, n_filters)
        out = add(shortcut, out)
        out = Activation('relu')(out)

    return out

def DecoderBlock(inputs, skip, n_filters):
    up_conv = transBlock(inputs, n_filters)
    merge = concat(up_conv, skip)
    net = MultiResBlock(merge, n_filters)

    return net

def transBlock(inputs, n_filters):
    net = Conv2DTranspose(n_filters, kernel_size=[2, 2], strides=2, activation='relu')(inputs)
    return net

def add(tensor_a, tensor_b):
    return Add()([tensor_a, tensor_b])

def pool(inputs, p, stride=[2, 2], pooling_type='MAX', padding='same'):
    pll = MaxPooling2D(pool_size=p, strides=stride, padding=padding)(inputs)
    return pll

def concat(input_A, input_B, axis=-1):
    net = Concatenate(axis)([input_A, input_B])
    return net

def Upsampling(inputs, n_filters=32):
    #def upBlock(inputs, n_filters, rate=2):
    up = UpSampling2D(2)(inputs)
    net =  convBlock(up, n_filters, kernel=2)
    return net

def build_multiUnet(input_shape, one_hot_label=False):
    #Downsampling Path
    n_filters = 64
    inputs = Input(input_shape)

    net = MultiResBlock(inputs, n_filters)
    pool1 = pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')
    skip1 = resPath(net, n_filters, b=4)

    net = MultiResBlock(pool1, 2 * n_filters)
    pool2 = pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')
    skip2 = resPath(net, 2 * n_filters, b=3)

    net = MultiResBlock(pool2, 4 *  n_filters)
    pool3 = pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')
    skip3 = resPath(net, 4 * n_filters, b=2)

    net = MultiResBlock(pool3, 8 * n_filters)
    pool4 = pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')
    skip4 = resPath(net, 8 * n_filters, b=1)

    #bridge
    net = MultiResBlock(pool4, 16 * n_filters)

    #Up Sampling Path
    up4 = DecoderBlock(net, skip4, 8 * n_filters)
    up3 = DecoderBlock(up4, skip3, 4 * n_filters)
    up2 = DecoderBlock(up3, skip2, 2 * n_filters)
    net = DecoderBlock(up2, skip1,  n_filters)

    if one_hot_label:
        net = Conv2D(2, kernel_size=1, strides=1, activation='relu', padding='same')(net)
        net = Reshape((2, input_size[0] * input_size[1]))(net)
        net = Permute((2, 1))(net)
        net = Activation('softmax')(net)
    else:
        net = Conv2D(1, kernel_size=1, strides=1, activation='sigmoid', padding='same')(net)

    model = Model(inputs=inputs, outputs=net)

    return model
