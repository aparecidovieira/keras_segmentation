import keras
from keras.layers import *
from keras.activations import *
from keras.models import *
from keras import backend as K

def convBlock(inputs, n_filters, kernel=3, stride=1):
    net = Conv2D(n_filters, kernel_size=[kernel, kernel], activation = 'relu', strides=1, padding ='same')(inputs)
    return net

def ConvBlock(inputs, n_filters0 = 64, n_filters1 = 32,  kernel_size0=3, kernel_size1=2, strides=[1, 1]):

    net = convBlock(inputs, n_filters0, kernel=kernel_size0, stride=strides)
    net = convBlock(net, n_filters1, kernel=kernel_size1, stride=strides)

    return net

def conv1x1(inputs, n_filters):
    net = Conv2D(n_filters, kernel_size=1, activation = None, strides=1,padding ='same')(inputs)
    return net

def shortcut(net, res, n_filters, not_equal=False):
    if not_equal:
        res = conv1x1(res, n_filters)
    net = Add()([net, res])
    return net

def encoder_block(inputs, n_filters, blocks=3):
    for _ in range(blocks):
        inputs = convBlock(inputs, n_filters)
    return net

def transBlock(inputs, n_filters):
    net = Conv2DTranspose(n_filters, kernel_size=[2, 2], strides=1, activation='relu')(inputs)
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

def build_deep(input_shape, one_hot_label=False):
    #Downsampling Path
    inputs = Input(input_shape)
    net = convBlock(inputs, 64, kernel=3)
    net = ConvBlock(net)

    skip0 = net#convBlock(inputs, 32, kernel=2)

    net = pool(skip0, [2, 2], stride=[2, 2], pooling_type='MAX')
    skip = net

    net = ConvBlock(net)

    skip1 = concat(net, skip)
    net = pool(skip1, [2, 2], stride=[2, 2], pooling_type='MAX')
    add1 = net

    #256
    net = ConvBlock(net)
    skip2 = concat(net, add1)
    net = pool(skip2, [2, 2], stride=[2, 2], pooling_type='MAX')
    add2 = net

    #128
    net = ConvBlock(net)
    skip3 = concat(net, add2)
    net = pool(skip3, [2, 2], stride=[2, 2], pooling_type='MAX')
    add3 = net

    #64
    net = ConvBlock(net)
    skip4 = concat(net, add3)
    net = pool(skip4, [2, 2], stride=[2, 2], pooling_type='MAX')
    add4 = net

    #32
    net = ConvBlock(net)
    skip5 = concat(net, add4)
    net = pool(skip5, [2, 2], stride=[2, 2], pooling_type='MAX')
    add5 = net

    #16
    net = ConvBlock(net)
    skip6 = concat(net, add5)
    net = pool(skip6, [2, 2], stride=[2, 2], pooling_type='MAX')


    #Up Sampling Path

    up1 = Upsampling(net)
    net = concat(up1, skip6)
    net = ConvBlock(net, kernel_size1=3)
    net = concat(net, up1)

    #add8 = net

    up2 = Upsampling(net)
    net = concat(up2, skip5)
    net = ConvBlock(net, kernel_size1=3)
    #net = tf.add(add8, net)
    net = concat(net, up2)
    #net = Upsampling(net)
    #add9 = net

    up3 = Upsampling(net)
    net = concat(up3, skip4)
    net = ConvBlock(net, kernel_size1=3)
    net =concat(net, up3)


    up4 = Upsampling(net)
    net = concat(up4,skip3)
    net = ConvBlock(net, kernel_size1=3)
    net = concat(net, up4)

    up5 = Upsampling(net)
    net = concat(up5, skip2)
    net = ConvBlock(net, kernel_size1=3)
    net = concat(net, up5)

    up6 = Upsampling(net)
    net = concat(up6, skip1)
    net = ConvBlock(net, kernel_size1=3)
    net = concat(net, up6)

    net = Upsampling(net)

    if one_hot_label:
        net = Conv2D(2, kernel_size=1, strides=1, activation='relu', padding='same')(net)
        net = Reshape((2, input_size[0] * input_size[1]))(net)
        net = Permute((2, 1))(net)
        net = Activation('softmax')(net)
    else:
        net = Conv2D(1, kernel_size=1, strides=1, activation='sigmoid', padding='same')(net)

    model = Model(inputs=inputs, outputs=net)

    return model
