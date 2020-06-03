import keras
from keras.layers import *
from keras.activations import *
from keras.models import *
from keras import backend as K

def convBlock(inputs, n_filters, kernel=3, strides=1, dilation=1):
    net = Conv2D(n_filters, kernel_size=[kernel, kernel], activation = 'relu', strides=strides, dilation_rate=dilation, padding ='same')(inputs)
    return net

def ConvBlock(inputs, n_filters0 = 64, n_filters1 = 32,  kernel_size0=3, kernel_size1=2, strides=[1, 1]):

    net = convBlock(inputs, n_filters0, kernel=kernel_size0, stride=strides)
    net = convBlock(net, n_filters1, kernel=kernel_size1, stride=strides)

    return net

def conv1x1(inputs, n_filters, activation=None, strides=1):
    net = Conv2D(n_filters, kernel_size=1, activation = activation, strides=strides, padding ='same')(inputs)
    return net

def shortcut(net, res, n_filters, not_equal=False):
    if not_equal:
        res = conv1x1(res, n_filters)
    net = Add()([net, res])
    return net

def decoderBlock(inputs, n_filters, blocks=3):
    net = convBlock(inputs, n_filters)
    net = transBlock(net, n_filters, kernel_size=[2, 2])
    net = convBlock(net, n_filters)

    return net

def transBlock(inputs, n_filters, kernel_size=[2, 2]):
    net = Conv2DTranspose(n_filters, kernel_size=kernel_size, strides=2, activation='relu', padding='same')(inputs)
    return net

def add(tensor_a, tensor_b):
    return Add()([tensor_a, tensor_b])

def pool(inputs, p=2, stride=[2, 2], pooling_type='MAX', padding='same'):
    pll = MaxPooling2D(pool_size=[p, p], strides=stride, padding=padding)(inputs)
    return pll

def concat(input_A, input_B, axis=-1):
    net = Concatenate(axis)([input_A, input_B])
    return net

def Upsampling(inputs, n_filters=512, size=(2, 2)):
    #def upBlock(inputs, n_filters, rate=2):
    up = UpSampling2D(size)(inputs)
    net =  convBlock(up, n_filters, kernel=2)
    return net

def resnet34Block(inputs, n_filters, blocks=3, kernel=3, strides=1):
    for i in range(blocks):
        if i == 0:
            shortcut_ = conv1x1(inputs, n_filters, activation='relu')
        else:
            shortcut_ = inputs
        for _ in range(2):
            inputs = convBlock(inputs, n_filters, kernel=kernel, strides=strides)
        inputs = add(inputs, shortcut_)
    return inputs

def dilateConv(inputs, n_filters, dilation=1):
    net = convBlock(inputs, n_filters, dilation=dilation)
    return net

def ResidualBlock(inputs, n_filters):
    short = inputs
    net = convBlock(inputs, n_filters)
    net = convBlock(net, n_filters)
    net = convBlock(net, n_filters)
    short = convBlock(inputs, n_filters, kernel=1)
    out = add(net, short)
    return out


def DCABlock(inputs, n_filters, d3=3, d5=5):
    dilate1 = dilateConv(inputs, n_filters)
    dilate2 = conv1x1((dilateConv(inputs, n_filters, dilation=d3)), n_filters)
    dilate3 = dilateConv(dilateConv(inputs, n_filters), n_filters, dilation=d3)
    dilate3 = conv1x1(dilate3, n_filters)
    dilate4 = dilateConv(dilateConv(inputs, n_filters), n_filters, dilation=d3)
    dilate4 = conv1x1(dilateConv(dilate4, n_filters, dilation=d5), n_filters)

    out = Add()([inputs, dilate1, dilate2, dilate3, dilate4])
    return out

def SPPBlock(inputs, n_filters=1):
    pool1 = pool(inputs)
    pool2 = pool(inputs, p=4, stride=[4, 4])
    pool3 = pool(inputs, p=8, stride=[8, 8])
    pool4 = pool(inputs, p=16, stride=[16, 16])
    shape= K.int_shape(inputs)
    #out = Concatenate(-1)([pool1, pool2, pool3, pool4])
    h, w = shape[1], shape[2]
    #print(h, w)
    pool1 = Upsampling(convBlock(pool1, n_filters), size=(2, 2))
    pool2 = Upsampling(convBlock(pool2, n_filters), size=(4, 4))
    pool3 = Upsampling(convBlock(pool3, n_filters), size=(8, 8))
    pool4 = Upsampling(convBlock(pool4, n_filters), size=(16, 16))

    out = Concatenate(-1)([pool1, pool2, pool3, pool4])
    return out

# def get_weights(shape):
#     return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.01))

def DilConv(inputs, n_filters, kernel_size=[3, 3], rate=1):
    # shape = [kernel_size[0], kernel_size[1], n_filters, n_filters]
    net = convBlock(inputs, n_filters, dilation=rate, kernel=kernel_size[0])

    return net

def desconv(inputs, n_filters, stride=[2, 2]):
    inputs = convBlock(inputs, n_filters, kernel=1)
    net = transBlock(inputs, n_filters, kernel_size=[3, 3])
    net = convBlock(net, n_filters, kernel=1)
    return net


def DACBlock(inputs, n_filters):
    conv_1 = DilConv(inputs, n_filters, rate=1)

    conv_2 = DilConv(inputs, n_filters, rate=3)
    conv_2 = DilConv(conv_2, n_filters, kernel_size=[1, 1], rate=1)

    conv_3 = DilConv(inputs, n_filters, rate=1)
    conv_3 = DilConv(conv_3, n_filters, rate=3)
    conv_3 = DilConv(conv_3, n_filters, kernel_size=[1, 1], rate=1)

    conv_4 = DilConv(inputs, n_filters, rate=1)
    conv_4 = DilConv(conv_4, n_filters, rate=3)
    conv_4 = DilConv(conv_4, n_filters, rate=5)
    conv_4 = DilConv(conv_4, n_filters, kernel_size=[1, 1], rate=1)

    out = Add()([conv_1, conv_2, conv_3, conv_4])
    return out

def RPMBlock(inputs, n_filters=512):
    pool2 = pool(inputs)
    pool3 = pool(inputs, p=3)
    pool5 = pool(inputs, p=5)
    pool6 = pool(inputs, p=6)

    pool2 = convBlock(pool2, n_filters, kernel=1)
    pool3 = convBlock(pool3, n_filters, kernel=1)
    pool5 = convBlock(pool5, n_filters, kernel=1)
    pool6 = convBlock(pool6, n_filters, kernel=1)

    pool2 = Upsampling(pool2)
    pool3 = Upsampling(pool3)
    pool5 = Upsampling(pool5)
    pool6 = Upsampling(pool6)

    out = Concatenate(-1)([inputs, pool2, pool3, pool5, pool6])
    return out

def build_ce(input_size, one_hot_label=False):
    inputs = Input(input_size)
    net = ResidualBlock(inputs, 64)
    skip1 = net
    net = pool(net, p=2, stride=[2, 2], pooling_type='MAX')#128

    net = ResidualBlock(net, 128)
    skip2 = net
    net = pool(net, p=2, stride=[2, 2], pooling_type='MAX')#64

    net = ResidualBlock(net, 256)
    skip3 = net
    net = pool(net, p=2, stride=[2, 2], pooling_type='MAX')#32

    net = ResidualBlock(net, 512)
    skip4 = net
    net = pool(net, p=2, stride=[2, 2], pooling_type='MAX')#16

    net = ResidualBlock(net, 1024)
    net = DACBlock(net, 1024)
    net = ResidualBlock(net ,1024)
    net = RPMBlock(net)
    #BRIDGE

    net = ResidualBlock(net, 512)
    #net = FPABlock(net, 1024)
    #net = tf.nn.dropout(net, keep_prob)

    net = desconv(net, 512)
    net = add(net, skip4)
    net = ResidualBlock(net, 512)

    net = desconv(net, 256)
    net = add(net, skip3)
    net = ResidualBlock(net, 256)

    net = desconv(net, 128)
    net = add(net, skip2)
    net = ResidualBlock(net, 128)

    net = desconv(net, 64)
    net = add(net, skip1)
    net = ResidualBlock(net, 64)

    if one_hot_label:
        net = Conv2D(2, kernel_size=1, strides=1, activation='relu', padding='same')(net)
        net = Reshape((2, input_size[0] * input_size[1]))(net)
        net = Permute((2, 1))(net)
        net = Activation('softmax')(net)
    else:
        net = Conv2D(1, kernel_size=1, strides=1, activation='sigmoid', padding='same')(net)

    model = Model(inputs=inputs, outputs=net)

    return model
