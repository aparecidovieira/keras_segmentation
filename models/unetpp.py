import keras
from keras.layers import *
from keras.activations import *
from keras.models import *
from keras import backend as K

def convBlock(inputs, n_filters, kernel=3, strides=1):
    net = Conv2D(n_filters, kernel_size=[kernel, kernel], activation = 'relu', strides=1, padding ='same')(inputs)
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

def desconv(inputs, n_filters):
    #def upBlock(inputs, n_filters, rate=2):
    up = UpSampling2D(2)(inputs)
    net =  convBlock(up, n_filters, kernel=2)
    return net

def build_unetpp(input_size, one_hot_label=False):

    input_layer = Input(input_size)
    conv1_1 = convBlock(input_layer, 32)
    pool1 = pool(conv1_1, [2, 2], stride=[2, 2], pooling_type='MAX')

    conv2_1 = convBlock(pool1, 64)
    pool2 = pool(conv2_1, [2, 2], stride=[2, 2], pooling_type='MAX')

    up1_1 = desconv(conv2_1, 32)
    conv1_2 = concat(up1_1, conv1_1)
    conv1_2 = convBlock(conv1_2, 32)

    conv3_1 = convBlock(pool2, 128)
    pool3 = pool(conv3_1, [2, 2], stride=[2, 2], pooling_type='MAX')

    up2_1 = desconv(conv3_1, 64)
    conv2_2 = concat(up2_1, conv2_1)
    conv2_2 = convBlock(conv2_2, 64)

    conv4_1 = convBlock(pool3, 256)
    pool4 = pool(conv4_1, [2, 2], stride=[2, 2], pooling_type='MAX')

    up3_1 = desconv(conv4_1, 128)
    conv3_2 = concat(up3_1, conv3_1)
    conv3_2 = convBlock(conv3_2, 128)

    up1_2 = desconv(conv2_2, 32)
    conv1_3 = Concatenate(-1)([conv1_1, conv1_2, up1_2])
    conv1_3 = convBlock(conv1_3, 32)

    up2_2 = desconv(conv3_2, 64)
    conv2_3 = Concatenate(-1)([conv2_1, up2_2, conv2_2])
    conv2_3 = convBlock(conv2_3, 64)

    up1_3 = desconv(conv2_3, 32)
    conv1_4 = Concatenate(-1)([conv1_1, conv1_2, up1_3, conv1_3])
    conv1_4 = convBlock(conv1_4, 32)

    conv5 = convBlock(pool4, 512)
    #conv5 = fun.FPABlock(conv5, n_filters=512)
    # Decoder

    desc1 = desconv(conv5, 256)
    desc1 = concat(desc1, conv4_1)
    desc1 = convBlock(desc1, 256)

    desc2 = desconv(desc1, 128)
    desc2 = Concatenate(-1)([desc2, conv3_1, conv3_2])
    desc2 = convBlock(desc2, 128)

    desc3 = desconv(desc2, 64)
    desc3 = Concatenate(-1)([desc3, conv2_1, conv2_2, conv2_3])
    desc3 = convBlock(desc3, 64)

    desc4 = desconv(desc3, 32)
    desc4 = Concatenate(-1)([desc4, conv1_1, conv1_2, conv1_3, conv1_4])
    net = convBlock(desc4, 32)

    if one_hot_label:
        net = Conv2D(2, kernel_size=1, strides=1, activation='relu', padding='same')(net)
        net = Reshape((2, input_size[0] * input_size[1]))(net)
        net = Permute((2, 1))(net)
        net = Activation('softmax')(net)
    else:
        net = Conv2D(1, kernel_size=1, strides=1, activation='sigmoid', padding='same')(net)

    model = Model(inputs=input_layer, outputs=net)

    return model
