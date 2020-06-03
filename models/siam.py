import keras
from keras.layers import *
from keras.activations import *
from keras.models import *
from keras import backend as K

axis = -1
freeze_bn = True

def convBlock(inputs, n_filters, kernel=3, strides=1):
    net = Conv2D(n_filters, kernel_size=[kernel, kernel], activation = None, strides=1, padding ='same')(inputs)
    net = Batch_Act(net)
    return net

def conv1x1(inputs, n_filters):
    net = Conv2D(n_filters, kernel_size=1, activation = None, strides=1,padding ='same')(inputs)
    return net

def shortcut(net, res, n_filters, not_equal=False):
    if not_equal:
        res = conv1x1(res, n_filters)
    net = Add()([net, res])
    return net

def downBlock(inputs, n_filters):
    net = MaxPooling2D(kernel=(2, 2), strides=(2, 2))(inputs)
    net = Batch_Act(net)
    return net

def Batch_Act(inputs):
    net = BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn)(inputs)
    net = Activation('relu')(net)
    return net

def res_block(inputs, n_filters=64, kernel_size=3):
    net = BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn)(inputs)
    net = Activation('relu')(net)
    net = convBlock(net, n_filters)
    net = Conv2D(n_filters, kernel_size=3, activation = None, strides=1, padding ='same')(net)
    return net


def resnet_block(inputs, n_filters, blocks=3):
    for i in range(blocks):
        residual = resBlock(inputs, n_filters)
        if i == 0:
            inputs = shortcut(inputs, n_filters)
        inputs = Add()([inputs, residual])
    return net

def transBlock(inputs, n_filters):
    net = Conv2DTranspose(n_filters, kernel_size=[2, 2], strides=1, activation='relu')(inputs)
    return net

def add(tensor_a, tensor_b, tensor_c = None):
    net_1 = Add()([tensor_a, tensor_b])
    if tensor_c:
        net_2 = Add()([tensor_c, net_1])
    return net_2

def Upsampling(inputs, size=(2, 2)):
    net = UpSampling2D(size)(inputs)
    return net

def concat(tensor_a, tensor_b):
    net = Concatenate(axis)([tensor_a, tensor_b])
    return net

def decoder_block(inputs, n_filters, kernel_size=3):
    net = convBlock(inputs, n_filters)
    net = convBlock(net, n_filters)
    net = Upsampling(net)
    return net

def model_gen(inputs):
    n_filters = 64
    net_1 = encoderBlock(inputs, n_filters)
    skip_1 = pool(net_1, [2, 2], stride=[2, 2], pooling_type='MAX')

    net_2 = encoderBlock(skip_1, n_filters)
    skip_2 = pool(net_2, [2, 2], stride=[2, 2], pooling_type='MAX')

def groupConv(inputs, n_filters):
    pass

def eespBlock(inputs, n_filters):
    net = groupConv(inputs, n_filters)


def siam(input_size, one_hot_label=False, freeze=True):
    inputs = Input(input_size)
    ref = inputs[:, :, :, :3]
    target = inputs[:, :, :, 3:]
    n_filters = 64

    # ref = BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn)(ref)
    # target = BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn)(target)


    net_1_1 = convBlock(ref, n_filters)
    skip_1_1 = downBlock(net_1_1, n_filters)

    net_2_1 = convBlock(target, n_filters)
    skip_2_1 = downBlock(net_2_1, n_filters)

    res_1_1 = resnet_block(skip_1_1, n_filters)
    res_2_1 = resnet_block(skip_2_1, n_filters)

    res_1_2 = resnet_block(res_1_1, 2 * n_filters, blocks=4)
    res_2_2 = resnet_block(res_2_1, 2 * n_filters, blocks=4)

    res_1_3 = resnet_block(res_1_2, 4 * n_filters, blocks=6)
    res_2_3 = resnet_block(res_2_2, 4 * n_filters, blocks=6)

    res_1_4 = resnet_block(res_1_3, 8 * n_filters)
    res_2_4 = resnet_block(res_2_3, 8 * n_filters)

    res_1_5 = Batch_Act(res_1_4)
    res_2_5 = Batch_Act(res_2_4)

    # Reference Architecture

    dec_4 = add(res_1_5, res_2_5)
    up_4 = Upsampling(dec_4)
    dec_4 = add(res_1_4, res_2_4, dec_4)

    dec_3 = decoderBlock(dec_4, 4 * n_filters)
    dec_3 = add(res_1_3, res_2_3, dec_3)

    dec_2 = decoderBlock(dec_3, 2 * n_filters)
    dec_3 = add(res_1_2, res_2_2, dec_2)

    dec_1 = decoderBlock(dec_2, n_filters)
    dec_1 = add(res_1_1, res_2_1, dec_1)

    net = convBlock(dec_1, n_filters)
    net = convBlock(net, n_filters)

    if one_hot_label:
        net = Conv2D(2, kernel_size=1, strides=1, activation='relu', padding='same')(net)
        net = Reshape((2, input_size[0] * input_size[1]))(net)
        net = Permute((2, 1))(net)
        net = Activation('softmax')(net)
    else:
        net = Conv2D(1, kernel_size=1, strides=1, activation='sigmoid', padding='same')(net)

    model = Model(inputs=inputs, outputs=net)

    return net
