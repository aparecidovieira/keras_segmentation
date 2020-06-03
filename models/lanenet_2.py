###
import keras
from keras.layers import *
import keras.backend as K
import common as C
from common import *
from keras.models import *
axis = -1
freeze_bn = True

def convBlock(inputs, n_filters, kernel=3, strides=1):
    net = Conv2D(n_filters, kernel_size=[kernel, kernel], activation = None, strides=strides, padding ='same')(inputs)
    net = Batch_Act(net)
    return net

def conv1x1(inputs, n_filters, strides=1):
    net = Conv2D(n_filters, kernel_size=1, activation = None, strides=strides,padding ='same')(inputs)
    return net

def shortcut(net, n_filters, strides=1, not_equal=False):
    #if not_equal:
    net = conv1x1(net, n_filters, strides=strides)
    #net = Add()([net, res])
    return net

def downBlock(inputs):
    net = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(inputs)
    # net = Batch_Act(net)
    return net

def Batch_Act(inputs):
    net = BatchNormalization(axis=axis, epsilon=1e-5)(inputs)
    net = Activation('relu')(net)
    return net

def res_block(inputs, n_filters=64, kernel_size=3, strides=1, n_conv=3):
    # net = BatchNormalization(axis=axis, epsilon=1e-5)(inputs)
    # net = Activation('relu')(net)
    for _ in range(n_conv):
        inputs = convBlock(inputs, n_filters, strides=strides)
    # net = Conv2D(n_filters, kernel_size=3, activation = None, strides=1, padding ='same')(net)
    return inputs


def resnet_block(inputs, n_filters, blocks=1, conv1=False):
    for i in range(blocks):

        ### Using strides = 2 instead of maxpooling
        # residual = res_block(inputs, n_filters, strides= (2 if i == 0 else 1))
        inputs = convBlock(inputs, n_filters)
        residual = res_block(inputs, n_filters, strides= 1)
        if i == 0 and conv1:
            inputs = shortcut(inputs, n_filters)
        inputs = Add()([inputs, residual])
    inputs = convBlock(inputs, n_filters)
    return inputs

def transBlock(inputs, n_filters):
    net = Conv2DTranspose(n_filters, kernel_size=[2, 2], strides=1, activation='relu')(inputs)
    return net

def add(tensor_a, tensor_b, tensor_c = None, third=False):
    net_1 = Add()([tensor_a, tensor_b])
    if third:
        net_1 = Add()([tensor_c, net_1])
    return net_1

def Upsampling(inputs, size=(2, 2)):
    net = UpSampling2D(size)(inputs)
    return net

def upBlock(inputs, n_filters, blocks=1):
    net = Upsampling(inputs)
    for _ in range(blocks):
        net = convBlock(net, n_filters, kernel=2)
    return net

def concat(tensor_a, tensor_b, tensor_c=False, third=False):

    net = Concatenate(axis)([tensor_a, tensor_b]) if not third else Concatenate(axis)([tensor_a, tensor_b, tensor_c])

    return net

def build_lanenet2(input_shape,input_shape1=[128, 128, 3],input_shape2=[64, 64, 3],
    input_shape3=[32, 32, 3],input_shape4=[16, 16, 3], one_hot_label=False):
    print(input_shape, 'Input size shape ~')

    inputs1 = Input((input_shape))
    inputs2= Input((input_shape))
    n_filters = 64

    inputs_waves1 = Input((input_shape1))
    inputs_waves2 = Input((input_shape2))
    inputs_waves3 = Input((input_shape3))
    inputs_waves4 = Input((input_shape4))
    #wavelet1, wavelet2, wavelet3, wavelet4 = C.lanenet_wavelet(inputs)

    res_1_1 = resnet_block(inputs1, n_filters)
    skip_1_1 = downBlock(res_1_1)
    skip_1_1 = C.concat(skip_1_1, inputs_waves1)

    res_2_1 = resnet_block(inputs2, n_filters)
    skip_2_1 = downBlock(res_2_1)


    res_1_2 = resnet_block(skip_1_1, 2 * n_filters, blocks=1)
    skip_1_2 = downBlock(res_1_2)
    skip_1_2 = C.concat(skip_1_2, inputs_waves2)


    res_2_2 = resnet_block(skip_2_1, 2 * n_filters, blocks=1)
    skip_2_2 = downBlock(res_2_2)


    res_1_3 = resnet_block(skip_1_2, 4 * n_filters, blocks=1)
    skip_1_3 = downBlock(res_1_3)
    skip_1_3 = C.concat(skip_1_3, inputs_waves3)


    res_2_3 = resnet_block(skip_2_2, 4 * n_filters, blocks=1)
    skip_2_3 = downBlock(res_2_3)


    res_1_4 = resnet_block(skip_1_3, 8 * n_filters)
    skip_1_4 = downBlock(res_1_4)
    skip_1_4 = C.concat(skip_1_4, inputs_waves4)


    res_2_4 = resnet_block(skip_2_3, 8 * n_filters)
    skip_2_4 = downBlock(res_2_4)


    res_1_5 = resnet_block(skip_1_4, 8 * n_filters)
    skip_1_5 = downBlock(res_1_5)
    # skip_1_4 = C.concat(skip_1_4, inputs_waves4)

    res_2_5 = resnet_block(skip_2_4, 8 * n_filters)
    skip_2_5 = downBlock(res_2_5)

    bridge_merge = add(skip_1_5, skip_2_5)
    bridge = resnet_block(bridge_merge, 16 * n_filters)

    # Reference Architecture

    up_4 = upBlock(bridge, 8 * n_filters)
    merge_4 = concat(up_4, skip_1_4, skip_2_4, third=True)
    dec_4 = resnet_block(merge_4, 8 * n_filters)


    up_3 = upBlock(dec_4, 4 * n_filters)
    merge_3 = concat(up_3, skip_1_3, skip_2_3, third=True)
    dec_3 = resnet_block(merge_3, 4 * n_filters)

    up_2 = upBlock(dec_3, 2 * n_filters)
    merge_2 = concat(up_2, skip_1_2, skip_2_2, third=True)
    dec_2 = resnet_block(merge_2, 2 * n_filters)

    up_1 = upBlock(dec_2, n_filters)
    merge_1 = concat(up_1, skip_1_1, skip_2_1, third=True)
    dec_1 = resnet_block(merge_1, n_filters)
    # dec_1 = add(res_1_1, res_2_1, dec_1)

    up_final = upBlock(dec_1, n_filters)

    net = up_final
    #merge1 = C.conv_block(merge1, n_filters)
    if one_hot_label:
        net = Conv2D(2, 1, 1, activation='relu', border_mode='same')(net)
        net = Reshape((2,input_shape[0]*input_shape[1]))(net)
        net = Permute((2,1))(net)
        net = Activation('softmax')(net)
    # model = Model(inputs=[inputs1, inputs2, inputs_waves1, inputs_waves2, inputs_waves3, inputs_waves4], outputs=net)
    else:
        net = Conv2D(1, kernel_size=1, strides=1, activation='sigmoid', padding='same')(net)
    model = Model(inputs=[inputs1, inputs2, inputs_waves1, inputs_waves2, inputs_waves3, inputs_waves4], outputs=net)

    # model = Model(inputs=[inputs1, inputs2], outputs=net)

    return model
