###
import keras
from keras.layers import *
import keras.backend as K
import common as C
from keras.models import *



def build_lanenet(input_shape,input_shape1=[128, 128, 3],input_shape2=[64, 64, 3],
    input_shape3=[32, 32, 3],input_shape4=[16, 16, 3], one_hot_label=False):
    print(input_shape, 'Input size shape ~')
    inputs = Input((input_shape))
    n_filters = 64

    inputs_waves1 = Input((input_shape1))
    inputs_waves2 = Input((input_shape2))
    inputs_waves3 = Input((input_shape3))
    inputs_waves4 = Input((input_shape4))
    #wavelet1, wavelet2, wavelet3, wavelet4 = C.lanenet_wavelet(inputs)

    enc1 = C.encoder_block(inputs, n_filters, blocks=2)
    skip1 = C.pool(enc1)
    _skip1 = C.concat(skip1, inputs_waves1)

    enc2 = C.encoder_block(_skip1, 2 * n_filters, blocks=2)
    skip2 = C.pool(enc2)
    _skip2 = C.concat(skip2, inputs_waves2)


    enc3 = C.encoder_block(_skip2, 4 * n_filters)
    skip3 = C.pool(enc3)
    _skip3 = C.concat(skip3, inputs_waves3)


    enc4 = C.encoder_block(_skip3, 8 * n_filters)
    skip4 = C.pool(enc4)
    _skip4 = C.concat(skip4, inputs_waves4)

    enc5 = C.encoder_block(_skip4, 8 * n_filters)
    skip5 = C.pool(enc5)

    bridge = C.encoder_block(skip5, 32 * n_filters, blocks=2)
    #bridge = C.encoder_block(bridge, 16 * n_filters)
    up4 = C.desconv(bridge, 8 * n_filters)
    merge4 = C.add(up4, skip4)
    merge4 = C.conv_block(merge4, 8 * n_filters, 1)

    up3 = C.desconv(merge4, 4 * n_filters)
    merge3 = C.add(up3, skip3)
    merge3 = C.conv_block(merge3, 4 * n_filters, 1)


    up2 = C.desconv(merge3, 2 * n_filters)
    merge2 = C.add(up2, skip2)
    merge2 = C.conv_block(merge2, 2 * n_filters, 1)

    up1 = C.desconv(merge2, n_filters)
    net = C.add(up1, skip1)
    net = C.conv_block(net, n_filters, 1)


    net = C.desconv(net, 1)
    #merge1 = C.conv_block(merge1, n_filters)
    if one_hot_label:
        net = Conv2D(2, 1, 1, activation='relu', border_mode='same')(net)
        net = Reshape((2,input_shape[0]*input_shape[1]))(net)
        net = Permute((2,1))(net)
        net = Activation('softmax')(net)
    else:
        net = Conv2D(1, 1, 1, activation='sigmoid')(net)
    model = Model(inputs=[inputs, inputs_waves1, inputs_waves2, inputs_waves3, inputs_waves4], outputs=net)

    return model
