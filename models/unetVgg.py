from keras.models import *
from keras.layers import *
from tensorflow.python.keras import losses
from keras.applications.vgg16 import VGG16

def concat(tensor_a,tensor_b):
    return Concatenate(axis=3)([tensor_a,tensor_b])

def ConvBlock(inputs,n_filters,kernel_size=3,relu_=True):
    net = Conv2D(n_filters, kernel_size=kernel_size, activation='relu', strides=(1,1), padding='same')(inputs)
    return net

def shortcut(inputs, res, n_filters, equal=True):
    if not equal:
        net = Conv2D(n_filters, kernel_size=1,activation = 'relu',strides=1,padding ='same')(inputs)
        net = add(res, net)
    else:
        net = add(inputs, res)
    return net

def add(tensor_a,tensor_b):
    return Add()([tensor_a,tensor_b])


def ResBlock(inputs, n_filters, n=2):

    net = ConvBlock(inputs, n_filters)
    net = ConvBlock(net, n_filters)
    net_1 = shortcut(inputs, net, n_filters, equal=False)

    net = ConvBlock(net_1, n_filters)
    net = ConvBlock(net, n_filters)
    net_2 = shortcut(net_1, net, n_filters)

    net = ConvBlock(net_2, n_filters)
    net = ConvBlock(net, n_filters)
    net_3 = shortcut(net_2, net, n_filters)

    return net_3

def vgg10_fusion(input_size=(256,256,3), weights='imagenet',one_hot_label=False):
    vgg16_model = VGG16(input_shape=input_size, weights=weights, include_top=False)

    block4_pool = vgg16_model.get_layer('block4_pool').output
    block5_conv2 = ResBlock(block4_pool, 1024)
    # block5_conv1 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block4_pool)
    # block5_conv2 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block5_conv1)
    block5_drop = (block5_conv2)

    block6_up = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(block5_drop))
    block6_merge = Concatenate(axis=3)([vgg16_model.get_layer('block4_conv3').output, block6_up])
    block6_conv3 = ResBlock(block6_merge, 512)
    #block6_conv1 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block6_merge)
    # block6_conv2 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block6_conv1)
    # block6_conv3 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block6_conv2)

    block7_up = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(block6_conv3))
    block7_merge = Concatenate(axis=3)([vgg16_model.get_layer('block3_conv3').output, block7_up])
    block7_conv3 = ResBlock(block7_merge, 256)
    # block7_conv1 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block7_merge)
    # block7_conv2 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block7_conv1)
    # block7_conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block7_conv2)

    block8_up = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(block7_conv3))
    block8_merge = Concatenate(axis=3)([vgg16_model.get_layer('block2_conv2').output, block8_up])
    block8_conv2 = ResBlock(block8_merge, 128)
    # block8_conv1 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block8_merge)
    # block8_conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block8_conv1)

    block9_up = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(block8_conv2))
    block9_merge = Concatenate(axis=3)([vgg16_model.get_layer('block1_conv2').output, block9_up])
    block9_conv2 = ResBlock(block9_merge, 64)
    # block9_conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block9_merge)
    # block9_conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block9_conv1)

    net = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block9_conv2)

    if one_hot_label:
        net = Conv2D(2, 1, 1, activation='relu', border_mode='same')(net)
        net = Reshape((2,input_size[0]*input_size[1]))(net)
        net = Permute((2,1))(net)
        net = Activation('softmax')(net)
    else:
        net = Conv2D(1, 1, activation='sigmoid')(net)

#     block10_conv2 = Conv2D(1, 1, activation='sigmoid')(block10_conv1)

    model = Model(inputs=vgg16_model.input, outputs=net)
    return model
