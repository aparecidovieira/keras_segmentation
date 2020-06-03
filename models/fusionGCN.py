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


def add(tensor_a, tensor_b):
    return Add()([tensor_a, tensor_b])


def resBlock(inputs, n_filters):
    #net1 = Conv2D(n_filters, kernel_size=3, activation='relu', strides=1, padding='same')(inputs)
    net1 = convBlock(inputs, n_filters)
    net1 = convBlock(net1, n_filters)
    net1 = shortcut(net1, inputs, n_filters, not_equal=True)

    net2 = convBlock(net1, n_filters)
    net2 = convBlock(net2, n_filters)
    net2 = shortcut(net2, net1, n_filters)

    net3 = convBlock(net2, n_filters)
    net3 = convBlock(net2, n_filters)
    net3 = shortcut(net3, net2, n_filters)
    return net3

def RerBlock(inputs, n_filters):
    #net_1x1 = convBlock(inputs, n_filters, kernel=1)
    net = resBlock(inputs, n_filters)
    #net = resBlock(net, n_filters)
    #net = Add()([net, net_1x1])
    return net

def poolBlock(inputs, pool=(2, 2), strides=2, padding='same'):
    pll = MaxPooling2D(pool_size=pool, strides=strides, padding=padding)(inputs)
    return pll

def attention(tensor, att_tensor, n_filters=64, kernel=1):
    g1 =  conv1x1(tensor, n_filters)
    x1  = conv1x1(att_tensor, n_filters)
    net = Add()([g1, x1])
    net = Activation('relu')(net)
    net = conv1x1(net, 1)
    net = Activation('sigmoid')(net)

    net = Multiply()([net, att_tensor])
    return net

def upBlock(inputs, n_filters, rate=2):
    #up = UpSampling2D(2)(inputs)
    net =  Conv2DTranspose(n_filters, kernel_size=2, strides=2, activation='relu')(inputs)
    return net
def convBlockv2(inputs, n_filters, kernel_size=[3, 3]):
    net = Conv2D(n_filters, kernel_size=kernel_size, activation='relu', strides=1, padding ='same')(inputs)
    return net

def GCN(inputs, n_filters, k=7):
    net_l1 = convBlockv2(inputs, n_filters, kernel_size=[k, 1])
    net_l2 = convBlockv2(net_l1, n_filters, kernel_size=[1, k])

    net_r1 = convBlockv2(inputs, n_filters, kernel_size=[1, k])
    net_r2 = convBlockv2(net_r1, n_filters, kernel_size=[k, 1])
    net = add(net_l2, net_r2)
    return net

def refinement(inputs, n_filters):
    net_1 = convBlock(inputs, n_filters)
    net_2 = convBlock(net_1, n_filters)
    net_3 = convBlock(net_2, n_filters)
    net = add(net_3, inputs)
    return net

def gcnBlock(inputs, n_filters):
    gcn = GCN(inputs, n_filters)
    ref = refinement(gcn, n_filters)
    return ref

def build_fusionGCN(input_size, one_hot_label=False):
    input_layer = Input(input_size)
    n_filters = 64

    net_1 = RerBlock(input_layer, n_filters)
    net = poolBlock(net_1)

    net_2 = RerBlock(net, 2 * n_filters)
    net = poolBlock(net_2)

    net_3 = RerBlock(net, 4 * n_filters)
    net = poolBlock(net_3)

    net_4 = RerBlock(net, 8 * n_filters)
    net = poolBlock(net_4)

    bridge = RerBlock(net, 16 * n_filters)

    up_4 = upBlock(bridge, 8 * n_filters)
    gcn4 = gcnBlock(net_4, 8 * n_filters)
    merge_4 = add(gcn4, up_4)
    #att_4 = attention(up_4, net_4, 8 * n_filters)
    #merge_4 = Add()([net_4, up_4])
    net = refinement(merge_4, 8 * n_filters)

    up_3 = upBlock(net, 4 * n_filters)
    gcn3 = gcnBlock(net_3, 4 * n_filters)
    merge_3 = add(gcn3, up_3)
    #att_3 = attention(up_3, net_3, 4 * n_filters)
    #merge_3 = Concatenate(-1)([att_3, up_3])
    net = refinement(merge_3, 4 * n_filters)

    up_2 = upBlock(net, 2 * n_filters)
    gcn2 = gcnBlock(net_2, 2 * n_filters)
    merge_2 = add(gcn2, up_2)
    #att_2 = attention(up_2, net_2, 2 * n_filters)
    #merge_2 = Concatenate(-1)([att_2, up_2])
    net = refinement(merge_2, 2 * n_filters)

    up_1 = upBlock(net, n_filters)
    gcn1 = gcnBlock(net_1, n_filters)
    merge_1 = add(gcn1, up_1)
    #att_1 = attention(up_1, net_1, n_filters)
    #merge_1 = Concatenate(-1)([att_1, up_1])
    net = refinement(merge_1, n_filters)

    if one_hot_label:
        net = Conv2D(2, kernel_size=1, strides=1, activation='relu', padding='same')(net)
        net = Reshape((2, input_size[0] * input_size[1]))(net)
        net = Permute((2, 1))(net)
        net = Activation('softmax')(net)
    else:
        net = Conv2D(1, kernel_size=1, strides=1, activation='sigmoid', padding='same')(net)

    model = Model(inputs=input_layer, outputs=net)
    return model
