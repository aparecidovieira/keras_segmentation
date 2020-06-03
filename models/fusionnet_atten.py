from keras.models import *
from keras.layers import *
from keras.activations import *
import tensorflow as tf
from keras.layers.advanced_activations import *
from keras import backend as K

def add(tensor_a,tensor_b):
    return Add()([tensor_a,tensor_b])

def multiply(tensor_a,tensor_b):
    return Multiply()([tensor_a,tensor_b])

def concat(tensor_a,tensor_b):
    return Concatenate(axis=3)([tensor_a,tensor_b])

def ConvBlock(inputs,n_filters,kernel_size=3,relu_=True):
    net = Conv2D(n_filters, kernel_size=kernel_size, activation='relu', strides=(1,1), padding='same')(inputs)
    return net

def dilate(inputs,n_filters,dilation):
    net = Conv2D(n_filters,kernel_size=3,strides=1,activation='elu',dilation_rate=(dilation,dilation),padding='same')(inputs)
    return net

def Dblock(net,n_filters=512):
    dil_2 = dilate(net,n_filters,2)
    dil_3 = dilate(dil_2, n_filters, 4)
    dil_4 = dilate(dil_3, n_filters, 8)
    dil_5 = dilate(dil_4, n_filters, 16)
    dil_6 = dilate(dil_5, n_filters, 32)
    
    net = add(dil_2,dil_3)
    net = add(net,dil_4)
    net = add(net,dil_5)
    net = add(net,dil_6)
    #net = dil_2 + dil_3 + dil_4 + dil_5 + dil_6
    return net

def Upsample(inputs,rate=2):
    return UpSampling2D(size=rate)(inputs)

def UpBlock(inputs,n_filters):
    net = UpSampling2D(2)(inputs)
    net = Conv2D(n_filters,kernel_size=2,activation='relu',strides=1,padding='same')(net)
    return net

def RRBlock(inputs,n_filters):
    net_1x1 = Conv2D(n_filters,kernel_size=1,activation='relu',strides=1,padding='same')(inputs)
    net = RerBlock(net_1x1,n_filters)
    net = RerBlock(net,n_filters)
    return add(net,net_1x1)

def shortcut(inputs, res, n_filters, equal=True):
    if not equal:
        net = Conv2D(n_filters, kernel_size=1,activation = 'relu',strides=1,padding ='same')(inputs)
        net = add(res, net)
    else:
        net = add(inputs, res)
    return net

def RerBlock(inputs, n_filters, n=2):

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

def attention(tensor,att_tensor,n_filters=512,kernel_size=1):
    g1 = Conv2D(n_filters, kernel_size=kernel_size, activation='relu', strides=1, padding='same')(tensor)
    x1 = Conv2D(n_filters, kernel_size=kernel_size, activation='relu', strides=1, padding='same')(att_tensor)
    net = add(g1,x1)
    #net = relu(net)
    net = Conv2D(1, kernel_size=kernel_size, activation='sigmoid', strides=1, padding='same')(net)
    net = multiply(net,att_tensor)
    return net

def build_model(input_size,  keep_prob=1.0,one_hot_label=False):
    input_layer = Input(input_size)
    n_filters = 64
    #input_layer = Input(input_size)
  
    net = RRBlock(input_layer,n_filters)
    skip1 = net
    net = MaxPooling2D(pool_size=2,strides=2,padding='valid')(net)
    
    net = RRBlock(net,n_filters*2)
    skip2 = net
    net = MaxPooling2D(pool_size=(2, 2),strides=2)(net)
    
    net = RRBlock(net,n_filters*4)
    skip3 = net
    net = MaxPooling2D(pool_size=(2, 2),strides=2)(net)
    
    net = RRBlock(net,n_filters*8)
    skip4 = net
    net = MaxPooling2D(pool_size=(2, 2),strides=2)(net)
    
    net = Dblock(net,n_filters*16)
    D = net
    up_4 = UpBlock(net, 512)
   
    net = attention(up_4, skip4, 8 * n_filters)
    net = concat(net, up_4)
    net = RRBlock(net, n_filters * 8)

    up_3 = UpBlock(net, n_filters * 4)
    net = attention(up_3, skip3, 4 * n_filters)
    net = concat(net, up_3)
    net = RRBlock(net, n_filters * 4)

    up_2 = UpBlock(net, n_filters * 2)
    net = attention(up_2, skip2, 2 * n_filters)
    net = concat(net, up_2)
    net = RRBlock(net, n_filters * 2)

    up_1 = UpBlock(net, n_filters)
    net = attention(up_1, skip1, n_filters)
    net = concat(net, up_1)
    net = RRBlock(net, n_filters)

    #net = Upsample(nei, rate=2)
    net = Conv2D(1, kernel_size=1, activation='sigmoid',strides=1, padding='same')(net)
    model = Model(inputs=input_layer,outputs=net)
    #print(model)
    return model








