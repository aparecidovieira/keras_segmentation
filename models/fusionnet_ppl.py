from keras.models import * 
from keras.layers import * 
from keras.activations import * 
import tensorflow as tf
from keras.layers.advanced_activations import *


 
def Upsample(input, rate=2):
    return UpSampling2D(size=rate)(input) #tf.image.resize_bilinear(input, (tf.shape(input)[1] * rate, tf.shape(input)[2] * rate))

 

def add(tensor_a, tensor_b):
    return Add()([tensor_a, tensor_b])

def convBlock(inputs, n_filters, kernel_size=[3, 3], relu_=True):
    net = Conv2D(n_filters, kernel_size=kernel_size,activation = 'relu', strides=1, padding = 'same', kernel_initializer = 
'he_normal')(inputs)

    return net


def shortcut(input, res, n_filters, equal=True):
    if not equal:
        net = Conv2D(n_filters, kernel_size=1,activation = None,strides=1,padding ='same')(input)
        net = add(res, net)
    else:
        net = add(input, res)
    return net

def desconv(inputs, n_filters):
    net = Conv2DTranspose(n_filters, kernel_size=[2, 2],activation = 'relu', strides=2, padding = 'same')(inputs)
    return net

def upSampling(inputs,rate):
    net = UpSampling2D(size=rate, interpolation='nearest')(inputs)
    return net

def upBlock(input_layer, n_filters):
    net = UpSampling2D(2)(input_layer)
    net = Conv2D(n_filters, kernel_size=2,activation = 'relu', strides=1, padding = 'same')(net)
    return net

 

def resBlock(input_layer, n_filters):
    net = Conv2D(n_filters, kernel_size=3,activation = 'relu', strides=1, padding = 'same')(input_layer)
    net = Conv2D(n_filters, kernel_size=3,activation = 'relu', strides=1, padding = 'same')(net)
    net_1 = shortcut(input_layer, net, n_filters, equal=False)
    
    net = Conv2D(n_filters, kernel_size=3,activation = 'relu', strides=1, padding = 'same')(net_1)
    net = Conv2D(n_filters, kernel_size=3,activation = 'relu', strides=1, padding = 'same')(net)
    net_2 = shortcut(net_1, net, n_filters, equal=True)
    
    net = Conv2D(n_filters, kernel_size=3,activation = 'relu', strides=1, padding = 'same')(net_2)
    net = Conv2D(n_filters, kernel_size=3,activation = 'relu', strides=1, padding = 'same')(net)
    net = shortcut(net_2, net, n_filters, equal=True)
    
    
 
        
    return net     



def FPABlock(inputs, n_filters=512, pooling_type='MAX'):
    
    net_1 = Conv2D(n_filters, kernel_size=1,activation = None, strides=1, padding='SAME',)(inputs)

    net_7 = MaxPooling2D(pool_size=2,strides=2,padding='valid')(inputs)
    net_7 = Conv2D(n_filters, kernel_size=7,activation = None, strides=1, padding='SAME',)(net_7)

    net_5 = MaxPooling2D(pool_size=2,strides=2,padding='valid')(net_7)
    net_5 = Conv2D(n_filters, kernel_size=5,activation = None, strides=1,padding='SAME',)(net_5)

    net_3 = MaxPooling2D(pool_size=(2, 2),strides=2,padding='valid')(net_5)
    net_3 = Conv2D(n_filters, kernel_size=[3, 3],activation = None, strides=1, padding='SAME',)(net_3)
    net_3 = Conv2D(n_filters, kernel_size=[3, 3],activation = "relu", strides=1, padding='SAME',)(net_3)
    net_3 = Upsample(net_3) 

    net_5 = Conv2D(n_filters, kernel_size=[5, 5],activation = "relu", strides=1, padding='SAME',)(net_5)    
    net_5 = add(net_3, net_5) 
    net_5 = Upsample(net_5)

    net_7 = Conv2D(n_filters, kernel_size=5,activation = "relu", strides=1,padding= "SAME")(net_7)
    net_7 = add(net_7, net_5)
    net_7 = Upsample(net_7)

    net = Multiply()([net_7, net_1])
    GAP = AveragePooling2D(pool_size=(16, 16),strides=16, padding='SAME')(inputs)
    
    
    net_GAP = Conv2D(n_filters, kernel_size=[1, 1],activation = "relu", strides=1, padding='SAME',)(GAP)
    net_GAP = Upsample(net_GAP,16)  
    net = add(net, net_GAP)
    return net

def build_model(input_size,  keep_prob=1.0,one_hot_label=False):
    input_layer = Input(input_size)

    net = resBlock(input_layer, 64)
    
    skip1 = net
    net =MaxPooling2D(pool_size=(2, 2),strides=2)(net)

    net = resBlock(net, 128)
    skip2 = net
    net = MaxPooling2D(pool_size=(2, 2),strides=2)(net)

    net = resBlock(net, 256)
    skip3 = net
    net = MaxPooling2D(pool_size=(2, 2),strides=2)(net)

    net = resBlock(net, 512)
    skip4 = net
    net = MaxPooling2D(pool_size=(2, 2),strides=2)(net)

    
    #BRIDGE
    net = resBlock(net, 1024)
#     net = PyramidPoolingModule()(net)
    net = FPABlock(net, 1024)

    
 
    # UPsampling
    net = upBlock(net,512)#desconv(net, 512)
    net = add(net, skip4)    
    net = resBlock(net, 512)
    
    net = upBlock(net,256)#desconv(net, 256)
    net = add(net, skip3)    
    net = resBlock(net, 256)
    
    net = upBlock(net,128)#desconv(net, 128)
    net = add(net, skip2)
    net = resBlock(net, 128)

    net = upBlock(net,64)#desconv(net, 64)
    net = add(net, skip1)
    net = resBlock(net, 64)
    
        
    if one_hot_label:
        net = Conv2D(2, 1, 1, activation='relu', border_mode='same')(net)
        net = Reshape((2,input_size[0]*input_size[1]))(net)
        net = Permute((2,1))(net)
        net = Activation('softmax')(net)
    else:
        net = Conv2D(1, kernel_size=1,activation = 'sigmoid', strides=1, padding = 'same')(net)
        
    model = Model(input_layer, net)

    return model