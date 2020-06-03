from keras.models import *
from keras.layers import *
from keras.activations import *




def cse_block(prevlayer, prefix):
    mean = Lambda(lambda xin: K.mean(xin, axis=[1, 2]))(prevlayer)
    lin1 = Dense(K.int_shape(prevlayer)[3] // 2, name=prefix + 'cse_lin1', activation='relu')(mean)
    lin2 = Dense(K.int_shape(prevlayer)[3], name=prefix + 'cse_lin2', activation='sigmoid')(lin1)
    x = Multiply()([prevlayer, lin2])
    return x


def sse_block(prevlayer, prefix):
    # Bug? Should be 1 here?
    conv = Conv2D(K.int_shape(prevlayer)[3], (1, 1), padding="same", kernel_initializer="he_normal",
                  activation='sigmoid', strides=(1, 1),
                  name=prefix + "_conv")(prevlayer)
    conv = Multiply(name=prefix + "_mul")([prevlayer, conv])
    return conv


def csse_block(x, prefix):
    '''
    Implementation of Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks
    https://arxiv.org/abs/1803.02579
    '''
    cse = cse_block(x, prefix)
    sse = sse_block(x, prefix)
    x = Add(name=prefix + "_csse_mul")([cse, sse])

    return x


def resBlock(input_layer, n_filters):
    net = Conv2D(n_filters, kernel_size=3,activation = 'relu', strides=1, padding = 'same', kernel_initializer = 'he_normal')(input_layer)
    net = Conv2D(n_filters, kernel_size=3,activation = 'relu', strides=1, padding = 'same', kernel_initializer = 'he_normal')(net)
    net = Conv2D(n_filters, kernel_size=3,activation = 'relu', strides=1, padding = 'same', kernel_initializer = 'he_normal')(net)
    net = Add()([net, input_layer])
    #net = tf.nn.relu(net)
    return net

def convBlock(input_layer, n_filters):
    net = Conv2D(n_filters, kernel_size=3,activation = 'relu', strides=1, padding = 'same', kernel_initializer = 'he_normal')(input_layer)
    return net

def desconv(input_layer, n_filters):
    net = Conv2DTranspose(n_filters, kernel_size=3,activation = 'relu', strides=2, padding = 'same', kernel_initializer = 'he_normal')(input_layer)
    return net




def build_fusion(input_size,one_hot_label):
    inputs = Input(input_size)

    net = convBlock(inputs, 64)

    net = resBlock(net, 64)
    net = convBlock(net, 64)
    skip1 = net
    net = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(net)

    net = convBlock(net, 128)
    net = resBlock(net, 128)
    net = convBlock(net, 128)
    skip2 = net
    net = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(net)

    net = convBlock(net, 256)
    net = resBlock(net, 256)
    net = convBlock(net, 256)
    skip3 = net
    net = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(net)

    net = convBlock(net, 512)
    net = resBlock(net, 512)
    net = convBlock(net, 512)
    skip4 = net
    net = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(net)

   #BRIDGE

    net = convBlock(net, 1024)
    net = resBlock(net, 1024)
    net = convBlock(net,1024)


 # UPsampling
    net = desconv(net, 512)
    net = Add()([net, skip4])
    net = csse_block(net, prefix='csse_block_{}'.format(0))
    net = convBlock(net, 512)
    net = resBlock(net, 512)
    net = convBlock(net, 512)

    net = desconv(net, 256)
    net = Add()([net, skip3])
    net = csse_block(net, prefix='csse_block_{}'.format(1))
    net = convBlock(net, 256)
    net = resBlock(net, 256)
    net = convBlock(net, 256)


    net = desconv(net, 128)
    net = Add()([net, skip2])
    net = csse_block(net, prefix='csse_block_{}'.format(2))
    net = convBlock(net, 128)
    net = resBlock(net, 128)
    net = convBlock(net, 128)


    net = desconv(net, 64)
    net = Add()([net, skip1])
    net = csse_block(net, prefix='csse_block_{}'.format(3))
    net = convBlock(net, 64)
    net = resBlock(net, 64)
    net = convBlock(net, 64)

    if one_hot_label:
        net = Conv2D(2, kernel_size=1, strides=1, activation='relu',  padding = 'same')(net)
        net = Reshape((2,input_size[0]*input_size[1]))(net)
        net = Permute((2,1))(net)
        net = Activation('softmax')(net)
   
    else:
        net = Conv2D(1, kernel_size=1,activation = 'sigmoid', strides=1, padding = 'same')(net)
        
        
        
    model = Model(inputs=inputs, outputs=net)
    return model
