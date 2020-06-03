from keras.models import * 
from keras.layers import * 
from keras.activations import * 
def resBlock(input_layer, n_filters):
    net = Conv2D(n_filters, kernel_size=3,activation = 'relu', strides=1, padding = 'same', kernel_initializer = 
'he_normal')(input_layer)
    net = Conv2D(n_filters, kernel_size=3,activation = 'relu', strides=1, padding = 'same', kernel_initializer = 
'he_normal')(net)
    net = Conv2D(n_filters, kernel_size=3,activation = 'relu', strides=1, padding = 'same', kernel_initializer = 
'he_normal')(net)
    net = Add()([net, input_layer])
    #net = tf.nn.relu(net)
    return net 
def convBlock(input_layer, n_filters):
    net = Conv2D(n_filters, kernel_size=3,activation = 'relu', strides=1, padding = 'same', kernel_initializer = 
'he_normal')(input_layer)
    return net 
def desconv(input_layer, n_filters):
    net = Conv2DTranspose(n_filters, kernel_size=3,activation = 'relu', strides=2, padding = 'same', 
kernel_initializer = 'he_normal')(input_layer)
    return net 


def upBlock(input_layer, n_filters):
    net = UpSampling2D(2)(input_layer)
    net = Conv2D(n_filters, kernel_size=2,activation = 'relu', strides=1, padding = 'same', kernel_initializer = 
'he_normal')(net)
    return net
    
def build_fusion(input_size):
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

#     net = desconv(net, 512)
    net = upBlock(net,512 )
    
    net = Add()([net, skip4])
    net = convBlock(net, 512)
    net = resBlock(net, 512)
    net = convBlock(net, 512)
    
#     net = desconv(net, 256)
    net = upBlock(net,256 )
   
    net = Add()([net, skip3])
    net = convBlock(net, 256)
    net = resBlock(net, 256)
    net = convBlock(net, 256)
    
#     net = desconv(net, 128)
    net = upBlock(net,128 )
    
    net = Add()([net, skip2])
    net = convBlock(net, 128)
    net = resBlock(net, 128)
    net = convBlock(net, 128)
    
#     net = desconv(net, 64)
    net = upBlock(net,64 )
    
    net = Add()([net, skip1])
    net = convBlock(net, 64)
    net = resBlock(net, 64)
    net = convBlock(net, 64)
#     net = slim.conv2d(net, n_classes, kernel_size=[1, 1], stride=[1, 1], activation_fn=None)
    net = Conv2D(1, kernel_size=1,activation = 'sigmoid', strides=1, padding = 'same', kernel_initializer = 
'he_normal')(net)
    model = Model(inputs=inputs, outputs=net)
    return model
