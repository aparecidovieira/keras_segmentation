###
import keras, os
from keras.layers import *
from keras.activations import *
from keras.models import *
from keras import backend as K

axis=-1
freeze_bn = False
parameters = {
    "kernel_initializer": "he_normal"
}

def convBlock(inputs, n_filters, kernel=3, strides=1, t=False):
    net = Conv2D(n_filters, kernel_size=[kernel, kernel], activation = 'relu', strides=1, padding ='same')(inputs)
    net = Conv2D(n_filters, kernel_size=[kernel, kernel], activation = 'relu', strides=1, padding ='same')(net)
    if t:
        net = Conv2D(n_filters, kernel_size=[kernel, kernel], activation = 'relu', strides=1, padding ='same')(net)


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

def resBlock(filters, stage=0, block=0, kernel_size=3, numerical_name=False, stride=None, freeze_bn=False):

    if stride is None:
        if block != 0 or stage == 0:
            stride = 1
        else:
            stride = 2


    if block > 0 and numerical_name:
        block_char = "b{}".format(block)
    else:
        block_char = chr(ord('a') + block)

    stage_char = str(stage + 2)

    def f(x):
        y = keras.layers.Conv2D(filters, (1, 1), strides=stride, use_bias=False, name="res{}{}_branch2a".format(stage_char, block_char), **parameters)(x)
        y = BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn{}{}_branch2a".format(stage_char, block_char))(y)
        y = keras.layers.Activation("relu", name="res{}{}_branch2a_relu".format(stage_char, block_char))(y)

        y = keras.layers.ZeroPadding2D(padding=1, name="padding{}{}_branch2b".format(stage_char, block_char))(y)
        y = keras.layers.Conv2D(filters, kernel_size, use_bias=False, name="res{}{}_branch2b".format(stage_char, block_char), **parameters)(y)
        y = BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn{}{}_branch2b".format(stage_char, block_char))(y)
        y = keras.layers.Activation("relu", name="res{}{}_branch2b_relu".format(stage_char, block_char))(y)

        y = keras.layers.Conv2D(filters * 4, (1, 1), use_bias=False, name="res{}{}_branch2c".format(stage_char, block_char), **parameters)(y)
        y = BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn{}{}_branch2c".format(stage_char, block_char))(y)

        if block == 0:
            shortcut = keras.layers.Conv2D(filters * 4, (1, 1), strides=stride, use_bias=False, name="res{}{}_branch1".format(stage_char, block_char), **parameters)(x)
            shortcut = BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn{}{}_branch1".format(stage_char, block_char))(shortcut)
        else:
            shortcut = x

        y = keras.layers.Add(name="res{}{}".format(stage_char, block_char))([y, shortcut])
        y = keras.layers.Activation("relu", name="res{}{}_relu".format(stage_char, block_char))(y)

        return y

    return f

class BatchNormalization(keras.layers.BatchNormalization):
    """
    Identical to keras.layers.BatchNormalization, but adds the option to free$
    """
    def __init__(self, freeze, *args, **kwargs):
        self.freeze = freeze
        super(BatchNormalization, self).__init__(*args, **kwargs)

        # set to non-trainable if freeze is true
        self.trainable = not self.freeze

    def call(self, *args, **kwargs):
        # Force test mode if frozen, otherwise use default behaviour (i.e., t$
        if self.freeze:
            kwargs['training'] = False
        return super(BatchNormalization, self).call(*args, **kwargs)

    def get_config(self):
        config = super(BatchNormalization, self).get_config()
        config.update({'freeze': self.freeze})
        return config




def build_resnet(input_size, blocks=[3, 4, 6, 3]):
    inputs = Input(input_size)
    numerical_names=None
    if numerical_names is None:
        numerical_names = [True] * len(blocks)
    x = inputs
    x = keras.layers.ZeroPadding2D(padding=3, name="padding_conv1")(inputs)
    x = keras.layers.Conv2D(64, (7, 7), strides=(2, 2), use_bias=False, name="conv1")(x)
    # print(K.int_shape(x), 'HEERRR')
    x = BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn_conv1")(x)
    x = keras.layers.Activation("relu", name="conv1_relu")(x)
    x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool1")(x)

    features = 64
    outputs = []
    for stage_id, iterations in enumerate(blocks):
        for block_id in range(iterations):
            x = resBlock(features, stage_id, block_id, numerical_name=(block_id > 0 and numerical_names[stage_id]), freeze_bn=freeze_bn)(x)
            #print('Wu tang clan')
        features *= 2

        outputs.append(x)
    return inputs, keras.models.Model(inputs=inputs, outputs=outputs)

def load_weights(model, weights):
    model.load_weights(weights, by_name=True, skip_mismatch=True)
    return model


def build_res_unet(input_size, one_hot_label=False, weights='./ResNet-50-model.keras.h5'):

    inputs, resnet = build_resnet(input_size)
    model = resnet
    features = 64
    cwd = os.getcwd()
    weight_f = cwd + weights
    #print(weight_f, 'hereeeeeee')
    if os.path.isfile(weights):
        print('Loading weights for resnet backbone')
        #model = load_weights(resnet, weights)
    else:
        print('XXXXXXX Loading resnet without imagenet weigths XXXXXX')
    #print(model.summary())
    layers_dic = ['conv1_relu','res2b2_relu','res3b3_relu', 'res4b5_relu','res5b2_relu']
    layers = {p_name:model.get_layer(p_name).output for p_name in layers_dic}

    bridge = convBlock(layers['res5b2_relu'], 16 * features)### [8, 8]

    # conv5 = layers[-1]
    up4 = desconv(bridge, 16 * features)
    merge4 = Concatenate(axis=-1)([up4, layers['res4b5_relu']])
    up4 = convBlock(merge4, 16 * features, t=True)
    # layers = {p_name:model.get_layer(p_name).output for p_name in layers_dic}

    up3 = desconv(up4, 8 * features)
    merge3 = Concatenate(axis=-1)([up3, layers['res3b3_relu']])
    up3 = convBlock(merge3, 8 * features, t=True)

    up2 = desconv(up3, 4 * features)
    merge2 = Concatenate(axis=-1)([up2, layers['res2b2_relu']])
    up2 = convBlock(merge2, 4 * features)

    up1 = desconv(up2, 2 * features)
    merge1 = Concatenate(axis=-1)([up1, layers['conv1_relu']])
    net = convBlock(merge1, 2 * features)

    up0 = desconv(up1, features)
    #merge1 = Concatenate(axis=-1)([up1, layers['conv1_relu']])
    net = convBlock(up0, features)

    #net = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up0)


    if one_hot_label:
        net = Conv2D(2, kernel_size=1, strides=1, activation='relu', padding='same')(net)
        net = Reshape((2, input_size[0] * input_size[1]))(net)
        net = Permute((2, 1))(net)
        net = Activation('softmax')(net)
    else:
        net = Conv2D(1, kernel_size=1, strides=1, activation='sigmoid', padding='same')(net)

    model = Model(inputs=inputs, outputs=net)

    return model
