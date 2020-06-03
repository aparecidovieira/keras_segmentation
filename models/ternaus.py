#####
import keras
import keras.backend as K
import resnet as R
import resnext as RX



def build_ternaus(input_shape, n_classes=2, resnet='ResNet34'):
    #inputs = Input((input_shape))

    #resnet_model = R.ResNet34(input_shape, n_classes)
    encoder = RX.ResNeXt50(input_shape, n_classes)
