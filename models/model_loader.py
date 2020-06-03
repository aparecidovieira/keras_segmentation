import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

import unet as unet
import vgg_unet as vgg_unet
import fusionNet as fusionNet
import fusionNet2 as fusionNet2
import fusionnet_ppl as fusionnet_ppl
import fusionnet_atten as fusionnet_atten
import temp as temp
import unetpp
import deepUnet
import multiUnet
import ce as ce_net
import fusionGCN
import unetVgg
import lanenet
import lanenet_2 as lanenet2
import res_unet
from keras.layers import *
import lanenet_att
def get_model(model_name,input_size,one_hot_label, change=False):

    if model_name == "unet":
        model = unet.unet(input_size=input_size)
    elif model_name == "vgg_unet":
        model = vgg_unet.vgg10_unet(input_size,'imagenet',one_hot_label)
    elif model_name == "lanenet":
        h,w,c = input_size
        print('Input size ~~~~~', input_size)
        model = lanenet.build_lanenet(input_shape=input_size, input_shape1=[h/2, w/2, c], input_shape2=[h/4, w/4, c],
            input_shape3=[h/8, w/8, c],input_shape4=[h/16, w/16, c], one_hot_label=one_hot_label)
    elif model_name == "lanenet_att" or model_name == 'attention_lane':
        h,w,c = input_size
        print('Input size ~~~~~', input_size)
        model = lanenet_att.build_lanenet_att(input_shape=input_size, input_shape1=[h/2, w/2, c], input_shape2=[h/4, w/4, c],
            input_shape3=[h/8, w/8, c],input_shape4=[h/16, w/16, c], one_hot_label=one_hot_label)
    elif model_name == "lanenet2":
        h,w,c = input_size
        print('Input size ~~~~~', input_size)
        model = lanenet2.build_lanenet2(input_shape=input_size, input_shape1=[h/2, w/2, c], input_shape2=[h/4, w/4, c],
            input_shape3=[h/8, w/8, c],input_shape4=[h/16, w/16, c], one_hot_label=one_hot_label)
    elif model_name == "vgg_fusion":
        model = unetVgg.vgg10_fusion(input_size,'imagenet',one_hot_label)
    elif model_name == "gcn" or model_name == "fusionGCN":
        model = fusionGCN.build_fusionGCN(input_size,one_hot_label=one_hot_label)
    elif model_name == "ce" or model_name=='cenet':
        model = ce_net.build_ce(input_size,one_hot_label)
    elif model_name == "res_unet" or model_name=='resunet':
        model = res_unet.build_res_unet(input_size, one_hot_label)
    elif model_name == "multiunet" or model_name == "munet":
        model = multiUnet.build_multiUnet(input_size, one_hot_label)
    elif model_name == "fusionNet" or model_name == "fusionnet":
        model = fusionNet.build_fusion(input_size,one_hot_label=one_hot_label)
    elif model_name == "deep" or model_name == "deepunet":
        model = deepUnet.build_deep(input_size,one_hot_label=one_hot_label)
    elif model_name == "unetpp" or model_name == "unet++":
        model = unetpp.build_unetpp(input_size,one_hot_label=one_hot_label)
    elif model_name == "fusionNet2" or model_name == "fusionnet2":
        model = fusionNet2.build_fusion(input_size,one_hot_label=one_hot_label)
    elif model_name == "fusionnet_ppl":
        model = fusionnet_ppl.build_model(input_size,one_hot_label=one_hot_label)
    elif model_name == "temp":
        model = temp.build_temp(input_size,one_hot_label=one_hot_label)
    elif model_name == "fusionnet_atten":
        model = fusionnet_atten.build_model(input_size,one_hot_label=one_hot_label)
    else:
        model = None

    return model
