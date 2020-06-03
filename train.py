import os
import numpy as np
from keras.models import *
from keras.layers import *
from tensorflow.python.keras import losses
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
import cv2
import glob
import matplotlib.pyplot as plt
import itertools
import warnings
warnings.filterwarnings("ignore")
from util import losses,custom_data_generator, metrics
from keras.utils import multi_gpu_model
#from models import model_loader
import datetime, sys
from keras.callbacks import TensorBoard,Callback
from keras.models import model_from_json
import random, argparse
sys.path.append("models")
import model_loader

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train for')
parser.add_argument('--save', type=int, default=4, help='Interval for saving weights')
parser.add_argument('--gpu', type=str, default='2', help='Choose GPU device to be used')
parser.add_argument('--checkpoint', type=str, default="checkpoint", help='Checkpoint folder.')
parser.add_argument('--class_balancing', type=str2bool, default=False, help='Whether to use median frequency class weights to balance the classes in the loss')
parser.add_argument('--continue_training', type=str2bool, default=False, help='Whether to continue training from a checkpoint')
parser.add_argument('--dataset', type=str, default='LANE_19', help='Dataset you are using.')
parser.add_argument('--batch_size', type=int, default=10, help='Number of images in each batch')
parser.add_argument('--one_hot_label', type=str, default=False, help='One hot label encoding')
parser.add_argument('--data_aug', type=str, default=False, help='Use or not augmentation')
parser.add_argument('--height', type=int, default=256, help='Height of input image to network')
parser.add_argument('--change', type=str2bool, default=False, help='Double image 256, 512')
parser.add_argument('--width', type=int, default=256, help='Width of input image to network')
parser.add_argument('--channels', type=int, default=3, help='Number of channels of input image to network')
parser.add_argument('--model', type=str, default="temp", help='The model you are using. Currently supports:\
    fusionNet, fusionNet2, unet, fusionnet_atten, temp, vgg_unet, fusionnet_ppl')
args = parser.parse_args()



batch_size = args.batch_size
model_name = args.model
input_size = [args.height, args.width, args.channels]
channeles = args.channels
checkpoint_name = args.checkpoint + '_' + args.model
one_hot_label = args.one_hot_label
data_aug = args.data_aug
dataset = args.dataset
num_epochs = args.num_epochs
change = args.change

gpu = str(args.gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

checkpoint_dir = "./checkpoints/%s/"%(checkpoint_name)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

path = '../../../Lane_Data/'+dataset


train_inputs_path = path + "/train/"
train_masks_path = path + "/train_labels/"
val_inputs_path = path + "/val/"
val_masks_path = path + "/val_labels/"



def get_dataset():

    # all_samples = [os.path.basename(x) for x in glob.glob(train_inputs_path + "*.png")]
    train_samples = [os.path.basename(x) for x in glob.glob(train_inputs_path + "*.png")]
    # train_samples = [s for s in train_samples if "seoul" in s] + \
    #                 [s for s in train_samples if "suwon" in s] + \
    #                 [s for s in train_samples if "daegu" in s]
    val_samples = [os.path.basename(x) for x in glob.glob(val_inputs_path + "*.png")]
    # val_samples_ = random.sample(all_samples, 40)#[os.path.basename(x) for x in glob.glob(val_inputs_path + "*.png")]
    # val_samples = [s for s in val_samples if "seoul" in s] + \
    #                 [s for s in val_samples if "suwon" in s] + \
    #                 [s for s in val_samples if "daegu" in s]
    #extra = random.sample(val_samples_, len(val_samples_)//6*5)
    #train_samples_ += extra
    #train_samples_ = [imgPath for imgPath in all_samples if imgPath not in val_samples_]
    #val_samples2 = [path for path in val_samples_ if path not in extra]
    return train_samples, val_samples

train_samples, val_samples = get_dataset()

print("\n\nTraining samples = %s"%(len(train_samples)))
print("Validation samples = %s\n\n"%(len(val_samples)))
train_generator = custom_data_generator.image_generator(train_samples,train_inputs_path,train_masks_path, batch_size,one_hot_label,data_aug, change=change)
val_generator = custom_data_generator.image_generator(val_samples,val_inputs_path,val_masks_path,batch_size,one_hot_label, change=change)


##-------- Model loading
model = model_loader.get_model(model_name=model_name,input_size=input_size,one_hot_label=one_hot_label)
# json_file = open("./models/unet_resnext_50_lovasz.json", 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
#print(model.summary())

model.save(model_name + '.h5')

if model == None:
    print("Model ..[ %s ] definition not found."%model_name)
    exit(0)
model_json = model.to_json()
with open(checkpoint_dir+model_name+".json",'w') as json_file:
    json_file.write(model_json)


##---------load pred trained weights
if args.continue_training:
    print('Loading pre-trained model ~~~')
    model.load_weights(checkpoint_dir+"lanenet_weights_400.h5")
model_parallel = model
avaliabe_gpus = len(K.tensorflow_backend._get_available_gpus())
if avaliabe_gpus > 1:
    model_parallel = multi_gpu_model(model,gpus=avaliabe_gpus)
    print("\nTraining using %s GPUs.."%avaliabe_gpus)



# CheckPoint and Callbacks
tensorboard = TensorBoard(log_dir=checkpoint_dir, histogram_freq=0,write_graph=True, write_images=True)
weights_path = checkpoint_dir+model_name+'_weights_{epoch:02d}.h5'
class onEachEpochCheckPoint(Callback):
    def __init__(self, model_parallel, path,model,one_hot_label=one_hot_label):
        super().__init__()
        self.path = path
        self.model_for_saving = model
        self.one_hot_label=one_hot_label

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            self.model_for_saving.save_weights(self.path.format(epoch=epoch), overwrite=True)
        BG_IU, BD_IU, BG_P, BD_P = metrics.calculate_IoU_Per_Epoch(model,val_inputs_path,val_masks_path,checkpoint_dir,epoch,one_hot_label)

        print("\nBackground IOU = %02f"%BG_IU)
        print("Main-Class IOU = %02f"%BD_IU)
        print("Mean IOU = %02f"%((BG_IU + BD_IU)/2))
        print("Background P-Accuracy = %02f"%BG_P)
        print("Main-Class P-Accuracy = %02f\n"%BD_P)



##--- Class weights
# class_weights=custom_data_generator.compute_class_weights(train_masks_path,[(0,0,0),(255,255,255)])
#class_weights = [0.23381152, 0.76618848]# building
# class_weights = [0.08381668, 0.91618332] #road
class_weights =  [0.15809653, 0.84190347]#New Buildings
print("\nClass weights  = ",class_weights)
import segmentation_models as sm

##------- Loss function accuracy matrics and optimizer
# loss_fucntions = [losses.categorical_crossentropy,'binary_crossentropy',losses.jaccard_distance, losses.dice_loss,losses.lovasz_loss,losses.binary_focal_loss(gamma=2., alpha=.25)]
loss = 'binary_crossentropy'#losses.binary_focal_loss(gamma=2., alpha=.25)#'binary_crossentropy'
# loss_focal = losses.binary_focal_loss(gamma=2., alpha=0.25)
metrics_list = ['accuracy', losses.dice_coef, losses.iouMetric]
optimizer=Adam(lr=1e-4)#RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.995)#Adam(lr=1e-4)
# loss=  sm.losses.bce_jaccard_loss
# metrics_list =[sm.metrics.iou_score,sm.metrics.f1_score]

##--- Model compile and training
model_parallel.compile(optimizer=optimizer, loss=loss, metrics=metrics_list)
# model_parallel.compile(
#     optimizer,
#     loss=  sm.losses.bce_jaccard_loss,
#     metrics=[sm.metrics.iou_score,sm.metrics.f1_score],
# )
model_checkpoint = onEachEpochCheckPoint(model_parallel, weights_path, model,one_hot_label=one_hot_label)


model_parallel.fit_generator(train_generator,
                        steps_per_epoch= int(len(train_samples)/batch_size),
                        epochs=num_epochs,
                        #initial_epoch=78,
                        validation_data=val_generator,
                        validation_steps=5,
                        #class_weight=class_weights,
                        callbacks=[model_checkpoint,tensorboard])
