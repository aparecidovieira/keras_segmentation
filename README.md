# Keras Segmentation
Keras segmentation models, U-Net, Attention Unet, Recurrent U-Net, Fusion Net, and Lane-Net (All variations of U-Net)

Image Segmentation using neural networks (NNs), designed for extracting the road network from remote sensing imagery and it can be used in other applications labels every pixel in the image (Semantic segmentation) 

Details can be found in these papers:

* [Unet: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
* [Attention U-Net: Learning Where to Look for the Pancreas](https://arxiv.org/abs/1804.03999)
* [FusionNet: A deep fully residual convolutional neural network for image segmentation in connectomics](https://arxiv.org/pdf/1612.05360)
* [Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net) for Medical Image Segmentation](https://arxiv.org/abs/1802.06955)

## Comparisson of different types of U-Net 

![types U-Net](Images/runet.png)


* (a) Basic convolutional unit in U-Net
* (b) Convolutional unint in RU-Net
* (c) Convolutional unit in Residual U-Net
* (d) Convolutional unit in R2U-Net

## Attention U-Net extra module

![AU-Net](Images/aunet.png)


## Requirements
* Python 3.6
* CUDA 10.0
* Keras 2.0


## Modules
utils.py and helper.py 
functions for preprocessing data and saving it.


## Trainig model:
```
usage: train.py [-h] [--num_epochs NUM_EPOCHS] [--save SAVE] [--gpu GPU]
                [--checkpoint CHECKPOINT] [--class_balancing CLASS_BALANCING]
                [--continue_training CONTINUE_TRAINING] [--dataset DATASET]
                [--batch_size BATCH_SIZE] [--one_hot_label ONE_HOT_LABEL]
                [--data_aug DATA_AUG] [--change CHANGE] [--height HEIGHT]
                [--width WIDTH] [--channels CHANNELS] [--model MODEL]

optional arguments:
  -h, --help            show this help message and exit
  --num_epochs NUM_EPOCHS
                        Number of epochs to train for
  --save SAVE           Interval for saving weights
  --gpu GPU             Choose GPU device to be used
  --checkpoint CHECKPOINT
                        Checkpoint folder.
  --class_balancing CLASS_BALANCING
                        Whether to use median frequency class weights to
                        balance the classes in the loss
  --continue_training CONTINUE_TRAINING
                        Whether to continue training from a checkpoint
  --dataset DATASET     Dataset you are using.
  --batch_size BATCH_SIZE
                        Number of images in each batch
  --one_hot_label ONE_HOT_LABEL
                        One hot label encoding
  --data_aug DATA_AUG   Use or not augmentation
  --change CHANGE       Double image 256, 512
  --height HEIGHT       Height of input image to network
  --width WIDTH         Width of input image to network
  --channels CHANNELS   Number of channels of input image to network
  --model MODEL         The model you are using. Currently supports:
                        fusionNet, fusionNet2, unet, fusionnet_atten, temp,
                        vgg_unet, fusionnet_ppl
```

## Results

Buildings Segmentation  Img - GT - Prediction

![Buildings](Images/Segmentation/building.png)
![Buidings group](Images/Segmentation/building0.png)
![Buidings group](Images/Segmentation/building1.png)
![Buidings group](Images/Segmentation/building2.png)


Roads Segmentation

![Roads](Images/Segmentation/roads.png)
![Roads_group](Images/Segmentation/image34.png)

