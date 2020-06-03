import numpy as np
import cv2
from util import custom_data_generator as data_util
import os, glob, shutil
from models.common import lanenet_wavelet


def calculate_IoU_Per_Epoch(model,val_inputs_path,val_masks_path,checkpoint_path,epoch_number,one_hot_label=False, change=False):

    val_samples = [os.path.basename(x) for x in glob.glob(val_inputs_path + "*.png")]
#     val_samples = [s for s in val_samples if "seoul" in s] + \
#                 [s for s in val_samples if "suwon" in s] + \
#                 [s for s in val_samples if "daegu" in s]
    val_samples_mini = val_samples[:] #+ val_samples[2000:2015]


    save_path = "%s/epoch_%s/"%(checkpoint_path,epoch_number)
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    IU_BG_TP,IU_BG_FN,IU_BG_FP,IU_BG_TN,IU_BD_TP,IU_BD_FN,IU_BD_FP,IU_BD_TN=[],[],[],[],[],[],[],[]

    for j in range(len(val_samples_mini)):
        img = val_samples_mini[j]
#         print(img)

        gt =  data_util.get_mask(val_masks_path+img,one_hot_label=one_hot_label,do_aug=[])[:,:,0].astype(int)
        input_image = data_util.get_image(val_inputs_path+img,do_aug=[],change=change)
        if change:
            input_image1, input_image2 = input_image[:, :, :3], input_image[:, :, 3:]

        input_image_gray = data_util.get_image(val_inputs_path+img,do_aug=[],gray=True)
        # input_image_gray = cv2.cvtColor(input_image1, cv2.COLOR_BGR2GRAY)
        w1, w2, w3, w4 = lanenet_wavelet(input_image_gray[:, :256])
        w1 = np.expand_dims(w1, axis=0)
        w2 = np.expand_dims(w2, axis=0)
        w3 = np.expand_dims(w3, axis=0)
        w4 = np.expand_dims(w4, axis=0)

        #input_double = cv2.imread(val_inputs_path+img, -1)/255
        if change:
            pred = model.predict([np.expand_dims(input_image1, axis=0), np.expand_dims(input_image2, axis=0), w1, w2, w3, w4], batch_size=None, verbose=0, steps=None)
        else:
            pred = model.predict([np.expand_dims(input_image, axis=0), w1, w2, w3, w4], batch_size=None, verbose=0, steps=None)


        if one_hot_label:
            pred = np.reshape(pred, (256 , 256, 2))
            pred = np.argmax(pred, axis=-1).astype(int)
        else:
            pred = np.round(pred[0,:,:,0]).astype(int)      #(pred *255).astype(int)

        classes = np.array([0, 1])
        for ii in classes:
            TP, FN, FP, TN = IoU(pred, gt, ii)
            if ii == 0:
                IU_BG_TP.append(TP)
                IU_BG_FN.append(FN)
                IU_BG_FP.append(FP)
                IU_BG_TN.append(TN)

            elif ii == 1:
                IU_BD_TP.append(TP)
                IU_BD_FN.append(FN)
                IU_BD_FP.append(FP)
                IU_BD_TN.append(TN)
        gt_pred = np.concatenate((data_util.changelabels(gt,'1d2rgb'),data_util.changelabels(pred,'1d2rgb')),axis=1)
        if change:
            input_image = np.concatenate((input_image[:, :, :3], input_image[:, :, 3:]), axis=1)
        
        cv2.imwrite(save_path+img,np.concatenate((input_image*255, gt_pred),axis=1))
#         break
    print(IU_BD_TP)
    BG_IU = 100 * divided_IoU(IU_BG_TP, IU_BG_FN, IU_BG_FP)
    BD_IU = 100 * divided_IoU(IU_BD_TP, IU_BD_FN, IU_BD_FP)
    BG_P = 100 * divided_PixelAcc(IU_BG_TP, IU_BG_FN)
    BD_P = 100 * divided_PixelAcc(IU_BD_TP, IU_BD_FN)

    return BG_IU, BD_IU, BG_P, BD_P




def IoU(pred, valid, cl):
    tp = np.count_nonzero(np.logical_and(pred == cl, valid == cl))
    fn = np.count_nonzero(np.logical_and(pred != cl, valid == cl))
    fp = np.count_nonzero(np.logical_and(pred == cl, valid != cl))
    tn = np.count_nonzero(np.logical_and(pred != cl, valid != cl))
    return tp, fn, fp, tn


def divided_IoU(tp, fn, fp):
    try:
        return float(sum(tp)) / (sum(tp) + sum(fn) + sum(fp))
    except ZeroDivisionError:
        return 0


def divided_PixelAcc(tp, fn):
    try:
        return float(sum(tp)) / (sum(tp) + sum(fn))
    except ZeroDivisionError:
        return 0
