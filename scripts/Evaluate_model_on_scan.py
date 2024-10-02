#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 15:32:10 2023

@author: deeperthought
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import resize
from scipy.stats import mode


import tensorflow as tf

if tf.__version__[0] == '1':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list="0"
    tf.keras.backend.set_session(tf.Session(config=config))

elif tf.__version__[0] == '2':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      # Restrict TensorFlow to only use the first GPU
      try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
      except RuntimeError as e:
        # Visible devices must be set at program startup
        print(e)


from tensorflow.keras import backend as K

partition = np.load('/home/deeperthought/Projects/Others/2D_brain_segmenter/Sessions/DATA/data.npy', allow_pickle=True).item()

#%% METRICS AND LOSSES
        
def dice_coef_numpy(y_true, y_pred):
    smooth = 1e-6
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    
def Generalised_dice_coef_multilabel_numpy(y_true, y_pred, numLabels=2):
    dice=0
    for index in range(numLabels):
        dice += dice_coef_numpy(y_true == index, y_pred == index)
    return  dice/float(numLabels)


def dice_loss(y_true, y_pred):
#  y_true = tf.cast(y_true, tf.float32)
#  y_pred = tf.math.sigmoid(y_pred)
  numerator = 2 * tf.math.reduce_sum(y_true * y_pred)
  denominator = tf.math.reduce_sum(y_true + y_pred)
  return 1 - numerator / denominator

def Generalised_dice_coef_multilabel7(y_true, y_pred, numLabels=7):
    """This is the loss function to MINIMIZE. A perfect overlap returns 0. Total disagreement returns numeLabels"""
    dice=0
    for index in range(numLabels):
        dice -= dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
    return numLabels + dice

def dice_coef(y_true, y_pred):
    smooth = 1e-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f**2) + tf.reduce_sum(y_pred_f**2) + smooth)


def dice_coef_multilabel_bin0(y_true, y_pred):
    dice = dice_coef(y_true[:,:,:,0], tf.math.round(y_pred[:,:,:,0]))
    return dice

def dice_coef_multilabel_bin1(y_true, y_pred):
    dice = dice_coef(y_true[:,:,:,1], tf.math.round(y_pred[:,:,:,1]))
    return dice

def dice_coef_multilabel_bin2(y_true, y_pred):
    dice = dice_coef(y_true[:,:,:,2], tf.math.round(y_pred[:,:,:,2]))
    return dice

def dice_coef_multilabel_bin3(y_true, y_pred):
    dice = dice_coef(y_true[:,:,:,3], tf.math.round(y_pred[:,:,:,3]))
    return dice

def dice_coef_multilabel_bin4(y_true, y_pred):
    dice = dice_coef(y_true[:,:,:,4], tf.math.round(y_pred[:,:,:,4]))
    return dice

def dice_coef_multilabel_bin5(y_true, y_pred):
    dice = dice_coef(y_true[:,:,:,5], tf.math.round(y_pred[:,:,:,5]))
    return dice

def dice_coef_multilabel_bin6(y_true, y_pred):
    dice = dice_coef(y_true[:,:,:,6], tf.math.round(y_pred[:,:,:,6]))
    return dice


def dice_coef_multilabel_bin0_numpy(y_true, y_pred):
    dice = dice_coef_numpy(y_true[:,:,:,0], np.round(y_pred[:,:,:,0]))
    return dice

def dice_coef_multilabel_bin1_numpy(y_true, y_pred):
    dice = dice_coef_numpy(y_true[:,:,:,1], np.round(y_pred[:,:,:,1]))
    return dice

def segment_and_dice(mri_padded, model, orientation='sagittal', seg_padded = np.array([])):
    INDEX = 0
    SLICES = mri_padded.shape[0]
    segmentation = np.zeros((mri_padded.shape))
    segmentation_probs = np.zeros((256,256,256,7))
    
    for INDEX in range(SLICES):
        #print('{}/{}'.format(INDEX,SLICES))
        if orientation == 'sagittal':
            x = mri_padded[INDEX]
        elif orientation == 'coronal':
            x = mri_padded[:,INDEX]
        elif orientation == 'axial':
            x = mri_padded[:,:,INDEX]
            
        x = np.expand_dims(np.expand_dims(x,0),-1)    
        yhat = model.predict(x)
        
        seg_slices = np.argmax(yhat,-1)
        
        
        if orientation == 'sagittal':
            segmentation[INDEX] = seg_slices
            segmentation_probs[INDEX] = yhat
        elif orientation == 'coronal':
            segmentation[:,INDEX] = seg_slices
            segmentation_probs[:,INDEX] = yhat
        elif orientation == 'axial':
            segmentation[:,:,INDEX] = seg_slices
            segmentation_probs[:,:,INDEX] = yhat
    
    if len(seg_padded) == 0:
        print('no segmentation given. No dice.')
        return segmentation, 0
            
    return segmentation, Generalised_dice_coef_multilabel_numpy(seg_padded, segmentation, 7), segmentation_probs

#%%
OUTPUT_PATH = '/home/deeperthought/Projects/DGNS/Detection_model/Sessions/'


my_custom_objects = {'Generalised_dice_coef_multilabel7':Generalised_dice_coef_multilabel7,
                                 'dice_coef_multilabel_bin0':dice_coef_multilabel_bin0,
                                 'dice_coef_multilabel_bin1':dice_coef_multilabel_bin1,
                                 'dice_coef_multilabel_bin2':dice_coef_multilabel_bin2,
                                 'dice_coef_multilabel_bin3':dice_coef_multilabel_bin3,
                                 'dice_coef_multilabel_bin4':dice_coef_multilabel_bin4,
                                 'dice_coef_multilabel_bin5':dice_coef_multilabel_bin5,
                                 'dice_coef_multilabel_bin6':dice_coef_multilabel_bin6}

SAGITTAL_MODEL_SESSION_PATH = '/home/deeperthought/Projects/Others/2D_brain_segmenter/Sessions/sagittal_segmenter_NoDataAug/'
AXIAL_MODEL_SESSION_PATH = '/home/deeperthought/Projects/Others/2D_brain_segmenter/Sessions/axial_segmenter_NoDataAug/'
CORONAL_MODEL_SESSION_PATH = '/home/deeperthought/Projects/Others/2D_brain_segmenter/Sessions/coronal_segmenter_NoDataAug/'


model_sagittal = tf.keras.models.load_model(SAGITTAL_MODEL_SESSION_PATH + 'best_model.h5', 
                                   custom_objects = my_custom_objects)

model_axial = tf.keras.models.load_model(AXIAL_MODEL_SESSION_PATH + 'best_model.h5', 
                                   custom_objects = my_custom_objects)

model_coronal = tf.keras.models.load_model(CORONAL_MODEL_SESSION_PATH + 'best_model.h5', 
                                   custom_objects = my_custom_objects)

#%%

OUT = SAGITTAL_MODEL_SESSION_PATH + 'predictions/'

# Stroke
# SUBJECT_PATH = '/home/deeperthought/kirby/home/AphasicStrokeTrial/NC033/NC033_T1.nii'
# SEGMENTATION_PATH = '/home/deeperthought/kirby/home/AphasicStrokeTrial/NC033/manualSeg.nii'

# Paris
SUBJECT_PATH = '/home/deeperthought/Projects/Others/2D_brain_segmenter/MRIs_and_labels/NIFTYS/Humna_Noor_Human_segmentation_Paris_and_Stroke/Paris_MRI/rAA048_3_MRI_rsc_padded30_T1orT2.nii'
SEGMENTATION_PATH = '/home/deeperthought/Projects/Others/2D_brain_segmenter/MRIs_and_labels/NIFTYS/Humna_Noor_Human_segmentation_Paris_and_Stroke/Humna_Paris/rAA048.nii'

# SUBJECT_PATH = '/home/deeperthought/Projects/Others/2D_brain_segmenter/MRIs_and_labels/NIFTYS/Humna_Noor_Human_segmentation_Paris_and_Stroke/Paris_MRI/rCR137_MRI_rsc_padded30_T1orT2.nii'
# SEGMENTATION_PATH = '/home/deeperthought/Projects/Others/2D_brain_segmenter/MRIs_and_labels/NIFTYS/Humna_Noor_Human_segmentation_Paris_and_Stroke/Humna_Paris/rCR137.nii'

# OTHER
#SUBJECT_PATH = '/home/deeperthought/Projects/MultiPriors/Adam_Buchwald/P908/P908_edit_ras.nii'
#SEGMENTATION_PATH = ''

SUBJECT_NAME = SUBJECT_PATH.split('/')[-1].split('.')[0]


print('Loading and preprocessing {} ..'.format(SUBJECT_NAME))
nii = nib.load(SUBJECT_PATH)
affine = nii.affine
mri = nii.get_data()
mri = resize(mri, output_shape=((mri.shape[0], 256, 256)),  anti_aliasing=True, preserve_range=True)    
mri = mri / np.percentile(mri, 95)

mri = np.random.random((mri.shape))*3

if len(SEGMENTATION_PATH) > 0:
    seg = nib.load(SEGMENTATION_PATH).get_data()     
    seg = np.array(seg, dtype=float)
    np.unique(seg)
    seg = seg - 1   
    
    if seg.shape[-1] < 300:
        seg = resize(seg, output_shape=(seg.shape[0], 256, 256), order=0, anti_aliasing=False)    
    
    else:
        # Gradual resizing seems to work better somehow...
        seg = resize(seg, output_shape=(seg.shape[0], 256, 300), order=0, anti_aliasing=False)    
        seg = resize(seg, output_shape=(seg.shape[0], 256, 275), order=0, anti_aliasing=False)    
        seg = resize(seg, output_shape=(seg.shape[0], 256, 256), order=0, anti_aliasing=False)    
    
    seg.max()
    seg = np.array(seg, dtype=int)

TARGET_WIDTH = 256

padding_width = TARGET_WIDTH - mri.shape[0]

mri_padded = np.pad(mri, ((int(padding_width/2),int(padding_width/2) + padding_width%2),(0,0),(0,0)), 'constant')
if len(SEGMENTATION_PATH) > 0:
    seg_padded = np.pad(seg, ((int(padding_width/2),int(padding_width/2) + padding_width%2),(0,0),(0,0)), 'minimum')
else:
    seg_padded = np.array([])
    
print('Segmenting scan in 3 axis..')
sagittal_segmentation, sagittal_dice, sagittal_segmentation_probs = segment_and_dice(mri_padded, model_sagittal, 'sagittal', seg_padded)
axial_segmentation, axial_dice, axial_segmentation_probs = segment_and_dice(mri_padded, model_axial, 'axial', seg_padded)
coronal_segmentation, coronal_dice, coronal_segmentation_probs = segment_and_dice(mri_padded, model_coronal, 'coronal', seg_padded)


INDEX = 80
mask = 2
plt.imshow(sagittal_segmentation_probs[INDEX,10:-10,10:-10,mask])#, vmin=sagittal_segmentation_probs[INDEX,:,:,mask].min(), vmax=sagittal_segmentation_probs[INDEX,:,:,mask].max()/20)

plt.plot(sagittal_segmentation_probs[INDEX,10:-10,10:-10,mask].reshape(-1))


OUT = SAGITTAL_MODEL_SESSION_PATH + 'predictions/'
print('Saving segmentation in {} ..'.format(OUT + '{}.nii'.format(SUBJECT_NAME)))
if not os.path.exists(OUT):
    os.mkdir(OUT)
nii_out = nib.Nifti1Image(sagittal_segmentation, affine)
nib.save(nii_out, OUT + '{}.nii'.format(SUBJECT_NAME))


OUT = CORONAL_MODEL_SESSION_PATH + 'predictions/'
print('Saving segmentation in {} ..'.format(OUT + '{}.nii'.format(SUBJECT_NAME)))
if not os.path.exists(OUT):
    os.mkdir(OUT)
nii_out = nib.Nifti1Image(coronal_segmentation, affine)
nib.save(nii_out, OUT + '{}.nii'.format(SUBJECT_NAME))


OUT = AXIAL_MODEL_SESSION_PATH + 'predictions/'
print('Saving segmentation in {} ..'.format(OUT + '{}.nii'.format(SUBJECT_NAME)))
if not os.path.exists(OUT):
    os.mkdir(OUT)
nii_out = nib.Nifti1Image(axial_segmentation, affine)
nib.save(nii_out, OUT + '{}.nii'.format(SUBJECT_NAME))





print('Making vote consensus..')
#v1 = np.reshape(sagittal_segmentation, (256*256*256))
#v2 = np.reshape(axial_segmentation, (256*256*256))
#v3 = np.reshape(coronal_segmentation, (256*256*256))
#cons = np.stack([v1,v2,v3],0)
#vote = mode(cons, axis=0)
#vote = vote[0]
#vote_vol = np.reshape(vote, (256,256,256))



cons = np.stack([sagittal_segmentation,axial_segmentation,coronal_segmentation],0)
#vote, _ = mode(cons, axis=0)

vote_vol = np.zeros(sagittal_segmentation.shape)
for x in range(cons.shape[1]):
#    print('{}/{}'.format(x,cons.shape[1]))
    for y in range(cons.shape[2]):
        vote, _ = mode(cons[:,x,y], axis=0)
        vote_vol[x,y] = vote

#v.shape
#plt.imshow(v[100])

#np.sum(v - vote_vol)




OUT = '/home/deeperthought/Projects/Others/2D_brain_segmenter/Sessions/consensus_predictions/' 
print('Saving segmentation in {} ..'.format(OUT + '{}.nii'.format(SUBJECT_NAME)))
if not os.path.exists(OUT):
    os.mkdir(OUT)
nii_out = nib.Nifti1Image(vote_vol, affine)
nib.save(nii_out, OUT + '{}_CONSENSUS.nii'.format(SUBJECT_NAME))

#%% Single slice evaluation - For Paris Data, comparison against Humna and Noor - segmentation on a single slice
#
AXIAL_SLICE = list(set(np.argwhere(seg_padded > 0)[:,-1]))[0]


dice_sagittal = round(Generalised_dice_coef_multilabel_numpy(seg_padded[:,:,AXIAL_SLICE], sagittal_segmentation[:,:,AXIAL_SLICE], 7),3)
dice_axial = round(Generalised_dice_coef_multilabel_numpy(seg_padded[:,:,AXIAL_SLICE], axial_segmentation[:,:,AXIAL_SLICE], 7),3)
dice_coronal = round(Generalised_dice_coef_multilabel_numpy(seg_padded[:,:,AXIAL_SLICE], coronal_segmentation[:,:,AXIAL_SLICE], 7),3)
dice_consensus = round(Generalised_dice_coef_multilabel_numpy(seg_padded[:,:,AXIAL_SLICE], vote_vol[:,:,AXIAL_SLICE], 7),3)

plt.figure(figsize=(10,15))
plt.subplot(3,2,1); plt.imshow(seg_padded[:,:,AXIAL_SLICE],vmin=0, vmax=6); plt.title('Segmentation')
plt.subplot(3,2,2); plt.imshow(sagittal_segmentation[:,:,AXIAL_SLICE],vmin=0, vmax=6); plt.title('Sagittal Dice : {}'.format(dice_sagittal))
plt.subplot(3,2,3); plt.imshow(axial_segmentation[:,:,AXIAL_SLICE],vmin=0, vmax=6); plt.title('Axial Dice : {}'.format(dice_axial))
plt.subplot(3,2,4); plt.imshow(coronal_segmentation[:,:,AXIAL_SLICE],vmin=0, vmax=6); plt.title('Coronal Dice : {}'.format(dice_coronal))
plt.subplot(3,2,5); plt.imshow(vote_vol[:,:,AXIAL_SLICE],vmin=0, vmax=6); plt.title('Consensus Dice : {}'.format(dice_consensus))
plt.savefig(OUT + '{}_ThreeAxis_And_Consensus_Dice_Slice.png'.format(SUBJECT_NAME), dpi=400)

#%%
#
dice_VOTE = Generalised_dice_coef_multilabel_numpy(seg_padded, vote_vol, 7)

OUT = '/home/deeperthought/Projects/Others/2D_brain_segmenter/Sessions/Examples/'


plt.figure(1, figsize=(15,15))
plt.subplot(2,2,1); plt.title('Dice: {}'.format(sagittal_dice))
plt.imshow(sagittal_segmentation[100]); plt.xlabel('Sagittal segmenter')
plt.subplot(2,2,2); plt.title('Dice: {}'.format(axial_dice))
plt.imshow(axial_segmentation[100]); plt.xlabel('Axial segmenter')
plt.subplot(2,2,3); plt.title('Dice: {}'.format(coronal_dice))
plt.imshow(coronal_segmentation[100]); plt.xlabel('Coronal segmenter')
plt.subplot(2,2,4); plt.title('Dice: {}'.format(dice_VOTE))
plt.imshow(vote_vol[100]); plt.xlabel('VOTE segmenter')
plt.savefig(OUT + 'VoteConsensus_NoDataAug_{}_segmentation_sagittal_view.png'.format(SUBJECT_NAME), dpi=400)

plt.figure(2, figsize=(15,15))
plt.subplot(2,2,1); plt.title('Dice: {}'.format(sagittal_dice))
plt.imshow(sagittal_segmentation[:,100]); plt.xlabel('Sagittal segmenter')
plt.subplot(2,2,2); plt.title('Dice: {}'.format(axial_dice))
plt.imshow(axial_segmentation[:,100]); plt.xlabel('Axial segmenter')
plt.subplot(2,2,3); plt.title('Dice: {}'.format(coronal_dice))
plt.imshow(coronal_segmentation[:,100]); plt.xlabel('Coronal segmenter')
plt.subplot(2,2,4); plt.title('Dice: {}'.format(dice_VOTE))
plt.imshow(vote_vol[:,100]); plt.xlabel('VOTE segmenter')
plt.savefig(OUT + 'VoteConsensus_NoDataAug_{}_segmentation_coronal_view.png'.format(SUBJECT_NAME), dpi=400)

plt.figure(3, figsize=(15,15))
plt.subplot(2,2,1); plt.title('Dice: {}'.format(sagittal_dice))
plt.imshow(sagittal_segmentation[:,:,170]); plt.xlabel('Sagittal segmenter')
plt.subplot(2,2,2); plt.title('Dice: {}'.format(axial_dice))
plt.imshow(axial_segmentation[:,:,170]); plt.xlabel('Axial segmenter')
plt.subplot(2,2,3); plt.title('Dice: {}'.format(coronal_dice))
plt.imshow(coronal_segmentation[:,:,170]); plt.xlabel('Coronal segmenter')
plt.subplot(2,2,4); plt.title('Dice: {}'.format(dice_VOTE))
plt.imshow(vote_vol[:,:,170]); plt.xlabel('VOTE segmenter')

plt.savefig(OUT + 'VoteConsensus_NoDataAug_{}_segmentation_axial_view.png'.format(SUBJECT_NAME), dpi=400)

nii_out = nib.Nifti1Image(vote_vol, affine)
nib.save(nii_out, OUT + 'VoteConsensus_NoDataAug_{}_segmentation.nii'.format(SUBJECT_NAME))
