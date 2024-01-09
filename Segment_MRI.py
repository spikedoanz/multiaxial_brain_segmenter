#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 13:37:39 2024

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
from tensorflow.keras import backend as K
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

#%% USER INPUT

SUBJECT_PATH = ''
SEGMENTATION_PATH = ''

OUTPUT_PATH = ''

SAGITTAL_MODEL_SESSION_PATH = ''
AXIAL_MODEL_SESSION_PATH = ''
CORONAL_MODEL_SESSION_PATH = ''

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


def segment_in_axis(mri_padded, model, orientation='sagittal'):
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
    
    return segmentation, segmentation_probs

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
print('Loading models..')
my_custom_objects = {'Generalised_dice_coef_multilabel7':Generalised_dice_coef_multilabel7,
                                 'dice_coef_multilabel_bin0':dice_coef_multilabel_bin0,
                                 'dice_coef_multilabel_bin1':dice_coef_multilabel_bin1,
                                 'dice_coef_multilabel_bin2':dice_coef_multilabel_bin2,
                                 'dice_coef_multilabel_bin3':dice_coef_multilabel_bin3,
                                 'dice_coef_multilabel_bin4':dice_coef_multilabel_bin4,
                                 'dice_coef_multilabel_bin5':dice_coef_multilabel_bin5,
                                 'dice_coef_multilabel_bin6':dice_coef_multilabel_bin6}


model_sagittal = tf.keras.models.load_model(SAGITTAL_MODEL_SESSION_PATH, 
                                   custom_objects = my_custom_objects)

model_axial = tf.keras.models.load_model(AXIAL_MODEL_SESSION_PATH,
                                   custom_objects = my_custom_objects)

model_coronal = tf.keras.models.load_model(CORONAL_MODEL_SESSION_PATH,
                                   custom_objects = my_custom_objects)

#%%

SUBJECT_NAME = SUBJECT_PATH.split(os.sep)[-1].split('.')[0]

print('Loading and preprocessing {} ..'.format(SUBJECT_NAME))
nii = nib.load(SUBJECT_PATH)
affine = nii.affine
mri = nii.get_data()
mri_shape = mri.shape
mri = resize(mri, output_shape=((mri.shape[0], 256, 256)),  anti_aliasing=True, preserve_range=True)    
mri = mri / np.percentile(mri, 95)

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
#sagittal_segmentation, sagittal_segmentation_probs = segment_in_axis(mri_padded, model_sagittal, 'sagittal')
#axial_segmentation, axial_segmentation_probs = segment_in_axis(mri_padded, model_axial, 'axial')
#coronal_segmentation, coronal_segmentation_probs = segment_in_axis(mri_padded, model_coronal, 'coronal')

sagittal_segmentation = model_sagittal.predict(np.expand_dims(mri_padded,-1), batch_size=1)
coronal_segmentation = model_coronal.predict(np.expand_dims(np.swapaxes(mri_padded, 0, 1),-1), batch_size=1)
axial_segmentation = model_axial.predict(np.expand_dims(np.swapaxes(np.swapaxes(mri_padded, 1,2), 0,1),-1), batch_size=1)
sagittal_segmentation = np.argmax(sagittal_segmentation,-1)
coronal_segmentation = np.swapaxes(np.argmax(coronal_segmentation,-1),0,1)
axial_segmentation = np.swapaxes(np.swapaxes(np.argmax(axial_segmentation,-1),0,1), 1,2)

# Remove padding
sagittal_segmentation = sagittal_segmentation[(int(padding_width/2):-(int(padding_width/2) + padding_width%2)]
axial_segmentation = axial_segmentation[(int(padding_width/2):-(int(padding_width/2) + padding_width%2)]
coronal_segmentation = coronal_segmentation[(int(padding_width/2):-(int(padding_width/2) + padding_width%2)]

# Resize to original
sagittal_segmentation = resize(sagittal_segmentation, output_shape=mri_shape, order=0, anti_aliasing=True, preserve_range=True)    
axial_segmentation = resize(axial_segmentation, output_shape=mri_shape, order=0, anti_aliasing=True, preserve_range=True)    
coronal_segmentation = resize(coronal_segmentation, output_shape=mri_shape, order=0, anti_aliasing=True, preserve_range=True)    


print('Saving segmentation in {} ..'.format(OUTPUT_PATH + '{}.nii'.format(SUBJECT_NAME)))
if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
nii_out = nib.Nifti1Image(sagittal_segmentation, affine)
nib.save(nii_out, OUTPUT_PATH + os.sep + '{}_sagittal.nii'.format(SUBJECT_NAME))

nii_out = nib.Nifti1Image(coronal_segmentation, affine)
nib.save(nii_out, OUTPUT_PATH + os.sep + '{}_coronal.nii'.format(SUBJECT_NAME))

nii_out = nib.Nifti1Image(axial_segmentation, affine)
nib.save(nii_out, OUTPUT_PATH + os.sep + '{}_axial.nii'.format(SUBJECT_NAME))

print('Making vote consensus..')
vote_vol = np.zeros(sagittal_segmentation.shape)
equals = np.logical_and( (sagittal_segmentation==axial_segmentation), (axial_segmentation==coronal_segmentation) )
vote_vol[equals == 1] = sagittal_segmentation[equals == 1]
sagittal_needs_consensus_vector = sagittal_segmentation[equals == 0]
axial_needs_consensus_vector = axial_segmentation[equals == 0]
coronal_needs_consensus_vector = coronal_segmentation[equals == 0]
needs_consensus_vector = np.stack([sagittal_needs_consensus_vector, axial_needs_consensus_vector, coronal_needs_consensus_vector],0)
vote_vector = mode(needs_consensus_vector, axis=0)
vote_vol[equals == 0] = vote_vector[0][0]


nii_out = nib.Nifti1Image(vote_vol, affine)
nib.save(nii_out, OUTPUT_PATH + os.sep + '{}_CONSENSUS.nii'.format(SUBJECT_NAME))
