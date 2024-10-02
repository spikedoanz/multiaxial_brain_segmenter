#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 13:37:39 2024

@author: deeperthought
"""


import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.transform import resize
from scipy.stats import mode
import tensorflow as tf
os.chdir('/home/deeperthought/Projects/Others/2D_brain_segmenter/')

from utils import dice_loss, Generalised_dice_coef_multilabel7,dice_coef_multilabel_bin0,dice_coef_multilabel_bin1, dice_coef_multilabel_bin2,dice_coef_multilabel_bin3
from utils import dice_coef_multilabel_bin4,dice_coef_multilabel_bin5,dice_coef_multilabel_bin6

GPU = 1

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
        tf.config.experimental.set_visible_devices(gpus[GPU], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[GPU], True)
      except RuntimeError as e:
        # Visible devices must be set at program startup
        print(e)

#%% USER INPUT

SUBJECT_PATH = '/home/deeperthought/Projects/MultiPriors/Adam_Buchwald/P910/P910.nii'
# SUBJECT_PATH = '/home/deeperthought/Projects/Others/2D_brain_segmenter/MRIs_and_labels/NIFTYS/Normal_heads/Andy.nii'
SEGMENTATION_PATH = ''

OUTPUT_PATH = '/home/deeperthought/Projects/MultiPriors/Adam_Buchwald/P910/'
# OUTPUT_PATH = '/home/deeperthought/Projects/Others/2D_brain_segmenter/MRIs_and_labels/NIFTYS/Normal_heads/'

SAGITTAL_MODEL_SESSION_PATH = '/home/deeperthought/Projects/Others/2D_brain_segmenter/Sessions/sagittal_segmenter_NoDataAug/best_model.h5'
AXIAL_MODEL_SESSION_PATH = '/home/deeperthought/Projects/Others/2D_brain_segmenter/Sessions/axial_segmenter_NoDataAug/best_model.h5'
CORONAL_MODEL_SESSION_PATH = '/home/deeperthought/Projects/Others/2D_brain_segmenter/Sessions/coronal_segmenter_NoDataAug/best_model.h5'

SEGMENTATION_METHOD = 2

#%% METRICS AND LOSSES
        

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
mri = nii.get_fdata()
mri_shape = mri.shape
mri = resize(mri, output_shape=((mri.shape[0], 256, 256)),  anti_aliasing=True, preserve_range=True)    

mri = mri / np.percentile(mri, 95)


# minval = np.percentile(mri, 0)
# maxval = np.percentile(mri, 100)
# mri = np.clip(mri, minval, maxval)
# mri = ((mri - minval) / (maxval - minval))# * 255


mri.max()
mri.min()



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
    
#%%
sagittal_segmentation = model_sagittal.predict([np.random.random((256,256,256,1)), np.arange(256)], batch_size=1)

sagittal_segmentation = np.argmax(sagittal_segmentation,-1)
sagittal_segmentation = np.array(sagittal_segmentation, dtype='int16')

SLICE = 100
plt.imshow(np.rot90(sagittal_segmentation[SLICE])); plt.title('Sagittal segmenter')

plt.imshow(np.rot90(sagittal_segmentation[:,SLICE])); plt.title('Sagittal segmenter')


#%%


model_sagittal.layers

#%%


emb = model_sagittal.layers[-12].get_weights()[0]

emb.shape
plt.imshow(emb, aspect='auto')
    

distance_matrix = np.linalg.norm(emb[:, None] - emb[None, :], axis=2)

# Ensure the distance matrix is symmetrical (optional)
# distance_matrix = np.triu(distance_matrix) + np.tril(distance_matrix.T, k=1)

# Print the correlation matrix (distances are negative for correlation)
print(distance_matrix)

plt.imshow(distance_matrix, aspect='auto')

#%%


print('Segmenting scan in 3 axis..')
if SEGMENTATION_METHOD == 1:
    sagittal_segmentation, sagittal_segmentation_probs = segment_in_axis(mri_padded, model_sagittal, 'sagittal')
    axial_segmentation, axial_segmentation_probs = segment_in_axis(mri_padded, model_axial, 'axial')
    coronal_segmentation, coronal_segmentation_probs = segment_in_axis(mri_padded, model_coronal, 'coronal')
    
elif SEGMENTATION_METHOD == 2:
    sagittal_segmentation = model_sagittal.predict([np.expand_dims(mri_padded,-1), np.arange(256)], batch_size=1)
    coronal_segmentation = model_coronal.predict(np.expand_dims(np.swapaxes(mri_padded, 0, 1),-1), batch_size=1)
    axial_segmentation = model_axial.predict(np.expand_dims(np.swapaxes(np.swapaxes(mri_padded, 1,2), 0,1),-1), batch_size=1)
    
    sagittal_segmentation = np.argmax(sagittal_segmentation,-1)
    coronal_segmentation = np.swapaxes(np.argmax(coronal_segmentation,-1),0,1)
    axial_segmentation = np.swapaxes(np.swapaxes(np.argmax(axial_segmentation,-1),0,1), 1,2)

# Remove padding
if padding_width > 0:
    sagittal_segmentation = sagittal_segmentation[int(padding_width/2):-(int(padding_width/2) + padding_width%2)]
    axial_segmentation = axial_segmentation[int(padding_width/2):-(int(padding_width/2) + padding_width%2)]
    coronal_segmentation = coronal_segmentation[int(padding_width/2):-(int(padding_width/2) + padding_width%2)]

# Resize to original
sagittal_segmentation = resize(sagittal_segmentation, output_shape=mri_shape, order=0, anti_aliasing=True, preserve_range=True)    
axial_segmentation = resize(axial_segmentation, output_shape=mri_shape, order=0, anti_aliasing=True, preserve_range=True)    
coronal_segmentation = resize(coronal_segmentation, output_shape=mri_shape, order=0, anti_aliasing=True, preserve_range=True)    

sagittal_segmentation = np.array(sagittal_segmentation, dtype='int16')
axial_segmentation = np.array(axial_segmentation, dtype='int16')
coronal_segmentation = np.array(coronal_segmentation, dtype='int16')

# print('Making vote consensus..')
# vote_vol = np.zeros(sagittal_segmentation.shape)
# equals = np.logical_and( (sagittal_segmentation==axial_segmentation), (axial_segmentation==coronal_segmentation) )
# vote_vol[equals == 1] = sagittal_segmentation[equals == 1]
# sagittal_needs_consensus_vector = sagittal_segmentation[equals == 0]
# axial_needs_consensus_vector = axial_segmentation[equals == 0]
# coronal_needs_consensus_vector = coronal_segmentation[equals == 0]
# needs_consensus_vector = np.stack([sagittal_needs_consensus_vector, axial_needs_consensus_vector, coronal_needs_consensus_vector],0)
# vote_vector = mode(needs_consensus_vector, axis=0, keepdims=True)
# vote_vol[equals == 0] = vote_vector[0][0]


SLICE = 96

plt.figure(figsize=(15,10))
plt.subplot(2,3,1); plt.imshow(np.rot90(mri[SLICE]), cmap='gray')
plt.subplot(2,3,2); plt.imshow(np.rot90(sagittal_segmentation[SLICE])); plt.title('Sagittal segmenter')
plt.subplot(2,3,3); plt.imshow(np.rot90(axial_segmentation[SLICE])); plt.title('Axial segmenter')
plt.subplot(2,3,5); plt.imshow(np.rot90(coronal_segmentation[SLICE])); plt.title('Coronal segmenter')
# plt.subplot(2,3,6); plt.imshow(np.rot90(vote_vol[SLICE])); plt.title('Majority vote segmenter')


if len(OUTPUT_PATH) > 0:
    print('Saving segmentation in {} ..'.format(OUTPUT_PATH + '{}.nii'.format(SUBJECT_NAME)))
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    nii_out = nib.Nifti1Image(sagittal_segmentation, affine)
    nib.save(nii_out, OUTPUT_PATH + os.sep + '{}_sagittal.nii'.format(SUBJECT_NAME))
    
    nii_out = nib.Nifti1Image(coronal_segmentation, affine)
    nib.save(nii_out, OUTPUT_PATH + os.sep + '{}_coronal.nii'.format(SUBJECT_NAME))
    
    nii_out = nib.Nifti1Image(axial_segmentation, affine)
    nib.save(nii_out, OUTPUT_PATH + os.sep + '{}_axial.nii'.format(SUBJECT_NAME))

    # nii_out = nib.Nifti1Image(vote_vol, affine)
    # nib.save(nii_out, OUTPUT_PATH + os.sep + '{}_CONSENSUS.nii'.format(SUBJECT_NAME))