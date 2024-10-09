#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 15:32:10 2023

@author: deeperthought
"""

import os
import numpy as np
import nibabel as nib
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt



GPU = 0

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[GPU], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[GPU], True)
  except RuntimeError as e:
    # Visible devices must be set at program startup
    print(e)


scan = '/media/HDD/MultiAxial/Data/Processed_New_MCS/MRI/PA020_1mm_ras.nii'
segmentation = '/media/HDD/MultiAxial/Data/Processed_New_MCS/GT/PA020_segmentation_fixed.nii'
anterior_commissure = [88, 135, 157]


# scan = '/media/HDD/MultiAxial/Data/NormalHeads/MRI/Parra.nii'
# segmentation = None
# anterior_commissure = [102,	141, 163]

    
save_segmentation = False

if __name__ == '__main__':
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    from preprocessing_lib import preprocess_head_MRI, reshape_back_to_original
    
    from utils import Generalised_dice_coef_multilabel7, dice_coef_multilabel_bin0, dice_coef_multilabel_bin1, dice_coef_multilabel_bin2, dice_coef_multilabel_bin3, dice_coef_multilabel_bin4, dice_coef_multilabel_bin5, dice_coef_multilabel_bin6
    

    OUTPUT_PATH = '/home/deeperthought/Projects/DGNS/Detection_model/Sessions/'
    
    
    my_custom_objects = {'Generalised_dice_coef_multilabel7':Generalised_dice_coef_multilabel7,
                                     'dice_coef_multilabel_bin0':dice_coef_multilabel_bin0,
                                     'dice_coef_multilabel_bin1':dice_coef_multilabel_bin1,
                                     'dice_coef_multilabel_bin2':dice_coef_multilabel_bin2,
                                     'dice_coef_multilabel_bin3':dice_coef_multilabel_bin3,
                                     'dice_coef_multilabel_bin4':dice_coef_multilabel_bin4,
                                     'dice_coef_multilabel_bin5':dice_coef_multilabel_bin5,
                                     'dice_coef_multilabel_bin6':dice_coef_multilabel_bin6}
    
    SAGITTAL_MODEL_SESSION_PATH = '/home/deeperthought/Projects/Multiaxial/Sessions/sagittalSegmenter_PositionalEncoding_100epochs_depth6_baseFilters16/'
    AXIAL_MODEL_SESSION_PATH = '/home/deeperthought/Projects/Multiaxial/Sessions/axialSegmenter_PositionalEncoding_100epochs_depth6_baseFilters16/'
    CORONAL_MODEL_SESSION_PATH = None #'/home/deeperthought/Projects/Others/2D_brain_segmenter/Sessions/coronal_segmenter_NoDataAug/'
    
    
    model_sagittal = tf.keras.models.load_model(SAGITTAL_MODEL_SESSION_PATH + 'best_model.h5', 
                                       custom_objects = my_custom_objects)
    if AXIAL_MODEL_SESSION_PATH is not None:
        model_axial = tf.keras.models.load_model(AXIAL_MODEL_SESSION_PATH + 'best_model.h5', 
                                           custom_objects = my_custom_objects)
    if CORONAL_MODEL_SESSION_PATH is not None:
        model_coronal = tf.keras.models.load_model(CORONAL_MODEL_SESSION_PATH + 'best_model.h5', 
                                           custom_objects = my_custom_objects)
           
    print(scan)
    nii = nib.load(scan)
    if segmentation is not None:
        nii_seg = nib.load(segmentation)
    else:
        nii_seg = None
    print(nii.shape)
    
    subject = scan.split('/')[-1].split('_')[0].replace('.nii','')
    
    if subject.startswith('r'):
        subject = subject[1:]
    
    nii_out, nii_seg_out, coords, anterior_commissure, reconstruction_parms = preprocess_head_MRI(nii, nii_seg, anterior_commissure=anterior_commissure, keep_parameters_for_reconstruction=True)  

    img = nii_out.get_fdata()
    p95 = np.percentile(img, 95)
    img = img/p95
    img = np.expand_dims(img, axis=-1)
    
    coords = coords[:,:,:,:3]
    coords = coords/256.
    
    #----- model prediction -------    
    yhat_sagittal = model_sagittal.predict([img, coords], batch_size=1)
    yhat_axial = model_axial.predict([np.swapaxes(np.swapaxes(img, 1,2), 0,1), np.swapaxes(np.swapaxes(coords, 1,2), 0,1)], batch_size=1)
        
    model_segmentation_sagittal = np.argmax(yhat_sagittal, axis=-1)            
    model_segmentation_axial = np.swapaxes(np.swapaxes(np.argmax(yhat_axial,-1),0,1), 1,2)
    
        
    nii_reconstructed = reshape_back_to_original(nii_out.get_fdata(), nii, reconstruction_parms)

    nii_seg_reconstructed = reshape_back_to_original(nii_seg_out.get_fdata(), nii_seg, reconstruction_parms, resample_order=0)

    nii_model_seg_reconstructed = reshape_back_to_original(model_segmentation_sagittal, nii_seg, reconstruction_parms, resample_order=0)


    # nii_reconstructed.shape

    # img1 = nii.get_fdata()
    # img2 = nii_reconstructed.get_fdata()
    
    # seg1 = nii_seg.get_fdata()
    # seg2 = nii_seg_reconstructed.get_fdata()
    # seg3 = nii_model_seg_reconstructed.get_fdata()
    
    # np.std(img1-img2)
    
    # np.std(seg1-seg2)
      
    
    # plt.imshow(seg1[100])
    # plt.imshow(seg2[100])
    # plt.imshow(seg3[100])
        
    
    
    INDEX = 120
    plt.subplot(131)
    plt.imshow(np.rot90(img[INDEX]))
    plt.subplot(132)
    plt.imshow(np.rot90(model_segmentation_sagittal[INDEX]))
    plt.subplot(133)
    plt.imshow(np.rot90(model_segmentation_axial[INDEX]))

    if save_segmentation:
        if not os.path.exists(SAGITTAL_MODEL_SESSION_PATH + '/predictions/'):
            os.mkdir(SAGITTAL_MODEL_SESSION_PATH + '/predictions/')
        nii_out_pred = nib.Nifti1Image(np.array(model_segmentation_sagittal, dtype='int16'), nii_out.affine)
        nib.save(nii_out_pred, SAGITTAL_MODEL_SESSION_PATH + '/predictions/' + subject + '_segmentation.nii')
        if not os.path.exists(AXIAL_MODEL_SESSION_PATH + '/predictions/'):
            os.mkdir(AXIAL_MODEL_SESSION_PATH + '/predictions/')
        nii_out_pred = nib.Nifti1Image(np.array(model_segmentation_axial, dtype='int16'), nii_out.affine)
        nib.save(nii_out_pred, AXIAL_MODEL_SESSION_PATH + '/predictions/' + subject + '_segmentation.nii')        