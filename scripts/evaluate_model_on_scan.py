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
import matplotlib.pyplot as plt


GPU = 1

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


# scan_path = '/media/HDD/MultiAxial/Data/Processed_New_MCS/MRI/PA020_1mm_ras.nii'
# segmentation_path = '/media/HDD/MultiAxial/Data/Processed_New_MCS/GT/PA020_segmentation_fixed.nii'
# anterior_commissure = [88, 135, 157]


OUTPUT_PATH = '/media/HDD/MultiAxial/Data/AdamBuchwald/Original/multiaxial/'

scan_path = '/media/HDD/MultiAxial/Data/AdamBuchwald/Original/P910_+05+T1_mprage_ns_sag_p2_iso_1mm_192.nii'
segmentation_path = None
anterior_commissure = None #[ 96,	143,	150]

    
save_segmentation = False

if __name__ == '__main__':
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    from preprocessing_lib import preprocess_head_MRI, reshape_back_to_original
    
    from utils import segment_MRI, Generalised_dice_coef_multilabel7, dice_coef_multilabel_bin0, dice_coef_multilabel_bin1, dice_coef_multilabel_bin2, dice_coef_multilabel_bin3, dice_coef_multilabel_bin4, dice_coef_multilabel_bin5, dice_coef_multilabel_bin6
    

    
    
    my_custom_objects = {'Generalised_dice_coef_multilabel7':Generalised_dice_coef_multilabel7,
                                     'dice_coef_multilabel_bin0':dice_coef_multilabel_bin0,
                                     'dice_coef_multilabel_bin1':dice_coef_multilabel_bin1,
                                     'dice_coef_multilabel_bin2':dice_coef_multilabel_bin2,
                                     'dice_coef_multilabel_bin3':dice_coef_multilabel_bin3,
                                     'dice_coef_multilabel_bin4':dice_coef_multilabel_bin4,
                                     'dice_coef_multilabel_bin5':dice_coef_multilabel_bin5,
                                     'dice_coef_multilabel_bin6':dice_coef_multilabel_bin6}
    
    SAGITTAL_MODEL_SESSION_PATH = '/home/deeperthought/Projects/Multiaxial/Sessions/HighL2_ShuffleVal_NoEmptySlices_OldDiceMetric_experiment_sagittalSegmenter_PositionalEncoding_100epochs_depth6_baseFilters16_AndrewPartition_Step2/'
    AXIAL_MODEL_SESSION_PATH = '/home/deeperthought/Projects/Multiaxial/Sessions/axialSegmenter_PositionalEncoding_100epochs_depth6_baseFilters16_AndrewPartition_Step2/'
    CORONAL_MODEL_SESSION_PATH = '/home/deeperthought/Projects/Multiaxial/Sessions/DiceLossMask_experiment_coronalSegmenter_PositionalEncoding_100epochs_depth6_baseFilters8_AndrewPartition_Step2/'
    


            
    if SAGITTAL_MODEL_SESSION_PATH is not None:
        model_sagittal = tf.keras.models.load_model(SAGITTAL_MODEL_SESSION_PATH + 'best_model.h5', 
                                       custom_objects = my_custom_objects)
    if AXIAL_MODEL_SESSION_PATH is not None:
        model_axial = tf.keras.models.load_model(AXIAL_MODEL_SESSION_PATH + 'best_model.h5', 
                                           custom_objects = my_custom_objects)
    if CORONAL_MODEL_SESSION_PATH is not None:
        model_coronal = tf.keras.models.load_model(CORONAL_MODEL_SESSION_PATH + 'best_model.h5', 
                                           custom_objects = my_custom_objects)           

    consensus_model = tf.keras.models.load_model('/home/deeperthought/Projects/Multiaxial/Sessions/Consensus_model_sessions/consensus_layer_fold2.h5')

    nii = nib.load(scan_path)
    if segmentation_path is not None:
        nii_seg = nib.load(segmentation_path)
    else:
        nii_seg = None
    print(nii.shape)
    
    subject = scan_path.split('/')[-1].replace('.nii','')
    
    if subject.startswith('r'):
        subject = subject[1:]
    
    nii_out, nii_seg_out, coords, anterior_commissure, reconstruction_parms = preprocess_head_MRI(nii, nii_seg, anterior_commissure=anterior_commissure, keep_parameters_for_reconstruction=True)     

    

    segmentation = segment_MRI(nii_out.get_fdata(), coords, model_sagittal,model_coronal, model_axial, consensus_model)

    # nii_model_seg_reconstructed = reshape_back_to_original(model_segmentation_sagittal, nii, reconstruction_parms, resample_order=0)
    
    nii_out_pred = nib.Nifti1Image(np.array(segmentation, dtype='int16'), nii_out.affine)
    nib.save(nii_out_pred, OUTPUT_PATH + subject + '_consensus_segmentation.nii')    

        
    

    # if save_segmentation:
        
    #     if not os.path.exists(SAGITTAL_MODEL_SESSION_PATH + '/predictions/'):
    #         os.mkdir(SAGITTAL_MODEL_SESSION_PATH + '/predictions/')
    #     nii_out_pred = nib.Nifti1Image(np.array(model_segmentation_sagittal, dtype='int16'), nii_out.affine)
    #     nib.save(nii_out_pred, SAGITTAL_MODEL_SESSION_PATH + '/predictions/' + subject + '_sagittal_segmentation.nii')

    #     if not os.path.exists(CORONAL_MODEL_SESSION_PATH + '/predictions/'):
    #         os.mkdir(CORONAL_MODEL_SESSION_PATH + '/predictions/')
    #     nii_out_pred = nib.Nifti1Image(np.array(model_segmentation_coronal, dtype='int16'), nii_out.affine)
    #     nib.save(nii_out_pred, CORONAL_MODEL_SESSION_PATH + '/predictions/' + subject + '_coronal_segmentation.nii')       
        
        
    #     if not os.path.exists(AXIAL_MODEL_SESSION_PATH + '/predictions/'):
    #         os.mkdir(AXIAL_MODEL_SESSION_PATH + '/predictions/')
    #     nii_out_pred = nib.Nifti1Image(np.array(model_segmentation_axial, dtype='int16'), nii_out.affine)
    #     nib.save(nii_out_pred, AXIAL_MODEL_SESSION_PATH + '/predictions/' + subject + '_axial_segmentation.nii')        
        
