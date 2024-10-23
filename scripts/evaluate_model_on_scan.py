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


OUTPUT_PATH = '/home/deeperthought/Projects/DGNS/Detection_model/Sessions/'

scan_path = '/media/HDD/MultiAxial/Data/AdamBuchwald/Adams_Manual_Fixed/MRI/P904.nii'
segmentation_path = None
anterior_commissure =[ 96,	143,	150]

    
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

    nii = nib.load(scan_path)
    if segmentation_path is not None:
        nii_seg = nib.load(segmentation_path)
    else:
        nii_seg = None
    print(nii.shape)
    
    subject = scan_path.split('/')[-1].split('_')[0].replace('.nii','')
    
    if subject.startswith('r'):
        subject = subject[1:]
    
    nii_out, nii_seg_out, coords, anterior_commissure, reconstruction_parms = preprocess_head_MRI(nii, nii_seg, anterior_commissure=anterior_commissure, keep_parameters_for_reconstruction=True)     


    

    # from utils import Generalised_dice_coef_multilabel7, generalized_dice_coef_multilabel7_present_class, dice_metric0, dice_metric1, dice_metric2, dice_metric3, dice_metric4, dice_metric5, dice_metric6
    # from tensorflow.keras.optimizers import Adam
    

    # model_sagittal.compile(loss=Generalised_dice_coef_multilabel7, 
    #           optimizer=Adam(lr=1e-5), 
    #           metrics=[dice_metric0, dice_metric1, dice_metric2, dice_metric3, dice_metric4, dice_metric5, dice_metric6]) 
    
    #                     # dice_metric1, 
    #                     # dice_metric2,
    #                     # dice_metric3,
    #                     # dice_metric4,
    #                     # dice_metric5,
    #                     # dice_metric6])    
    
    # ypred = model_sagittal.predict([np.random.random((1,256,256,1)), np.random.random((1,256,256,3))])
    # model_sagittal.evaluate([np.random.random((1,256,256,1)), np.random.random((1,256,256,3))], ypred)
    

    model_segmentation_sagittal, model_segmentation_coronal, model_segmentation_axial = segment_MRI(nii_out.get_fdata(), coords, model_sagittal,model_coronal, model_axial)

    nii_model_seg_reconstructed = reshape_back_to_original(model_segmentation_sagittal, nii, reconstruction_parms, resample_order=0)


    INDEX = 120
    plt.subplot(221)
    plt.imshow(np.rot90(nii_out.get_fdata()[INDEX]))
    plt.subplot(222)
    plt.imshow(np.rot90(model_segmentation_sagittal[INDEX]))
    plt.subplot(223)
    plt.imshow(np.rot90(model_segmentation_coronal[INDEX]))
    plt.subplot(224)
    plt.imshow(np.rot90(model_segmentation_axial[INDEX]))
    

    # Making vote consensus
    print('Making vote consensus...')
    vote_vol = np.zeros(model_segmentation_sagittal.shape, dtype=np.uint8)  # Initialize with zeros
    equals = np.logical_and((model_segmentation_sagittal == model_segmentation_axial), 
                            (model_segmentation_axial == model_segmentation_coronal))
    
    
    # Assign values where models agree
    vote_vol[equals] = model_segmentation_sagittal[equals]
    
    # Get vectors that need consensus
    sagittal_needs_consensus_vector = model_segmentation_sagittal[~equals]
    axial_needs_consensus_vector = model_segmentation_axial[~equals]
    coronal_needs_consensus_vector = model_segmentation_coronal[~equals]
    
    # Stack vectors for voting
    needs_consensus_vector = np.stack([sagittal_needs_consensus_vector, 
                                        axial_needs_consensus_vector, 
                                        coronal_needs_consensus_vector], axis=0)
    
    # Get the mode of the consensus vectors
    import scipy
    vote_vector = scipy.stats.mode(needs_consensus_vector, axis=0)
    vote_vol[~equals] = vote_vector[0][0]
    
    plt.figure(figsize=(10,10))

    plt.subplot(221)
    plt.imshow(model_segmentation_sagittal[100]); plt.xticks([]); plt.yticks([])
    plt.subplot(222)
    plt.imshow(model_segmentation_coronal[100]); plt.xticks([]); plt.yticks([])
    plt.subplot(223)
    plt.imshow(model_segmentation_axial[100]); plt.xticks([]); plt.yticks([])
    plt.subplot(224)
    plt.imshow(vote_vol[100]); plt.xticks([]); plt.yticks([])
    
    if save_segmentation:
        
        if not os.path.exists(SAGITTAL_MODEL_SESSION_PATH + '/predictions/'):
            os.mkdir(SAGITTAL_MODEL_SESSION_PATH + '/predictions/')
        nii_out_pred = nib.Nifti1Image(np.array(model_segmentation_sagittal, dtype='int16'), nii_out.affine)
        nib.save(nii_out_pred, SAGITTAL_MODEL_SESSION_PATH + '/predictions/' + subject + '_sagittal_segmentation.nii')

        if not os.path.exists(CORONAL_MODEL_SESSION_PATH + '/predictions/'):
            os.mkdir(CORONAL_MODEL_SESSION_PATH + '/predictions/')
        nii_out_pred = nib.Nifti1Image(np.array(model_segmentation_coronal, dtype='int16'), nii_out.affine)
        nib.save(nii_out_pred, CORONAL_MODEL_SESSION_PATH + '/predictions/' + subject + '_coronal_segmentation.nii')       
        
        
        if not os.path.exists(AXIAL_MODEL_SESSION_PATH + '/predictions/'):
            os.mkdir(AXIAL_MODEL_SESSION_PATH + '/predictions/')
        nii_out_pred = nib.Nifti1Image(np.array(model_segmentation_axial, dtype='int16'), nii_out.affine)
        nib.save(nii_out_pred, AXIAL_MODEL_SESSION_PATH + '/predictions/' + subject + '_axial_segmentation.nii')        
        
