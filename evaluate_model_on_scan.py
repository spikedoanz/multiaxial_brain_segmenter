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
from scripts.utils import segment_MRI


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



OUTPUT_PATH = 'output.nii.gz'

scan_path = 'input.nii.gz'

SAGITTAL_MODEL_SESSION_PATH = 'models/sagittal_model.h5'
AXIAL_MODEL_SESSION_PATH = 'models/axial_model.h5'
CORONAL_MODEL_SESSION_PATH = 'models/coronal_model.h5'
CONSENSUS_LAYER_PATH = 'models/consensus_layer.h5'

segmentation_path = None
anterior_commissure = None 
    
save_segmentation = False

if __name__ == '__main__':
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    from scripts.preprocessing_lib import preprocess_head_MRI, reshape_back_to_original
    

    
    
            
    if SAGITTAL_MODEL_SESSION_PATH is not None:
        model_sagittal = tf.keras.models.load_model(SAGITTAL_MODEL_SESSION_PATH)
    if AXIAL_MODEL_SESSION_PATH is not None:
        model_axial = tf.keras.models.load_model(AXIAL_MODEL_SESSION_PATH)
    if CORONAL_MODEL_SESSION_PATH is not None:
        model_coronal = tf.keras.models.load_model(CORONAL_MODEL_SESSION_PATH) 
    consensus_model = tf.keras.models.load_model(CONSENSUS_LAYER_PATH)

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
    nib.save(nii_out_pred, OUTPUT_PATH)    

        
    
