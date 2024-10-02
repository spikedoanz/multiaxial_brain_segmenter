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



if __name__ == '__main__':
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    from preprocessing_lib import preprocess_head_MRI
    
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
    
    SAGITTAL_MODEL_SESSION_PATH = '/home/deeperthought/Projects/Others/2D_brain_segmenter/Sessions/sagittal_segmenter_NoDataAug/'
    AXIAL_MODEL_SESSION_PATH = '/home/deeperthought/Projects/Others/2D_brain_segmenter/Sessions/axial_segmenter_NoDataAug/'
    CORONAL_MODEL_SESSION_PATH = '/home/deeperthought/Projects/Others/2D_brain_segmenter/Sessions/coronal_segmenter_NoDataAug/'
    
    
    model_sagittal = tf.keras.models.load_model(SAGITTAL_MODEL_SESSION_PATH + 'best_model.h5', 
                                       custom_objects = my_custom_objects)
    
    model_axial = tf.keras.models.load_model(AXIAL_MODEL_SESSION_PATH + 'best_model.h5', 
                                       custom_objects = my_custom_objects)
    
    model_coronal = tf.keras.models.load_model(CORONAL_MODEL_SESSION_PATH + 'best_model.h5', 
                                       custom_objects = my_custom_objects)
        
    PATH = '/media/HDD/MultiAxial/Data/Processed_New_MCS/MRI/'
    SEG_PATH = '/media/HDD/MultiAxial/Data/Processed_New_MCS/GT/'
    anterior_commissure_df = pd.read_csv('/media/HDD/MultiAxial/Data/Processed_New_MCS_AnteriorCommissure.csv')
       
    
    scans = [PATH + x for x in os.listdir(PATH)]
    labels= [SEG_PATH + x for x in os.listdir(SEG_PATH)]
    
    
    OUTPUT_PATH = '/home/deeperthought/Projects/Multiaxial/predictions/'
    
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
       
        
    for scan in scans[:1]:
        print(scan)
        nii = nib.load(scan)
        print(nii.shape)
        
        subject = scan.split('/')[-1].split('_')[0].replace('.nii','')
        
        if subject.startswith('r'):
            subject = subject[1:]
        
        segmentation = [x for x in labels if subject in x][0]
        nii_seg = nib.load(segmentation)
        
        if np.any(anterior_commissure_df['MRI'].str.contains(subject)):
            print(f'found pre stored anterior commissure for subj: {subject}')
            prestored_anterior_commissure = anterior_commissure_df.loc[anterior_commissure_df['MRI'].str.contains(subject)][['x','y','z']].values[0]
        
        nii_out, nii_seg_out, coords, anterior_commissure = preprocess_head_MRI(nii, nii_seg, anterior_commissure=prestored_anterior_commissure)  
    
        img = nii_out.get_fdata()
        p95 = np.percentile(img, 95)
        img = img/p95
        
        coords = coords/256.
        
        yhat = model_sagittal.predict([img, coords])
        