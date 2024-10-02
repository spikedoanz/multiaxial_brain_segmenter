#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14:34:36 2024

@author: deeperthought
"""


import os
import pandas as pd

import nibabel as nib
import numpy as np

if __name__ == '__main__':
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    from headMRI_preProcessing_library import preprocess_head_MRI
    
    PATH = '/media/HDD/MultiAxial/Data/Processed_New_MCS/MRI/'
    SEG_PATH = '/media/HDD/MultiAxial/Data/Processed_New_MCS/GT/'
    anterior_commissure_df = pd.read_csv('/media/HDD/MultiAxial/Data/Processed_New_MCS_AnteriorCommissure.csv')
    
    # PATH = '/media/HDD/MultiAxial/Data/NormalHeads/MRI/'
    # anterior_commissure_df = pd.read_csv('/media/HDD/MultiAxial/Data/Normal_Heads_Anterior_Commissure.csv')
    
    # PATH = '/media/HDD/MultiAxial/Data/AphasicStroke/MRI/'
    # SEG_PATH = '/media/HDD/MultiAxial/Data/AphasicStroke/GT/'
    # anterior_commissure_df = pd.read_csv('/media/HDD/MultiAxial/Data/AphasicStroke_AnteriorCommissure.csv')
    
    
    # PATH = '/media/HDD/MultiAxial/Data/AdamBuchwald/Adams_Manual_Fixed/MRI/'
    # SEG_PATH = '/media/HDD/MultiAxial/Data/AdamBuchwald/Adams_Manual_Fixed/GT/'
    # anterior_commissure_df = pd.read_csv('/media/HDD/MultiAxial/Data/AdamBuchwald_Anterior_Commissure.csv')
    
    
    scans = [PATH + x for x in os.listdir(PATH)]
    labels= [SEG_PATH + x for x in os.listdir(SEG_PATH)]
    
    # OUTPUT_PATH = '/media/HDD/MultiAxial/Data/Slices/'
    
    OUTPUT_PATH = '/home/deeperthought/Projects/Multiaxial/Data/preprocessed_heads/'
    
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
        
        
        nii_out = nib.Nifti1Image(img, nii_out.affine)
        nii_coords_out = nib.Nifti1Image(coords, nii_out.affine)
        
        nib.save(nii_out, OUTPUT_PATH + subject + '.nii')
        nib.save(nii_seg_out, OUTPUT_PATH + subject + '_labels.nii')
        nib.save(nii_coords_out, OUTPUT_PATH + subject + '_coordinates.nii')
    