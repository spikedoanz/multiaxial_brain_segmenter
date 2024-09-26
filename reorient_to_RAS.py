#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14:34:36 2024

@author: deeperthought
"""

import nibabel as nib
import numpy as np
import os
from skimage.transform import resize
import pandas as pd
import matplotlib.pyplot as plt

from nibabel.orientations import axcodes2ornt
from nibabel.orientations import ornt_transform


def reorient( nii, orientation) -> nib.Nifti1Image:
    """Reorients a nifti image to specified orientation. Orientation string or tuple
    must consist of "R" or "L", "A" or "P", and "I" or "S" in any order."""
    orig_ornt = nib.io_orientation(nii.affine)
    targ_ornt = axcodes2ornt(orientation)
    transform = ornt_transform(orig_ornt, targ_ornt)
    reoriented_nii = nii.as_reoriented(transform)
    return reoriented_nii



#%% Pre-Process heads, without intensity normalization


def create_coordinate_matrix(shape, anterior_commissure):
    x, y, z = shape
    meshgrid = np.meshgrid(np.linspace(0, x - 1, x), np.linspace(0, y - 1, y), np.linspace(0, z - 1, z), indexing='ij')
    coordinates = np.stack(meshgrid, axis=-1) - np.array(anterior_commissure)
    matrix_with_ones = np.concatenate([coordinates, np.ones((coordinates.shape[0], coordinates.shape[1], coordinates.shape[2], 1))], axis=-1)

    return matrix_with_ones

def make_spatial_coordinates(nii, anterior_commissure):
    matrix = create_coordinate_matrix(nii.shape, anterior_commissure)
    
    matrix_with_ones = np.concatenate([matrix, np.ones((matrix.shape[0], matrix.shape[1], matrix.shape[2], 1))], axis=-1)
    
    # orientation = nib.aff2axcodes(nii.affine)
    
    # # Here I am resampling the coordinate matrix... 
    result_coords = np.matmul(matrix_with_ones, np.linalg.inv(nii.affine))
    
    return result_coords


# nii = nib.load('/home/deeperthought/Projects/MultiPriors/Adam_Buchwald/P906/NIFTY/P906_T1_original.nii')
# anterior_commissure = [95, 152, 153]

def preprocess_head_MRI(nii, nii_seg=0, anterior_commissure=0):
    'Anterior commissure found using MRIcron which already rotates to RAS on display' 
    
    assert nii.shape == nii_seg.shape
    
    if len(anterior_commissure) == 1:
        anterior_commissure = nii.shape[0]//2, nii.shape[1]//2, nii.shape[2]//2
    
    orientation = nib.aff2axcodes(nii.affine)
    
    if ''.join(orientation) != 'RAS':
    
        print(f'Image orientation : {orientation}. Changing to RAS..')
        
        nii = reorient(nii, "RAS")
        nii_seg = reorient(nii_seg, "RAS")
            
    ############### ISOTROPIC #######################
    
    res = nii.header['pixdim'][1:4]
    
    img = nii.get_fdata()
    img_seg = nii_seg.get_fdata()
    
    new_shape = np.array(np.array(nii.shape)*res, dtype='int')
    
    if np.any(np.array(nii.shape) != new_shape):
        img = resize(img, new_shape, anti_aliasing=True, preserve_range=True)
        img_seg = resize(img_seg, new_shape, order=0, anti_aliasing=True, preserve_range=True)
       
    nii.affine[0][0] = 1.
    nii.affine[1][1] = 1.
    nii.affine[2][2] = 1.
    
    nii.header['pixdim'][1:4] = np.diag(nii.affine)[0:3]
    
    ############### Crop/Pad to make shape 256,256,256 ###############
    
    d1, d2, d3 = new_shape
    
    if d1 < 256:
        pad1 = 256-d1
        img = np.pad(img, ((pad1//2, pad1//2+pad1%2),(0,0),(0,0)))
        img_seg = np.pad(img_seg, ((pad1//2, pad1//2+pad1%2),(0,0),(0,0)))

        anterior_commissure[0] += pad1//2
        
    
    if d2 > 256: 
        crop2 = d2-256
        img = img[:,crop2//2:-(crop2//2+crop2%2)]
        img_seg = img_seg[:,crop2//2:-(crop2//2+crop2%2)]
        anterior_commissure[1] -= crop2//2
            
    elif d2 < 256:
        pad2 = 256-d2
        img = np.pad(img, ((0,0),(pad2//2, pad2//2+pad2%2),(0,0)))
        img_seg = np.pad(img_seg, ((0,0),(pad2//2, pad2//2+pad2%2),(0,0)))
        anterior_commissure[1] += pad2//2
        
    if d3 > 256: 

        #--- head start
        proj = np.max(img,(0,1))
        proj[proj < np.percentile(proj, 50) ] = 0
        proj[proj > 0] = 1
        
        end = np.max(np.argwhere(proj == 1))
        
        end = np.min([end + 20, d3]) # leave some space above the head
        start = end-256

        if start < 0:
            crop3 = d3 - 256
           
            img = img[:,:,crop3:]
            img_seg = img_seg[:,:,crop3:]
            anterior_commissure[2] -= crop3
         
        else:
            img = img[:,:,start:end]
            img_seg = img_seg[:,:,start:end]
            anterior_commissure[2] -= start

    elif d3 < 256:
        pad3 = 256-d3
        img = np.pad(img, ((0,0),(0,0),(pad3//2, pad3//2+pad3%2)))
        img_seg = np.pad(img_seg, ((0,0),(0,0),(pad3//2, pad3//2+pad3%2)))
        anterior_commissure[2] += pad3//2

    anterior_commissure = np.array(anterior_commissure, dtype='int')

    coords = create_coordinate_matrix(img.shape, anterior_commissure)        
    
    
    
    # coords = np.matmul(coords, np.linalg.inv(nii.affine))


    nii_out = nib.Nifti1Image(img, nii.affine)
    nii_seg_out = nib.Nifti1Image(img_seg, nii.affine)
    
    
    return nii_out, nii_seg_out, coords, anterior_commissure


#%%

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

OUTPUT_PATH = '/media/SD/New_slices_coordinates/'

for scan in scans:
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
    img_seg = nii_seg_out.get_fdata()
    
    p95 = np.percentile(img, 95)

    img = img/p95

    # Store slices:
    for i in range(256):
        
        np.save(OUTPUT_PATH + '/sagittal/MRI/' + subject + f'_slice{i}.npy', img[i])
        np.save(OUTPUT_PATH + '/sagittal/GT/' + subject + f'_slice{i}.npy', img_seg[i])
        np.save(OUTPUT_PATH + '/sagittal/coords/' + subject + f'_slice{i}.npy', coords[i,:,:,:3])
    
    
        
        np.save(OUTPUT_PATH + '/coronal/MRI/' + subject + f'_slice{i}.npy', img[:,i])
        np.save(OUTPUT_PATH + '/coronal/GT/' + subject + f'_slice{i}.npy', img_seg[:,i])
        np.save(OUTPUT_PATH + '/coronal/coords/' + subject + f'_slice{i}.npy', coords[:,i,:,:3])
    
        
        np.save(OUTPUT_PATH + '/axial/MRI/' + subject + f'_slice{i}.npy', img[:,:,i])
        np.save(OUTPUT_PATH + '/axial/GT/' + subject + f'_slice{i}.npy', img_seg[:,:,i])
        np.save(OUTPUT_PATH + '/axial/coords/' + subject + f'_slice{i}.npy', coords[:,:,i,:3])
    
 
    
    # print(nii_out.shape)
    
    # assert nii_out.shape == (256,256,256)
    # assert nii_seg_out.shape == (256,256,256)
    
    # center = np.argwhere(np.all(coords[:,:,:,:3] == 0, axis=-1))[0]
    
    # assert np.all(np.array(anterior_commissure) == center)
    
    # img[anterior_commissure[0]-5:anterior_commissure[0]+5,
    #     anterior_commissure[1]-5:anterior_commissure[1]+5,
    #     anterior_commissure[2]-5:anterior_commissure[2]+5] += 800

    # img_seg[anterior_commissure[0]-5:anterior_commissure[0]+5,
    #     anterior_commissure[1]-5:anterior_commissure[1]+5,
    #     anterior_commissure[2]-5:anterior_commissure[2]+5] += 10
    
    
    
    # Visual check anterior commissure location after preprocessing
    
    # plt.subplot(131)
    # plt.imshow(np.rot90(img[anterior_commissure[0]])); plt.xticks([]); plt.yticks([])
    # plt.subplot(132); plt.title(subject)    
    # plt.imshow(np.rot90(img[:,anterior_commissure[1]])); plt.xticks([]); plt.yticks([])
    # plt.subplot(133)
    # plt.imshow(np.rot90(img[:,:,anterior_commissure[2]])); plt.xticks([]); plt.yticks([])
    # plt.tight_layout()
    # plt.savefig('/media/HDD/MultiAxial/Data/AphasicStroke/anterior_commissure_visualCheck/' + subject + '.png', dpi=300)
    
    
    
    
    
    # Visual Check coordinates
    
    # coords = np.array(coords*0.1, 'int8')
    
    # plt.figure(figsize=(20,24))
    
    # plt.subplot(431)
    # plt.imshow(np.rot90(img[anterior_commissure[0]])); plt.xticks([]); plt.yticks([])

    # plt.subplot(434); plt.ylabel('X')
    # plt.imshow(np.rot90(coords[anterior_commissure[0], :, :, 0])); plt.xticks([]); plt.yticks([]); plt.colorbar()
    # plt.subplot(437); plt.ylabel('Y')
    # plt.imshow(np.rot90(coords[anterior_commissure[0], :, :, 1])); plt.xticks([]); plt.yticks([]); plt.colorbar()
    # plt.subplot(4,3,10); plt.ylabel('Z')
    # plt.imshow(np.rot90(coords[anterior_commissure[0], :, :, 2])); plt.xticks([]); plt.yticks([]); plt.colorbar()
    
    
    
    # plt.subplot(432); plt.title(subject)    
    # plt.imshow(np.rot90(img[:,anterior_commissure[1]])); plt.xticks([]); plt.yticks([])

    # plt.subplot(435); 
    # plt.imshow(np.rot90(coords[:,anterior_commissure[1], :, 0])); plt.xticks([]); plt.yticks([]); plt.colorbar()
    # plt.subplot(438)
    # plt.imshow(np.rot90(coords[:,anterior_commissure[1], :, 1])); plt.xticks([]); plt.yticks([]); plt.colorbar()
    # plt.subplot(4,3,11)
    # plt.imshow(np.rot90(coords[:,anterior_commissure[1], :, 2])); plt.xticks([]); plt.yticks([]); plt.colorbar()
    
    
    # plt.subplot(433)
    # plt.imshow(np.rot90(img[:,:,anterior_commissure[2]])); plt.xticks([]); plt.yticks([])

    # plt.subplot(436)
    # plt.imshow(np.rot90(coords[:,:,anterior_commissure[2],  0])); plt.xticks([]); plt.yticks([]); plt.colorbar()
    # plt.subplot(439)
    # plt.imshow(np.rot90(coords[:,:,anterior_commissure[2],  1])); plt.xticks([]); plt.yticks([]); plt.colorbar()
    # plt.subplot(4,3,12)
    # plt.imshow(np.rot90(coords[:,:,anterior_commissure[2],  2])); plt.xticks([]); plt.yticks([]); plt.colorbar()
    
    
    
    
    # Visual check segmentation

    # plt.subplot(231)
    # plt.imshow(np.rot90(img[anterior_commissure[0]])); plt.xticks([]); plt.yticks([])
    # plt.subplot(232); plt.title(subject)    
    # plt.imshow(np.rot90(img[:,anterior_commissure[1]])); plt.xticks([]); plt.yticks([])
    # plt.subplot(233)
    # plt.imshow(np.rot90(img[:,:,anterior_commissure[2]])); plt.xticks([]); plt.yticks([])
    
    # plt.subplot(234)
    # plt.imshow(np.rot90(img_seg[anterior_commissure[0]])); plt.xticks([]); plt.yticks([])
    # plt.subplot(235); plt.title(subject)    
    # plt.imshow(np.rot90(img_seg[:,anterior_commissure[1]])); plt.xticks([]); plt.yticks([])
    # plt.subplot(236)
    # plt.imshow(np.rot90(img_seg[:,:,anterior_commissure[2]])); plt.xticks([]); plt.yticks([])
    

    # plt.tight_layout()
    # plt.savefig('/media/HDD/MultiAxial/Data/AdamBuchwald/Anterior_Commissure_Check/' + subject + '.png', dpi=300)
    # plt.close()