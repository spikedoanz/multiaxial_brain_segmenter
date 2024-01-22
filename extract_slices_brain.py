#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 12:25:36 2023

@author: deeperthought
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt

from skimage.transform import resize

PATH = '/home/deeperthought/kirby/home/AphasicStrokeTrial'

SEG = pd.read_csv('/home/deeperthought/kirby/home/AphasicStrokeTrial/all_LABEL.txt', header=None)[0].values
MRI = pd.read_csv('/home/deeperthought/kirby/home/AphasicStrokeTrial/all_MRI.txt', header=None)[0].values

affines = []
mris_list = []
seg_list = []

labelsunique_list = []

TARGET_WIDTH = 256
i = 0
for i in range(len(MRI)):
    print(i)
#    mri_path = PATH + MRI[i].split('AphasicStrokeTrial')[-1]
#    mri = nib.load(mri_path)
#
#    affines.append(mri.affine)
#
#    mri = mri.get_data()
#
#    mri = resize(mri, output_shape=((mri.shape[0], 256, 256)),  anti_aliasing=True, preserve_range=True)    
#
#    mri = mri / np.percentile(mri, 95)


    seg_path = PATH + SEG[i].split('AphasicStrokeTrial')[-1]
    seg = nib.load(seg_path).get_data()     
    
    seg = seg - 1   
    
    seg = resize(seg, output_shape=(seg.shape[0], 256, 256), order=0, anti_aliasing=False, preserve_range=True)    

    
    padding_width = TARGET_WIDTH - mri.shape[0]
    
    mri_padded = np.pad(mri, ((padding_width/2,padding_width/2 + padding_width%2),(0,0),(0,0)), 'constant')
    seg_padded = np.pad(seg, ((padding_width/2,padding_width/2 + padding_width%2),(0,0),(0,0)), 'minimum')

    mris_list.append(mri_padded)
    seg_list.append(seg_padded)

len(mris_list)
len(seg_list)

shapes = [x.shape for x in mris_list]

all_slices_x = np.sum([x.shape[0] for x in mris_list])
all_slices_y = np.sum([x.shape[1] for x in mris_list])
all_slices_z = np.sum([x.shape[2] for x in mris_list])

DATA_X = np.zeros((all_slices_x, 256, 256))
LABELS_X = np.zeros((all_slices_x, 256, 256))

DATA_Y = np.zeros((256, all_slices_y, 256))
LABELS_Y = np.zeros((256, all_slices_y, 256))

DATA_Z = np.zeros((256, 256, all_slices_z))
LABELS_Z = np.zeros((256, 256, all_slices_z))

n = 0
for i in range(len(MRI)):
    
    width = mris_list[i].shape[0]
    DATA_X[n:n+width] = mris_list[i]
    LABELS_X[n:n+width] = seg_list[i]
    
    DATA_Y[:,n:n+width] = mris_list[i]
    LABELS_Y[:,n:n+width] = seg_list[i]
    
    DATA_Z[:,:,n:n+width] = mris_list[i]
    LABELS_Z[:,:,n:n+width] = seg_list[i]
    
    n = n+width

np.unique(LABELS_X)

LABELS_X = LABELS_X - 1
LABELS_Y = LABELS_Y - 1
LABELS_Z = LABELS_Z - 1


plt.figure(figsize=(15,15))
plt.subplot(3,2,1)
plt.imshow(DATA_X[6595])
plt.subplot(3,2,2)
plt.imshow(LABELS_X[6595])

plt.subplot(3,2,3)
plt.imshow(DATA_Y[:,6595])
plt.subplot(3,2,4)
plt.imshow(LABELS_Y[:,6595])

plt.subplot(3,2,5)
plt.imshow(DATA_Z[:,:,6595])
plt.subplot(3,2,6)
plt.imshow(LABELS_Z[:,:,6595])

OUT_DATA = '/home/deeperthought/kirby/home/AphasicStrokeTrial/ALL_DATA/slices/'
OUT_LABELS = '/home/deeperthought/kirby/home/AphasicStrokeTrial/ALL_DATA/labels/'

os.mkdir(OUT_DATA + 'sagittal')
os.mkdir(OUT_LABELS + 'sagittal')
os.mkdir(OUT_DATA + 'axial')
os.mkdir(OUT_LABELS + 'axial')
os.mkdir(OUT_DATA + 'coronal')
os.mkdir(OUT_LABELS + 'coronal')

for i in range(DATA_X.shape[0]):
    np.save(OUT_DATA + '/sagittal/slice_{}.npy'.format(i), DATA_X[i])
    np.save(OUT_LABELS + '/sagittal/slice_{}.npy'.format(i), LABELS_X[i])
    
    np.save(OUT_DATA + '/axial/slice_{}.npy'.format(i), DATA_Z[:,:,i])
    np.save(OUT_LABELS + '/axial/slice_{}.npy'.format(i), LABELS_Z[:,:,i])    
    
    np.save(OUT_DATA + '/coronal/slice_{}.npy'.format(i), DATA_Y[:,i])
    np.save(OUT_LABELS + '/coronal/slice_{}.npy'.format(i), LABELS_Y[:,i])
#    
#np.unique(LABELS_X)
#
#np.save('/home/deeperthought/kirby/home/AphasicStrokeTrial/ALL_DATA/mri.npy', DATA)
#np.save('/home/deeperthought/kirby/home/AphasicStrokeTrial/ALL_DATA/segmentations.npy', LABELS)




y = np.load('/home/deeperthought/kirby/home/AphasicStrokeTrial/ALL_DATA/labels/sagittal/slice_1097.npy', allow_pickle=True)

np.unique(y)

plt.imshow(y)
