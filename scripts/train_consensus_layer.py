#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 13:05:28 2024

@author: deeperthought
"""


#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 10:56:33 2022

Get ROIs from malignants in the training set, and compare with ground truth segmented locations (if available)

Get some metric for detection. 

@author: deeperthought
"""




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

import os
os.chdir('/home/deeperthought/Projects/Multiaxial/scripts/')

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from tensorflow.keras.optimizers import Adam

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D,Conv3D, MaxPooling2D, UpSampling2D, Activation, BatchNormalization, Conv2DTranspose, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

# from utils import dice_loss, dice_coef, Generalised_dice_coef_multilabel7,dice_coef_multilabel_bin0,dice_coef_multilabel_bin1, dice_coef_multilabel_bin2,dice_coef_multilabel_bin3, dice_coef_multilabel_bin4,dice_coef_multilabel_bin5,dice_coef_multilabel_bin6
from preprocessing_lib import preprocess_head_MRI

from tensorflow.keras import backend as K

#%% 3D



def dice_coef(y_true, y_pred):
    smooth = 1e-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f**2) + tf.reduce_sum(y_pred_f**2) + smooth)


def Generalised_dice_coef_multilabel7(y_true, y_pred, numLabels=7):
    """This is the loss function to MINIMIZE. A perfect overlap returns 0. Total disagreement returns numeLabels"""
    dice=0
    for index in range(numLabels):
        dice -= dice_coef(y_true[:,:,:,:,index], y_pred[:,:,:,:,index])
        
    return numLabels + dice


def dice_coef_multilabel_bin0(y_true, y_pred):
  numerator = 2 * tf.math.reduce_sum(y_true[:,:,:,:,0] * y_pred[:,:,:,:,0])
  denominator = tf.math.reduce_sum(y_true[:,:,:,:,0] + y_pred[:,:,:,:,0])
  return numerator / denominator

def dice_coef_multilabel_bin1(y_true, y_pred):
  numerator = 2 * tf.math.reduce_sum(y_true[:,:,:,:,1] * y_pred[:,:,:,:,1])
  denominator = tf.math.reduce_sum(y_true[:,:,:,:,1] + y_pred[:,:,:,:,1])
  return numerator / denominator

def dice_coef_multilabel_bin2(y_true, y_pred):
    
  numerator = 2 * tf.math.reduce_sum(y_true[:,:,:,:,2] * y_pred[:,:,:,:,2])
  denominator = tf.math.reduce_sum(y_true[:,:,:,:,2] + y_pred[:,:,:,:,2])
  return numerator / denominator

def dice_coef_multilabel_bin3(y_true, y_pred):
  numerator = 2 * tf.math.reduce_sum(y_true[:,:,:,:,3] * y_pred[:,:,:,:,3])
  denominator = tf.math.reduce_sum(y_true[:,:,:,:,3] + y_pred[:,:,:,:,3])
  return numerator / denominator

def dice_coef_multilabel_bin4(y_true, y_pred):
  numerator = 2 * tf.math.reduce_sum(y_true[:,:,:,:,4] * y_pred[:,:,:,:,4])
  denominator = tf.math.reduce_sum(y_true[:,:,:,:,4] + y_pred[:,:,:,:,4])
  return numerator / denominator

def dice_coef_multilabel_bin5(y_true, y_pred):
  numerator = 2 * tf.math.reduce_sum(y_true[:,:,:,:,5] * y_pred[:,:,:,:,5])
  denominator = tf.math.reduce_sum(y_true[:,:,:,:,5] + y_pred[:,:,:,:,5])
  return numerator / denominator

def dice_coef_multilabel_bin6(y_true, y_pred):
  numerator = 2 * tf.math.reduce_sum(y_true[:,:,:,:,6] * y_pred[:,:,:,:,6])
  denominator = tf.math.reduce_sum(y_true[:,:,:,:,6] + y_pred[:,:,:,:,6])
  return numerator / denominator


############### MODEL #########################

def make_consensus_model(N_DIMS=8):
    input_layer = Input(shape=(None,None,None,22))
    conv_layer = Conv3D(filters = N_DIMS, kernel_size=(3,3,3), activation='relu', padding='same')(input_layer)
    conv_layer = Conv3D(filters = 7, kernel_size=(1,1,1), activation='softmax', padding='same')(conv_layer)
    
    model = Model(inputs=input_layer, outputs=conv_layer)
    model.summary()
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=tf.keras.losses.categorical_crossentropy)
    
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=Generalised_dice_coef_multilabel7)
    
    model.compile(loss=Generalised_dice_coef_multilabel7, 
              optimizer=Adam(lr=1e-4), 
              metrics=[dice_coef_multilabel_bin0, 
                       dice_coef_multilabel_bin1, 
                       dice_coef_multilabel_bin2,
                       dice_coef_multilabel_bin3,
                       dice_coef_multilabel_bin4,
                       dice_coef_multilabel_bin5,
                       dice_coef_multilabel_bin6])
    
    return model

#%%

def split_image_into_blocks(image, block_size):
  """Splits a volumetric image into smaller blocks.

  Args:
    image: A 3D NumPy array representing the volumetric image.
    block_size: A tuple specifying the size of each block (x, y, z).

  Returns:
    A list of 3D NumPy arrays, each representing a block of the original image.
  """

  x_blocks = image.shape[0] // block_size[0]
  y_blocks = image.shape[1] // block_size[1]
  z_blocks = image.shape[2] // block_size[2]

  blocks = []
  for x in range(x_blocks):
    for y in range(y_blocks):
      for z in range(z_blocks):
        start_x = x * block_size[0]
        end_x = start_x + block_size[0]
        start_y = y * block_size[1]
        end_y = start_y + block_size[1]
        start_z = z * block_size[2]
        end_z = start_z + block_size[2]
        block = image[start_x:end_x, start_y:end_y, start_z:end_z]
        blocks.append(block)

  return np.array(blocks)


def load_scans(subjects, segmentations_path, dtype='float16', normalize=False):
    if type(subjects) != list:
        subjects= [subjects]
    segmentations = []
    scans = [x for x in os.listdir(segmentations_path) if x.split('_')[0].split('.')[0] in subjects]
    if len(scans)!=len(subjects): print('Warning: Not all subjects found!')
    for scan in scans:
        print(f'loading {scan}')
        img = nib.load(segmentations_path + scan).get_fdata()
        if normalize:
            img = img/np.percentile(img, 95)
        segmentations.append(img.astype(dtype))
    if len(subjects) == 1:
        return np.array(np.squeeze(segmentations))
    elif len(subjects) > 1:
        return np.concatenate(segmentations)



def prepare_data_for_model(X_MRI, X_sagittal, X_coronal,X_axial, Y):
    X_MRI = X_MRI/np.percentile(X_MRI, 95)
    X_sagittal = tf.keras.utils.to_categorical(np.array(X_sagittal), num_classes=7) 
    X_coronal = tf.keras.utils.to_categorical(np.array(X_coronal), num_classes=7) 
    X_axial = tf.keras.utils.to_categorical(np.array(X_axial), num_classes=7) 
    
    X = np.concatenate([np.expand_dims(X_MRI,-1), X_sagittal,X_coronal,X_axial ],-1)
    
    y = tf.keras.utils.to_categorical(np.array(Y), num_classes=7) 
    
    return X.astype('float16'),y.astype('float16')

def dice_coefficient(y_true, y_pred, num_classes=7):

  dice = 0.0
  for i in range(num_classes):
    y_true_class = np.where(y_true == i, 1, 0).astype(np.float32)
    y_pred_class = np.where(y_pred == i, 1, 0).astype(np.float32)

    intersection = np.sum(y_true_class * y_pred_class)
    union = np.sum(y_true_class + y_pred_class)
    dice_class = 2 * intersection / (union + np.finfo(float).eps)
    #print(f'class:{i} = {dice_class}')
    dice +=  dice_class

  return dice / num_classes




def load_data_for_training(train_subjects, sagittal_segmentations_path,coronal_segmentations_path, axial_segmentations_path, MRI_PATH, GT_PATH, BLOCK_SIZE):

    #-------------- train data 
    train_MRI = load_scans(train_subjects, MRI_PATH, normalize=True)
    train_GT = load_scans(train_subjects, GT_PATH)   
    train_sagittal = load_scans(train_subjects, sagittal_segmentations_path)
    train_coronal = load_scans(train_subjects, coronal_segmentations_path)
    train_axial = load_scans(train_subjects, axial_segmentations_path)
    
    mri_blocks = split_image_into_blocks(train_MRI, block_size=BLOCK_SIZE)
    gt_blocks = split_image_into_blocks(train_GT, block_size=BLOCK_SIZE)
    sagittal_blocks = split_image_into_blocks(train_sagittal, block_size=BLOCK_SIZE)
    coronal_blocks = split_image_into_blocks(train_coronal, block_size=BLOCK_SIZE)
    axial_blocks = split_image_into_blocks(train_axial, block_size=BLOCK_SIZE)
    
    # Remove blocks where > 95% is empty...
    cubes_gt_background_percent_mask = [np.sum(x == 0)/np.prod(BLOCK_SIZE) < 0.95 for x in gt_blocks]

    mri_blocks = mri_blocks[cubes_gt_background_percent_mask]
    gt_blocks = gt_blocks[cubes_gt_background_percent_mask]
    sagittal_blocks = sagittal_blocks[cubes_gt_background_percent_mask]
    coronal_blocks = coronal_blocks[cubes_gt_background_percent_mask]
    axial_blocks = axial_blocks[cubes_gt_background_percent_mask]
    
        
    sagittal_blocks = tf.keras.utils.to_categorical(sagittal_blocks, num_classes=7) 
    coronal_blocks = tf.keras.utils.to_categorical(coronal_blocks, num_classes=7) 
    axial_blocks = tf.keras.utils.to_categorical(axial_blocks, num_classes=7) 
    
    X = np.concatenate([np.expand_dims(mri_blocks,-1), sagittal_blocks,coronal_blocks,axial_blocks ],-1) 
    y = tf.keras.utils.to_categorical(np.array(gt_blocks), num_classes=7) 

  

    return X.astype('float16'),y.astype('float16')



############## VOTE CONSENSUS ######################################
def make_vote_consensus(subj_sagittal_seg, subj_coronal_seg, subj_axial_seg):
    print('Making vote consensus...')
    vote_vol = np.zeros(subj_sagittal_seg.shape, dtype=np.uint8)  # Initialize with zeros
    equals = np.logical_and((subj_sagittal_seg == subj_coronal_seg), 
                            (subj_axial_seg == subj_coronal_seg))
    
    
    # Assign values where models agree
    vote_vol[equals] = subj_sagittal_seg[equals]
    
    # Get vectors that need consensus
    sagittal_needs_consensus_vector = subj_sagittal_seg[~equals]
    axial_needs_consensus_vector = subj_axial_seg[~equals]
    coronal_needs_consensus_vector = subj_coronal_seg[~equals]
    
    # Stack vectors for voting
    needs_consensus_vector = np.stack([sagittal_needs_consensus_vector, 
                                        axial_needs_consensus_vector, 
                                        coronal_needs_consensus_vector], axis=0)
    
    # Get the mode of the consensus vectors
    import scipy
    vote_vector = scipy.stats.mode(needs_consensus_vector, axis=0)
    vote_vol[~equals] = vote_vector[0][0]

    return vote_vol
#%%

coronal_segmentations_path = "/media/HDD/MultiAxial/Fold1/coronal_predictions/"
sagittal_segmentations_path = "/media/HDD/MultiAxial/Fold1/sagittal_predictions/"
axial_segmentations_path = "/media/HDD/MultiAxial/Fold1/axial_predictions/"
GT_PATH = "/media/HDD/MultiAxial/Data/All_heads/GT/"
MRI_PATH = "/media/HDD/MultiAxial/Data/All_heads/MRI/"

subjects = list(set([x.split('_')[0] for x in os.listdir(coronal_segmentations_path)]))


train_subjects = subjects[5:]#[:5]
val_subjects = subjects[:5]#[5:]

X,y = load_data_for_training(train_subjects,  
                             sagittal_segmentations_path,coronal_segmentations_path, axial_segmentations_path, MRI_PATH, GT_PATH, 
                             BLOCK_SIZE = (64,64,64))

X_val, y_val = load_data_for_training(val_subjects,  
                             sagittal_segmentations_path,coronal_segmentations_path, axial_segmentations_path, MRI_PATH, GT_PATH, 
                             BLOCK_SIZE = (64,64,64))

X.shape
y.shape
X_val.shape
y_val.shape
X.dtype


#%%  ######################## MODEL TRAINING #######################3

model = make_consensus_model()


history = model.fit(X,y,epochs=500, batch_size=6, validation_data=(X_val,y_val), shuffle=True)


plt.figure(figsize=(15,5))
plt.subplot(131)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()

plt.subplot(132)
for k in ['dice_coef_multilabel_bin0', 'dice_coef_multilabel_bin1', 'dice_coef_multilabel_bin2', 'dice_coef_multilabel_bin3', 'dice_coef_multilabel_bin4', 'dice_coef_multilabel_bin5', 'dice_coef_multilabel_bin6']:
    plt.plot(history.history[k], label=k)
plt.legend(loc='lower right')

plt.subplot(133)
for k in 'val_dice_coef_multilabel_bin0', 'val_dice_coef_multilabel_bin1', 'val_dice_coef_multilabel_bin2', 'val_dice_coef_multilabel_bin3', 'val_dice_coef_multilabel_bin4', 'val_dice_coef_multilabel_bin5', 'val_dice_coef_multilabel_bin6':
    plt.plot(history.history[k], label=k)
plt.legend(loc='lower right')



#%%   ################# Full head prediction ##################3

val_results = {}

for subj in val_subjects:
    print(subj)
    subj_sagittal_seg = load_scans(subj, sagittal_segmentations_path)
    subj_coronal_seg = load_scans(subj, coronal_segmentations_path)
    subj_axial_seg = load_scans(subj, axial_segmentations_path)
    subj_MRI = load_scans(subj, MRI_PATH)
    subj_GT = load_scans(subj, GT_PATH)
           
    X,y = prepare_data_for_model(subj_MRI, subj_sagittal_seg, subj_coronal_seg,subj_axial_seg, subj_GT)
    
    print('getting model consensus')

    yhat = model.predict(np.expand_dims(X,0))
    pred = np.argmax(yhat[0],-1)

    vote_pred = make_vote_consensus(subj_sagittal_seg, subj_coronal_seg, subj_axial_seg)


    dice_vote = dice_coefficient(subj_GT, vote_pred) 
    dice_consensus = dice_coefficient(subj_GT, pred)
    dice_sagittal = dice_coefficient(subj_GT, subj_sagittal_seg)
    dice_coronal = dice_coefficient(subj_GT, subj_coronal_seg)
    dice_axial = dice_coefficient(subj_GT, subj_axial_seg)
    
    val_results[subj] = {'consensus':dice_consensus, 'vote':dice_vote,'sagittal':dice_sagittal, 'coronal':dice_coronal, 'axial':dice_axial}
        
    INDEX=50
    plt.figure(figsize=(10,10))
    plt.subplot(231); plt.imshow(np.rot90(subj_GT[INDEX]))
    plt.subplot(232); plt.imshow(np.rot90(pred[INDEX]))
    plt.subplot(233); plt.imshow(np.rot90(subj_sagittal_seg[INDEX]))
    plt.subplot(234); plt.imshow(np.rot90(subj_coronal_seg[INDEX]))
    plt.subplot(235); plt.imshow(np.rot90(subj_axial_seg[INDEX]))
    
    
    

    
consensus_results = []
vote_results = []
sagittal_results = []
coronal_results = []
axial_results = []

for subj in val_results.keys():
    consensus_results.append(val_results[subj]['consensus'])
    vote_results.append(val_results[subj]['vote'])
    sagittal_results.append(val_results[subj]['sagittal'])
    coronal_results.append(val_results[subj]['coronal'])
    axial_results.append(val_results[subj]['axial'])


plt.plot([sagittal_results,coronal_results, axial_results, vote_results, consensus_results], color='lightblue')

plt.plot(np.random.normal(loc=0, scale=0.1,size=len(consensus_results)), sagittal_results, '.')
plt.plot(np.random.normal(loc=1, scale=0.1,size=len(consensus_results)), coronal_results, '.')
plt.plot(np.random.normal(loc=2, scale=0.1,size=len(consensus_results)), axial_results, '.')
plt.plot(np.random.normal(loc=3, scale=0.1,size=len(consensus_results)), vote_results, '.')
plt.plot(np.random.normal(loc=4, scale=0.1,size=len(consensus_results)), consensus_results, '.')
plt.plot([np.median(sagittal_results), np.median(coronal_results), np.median(axial_results), np.median(vote_results), np.median(consensus_results)], color='k')


plt.xticks([0,1,2,3,4],['sagittal','coronal','axial','vote','consensus_3DCNN'])
plt.ylabel('Dice')
# plt.ylim([0.5,1])
plt.grid()



