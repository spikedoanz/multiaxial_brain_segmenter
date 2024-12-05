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




GPU = 3

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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import scipy.stats

np.random.seed(42)
tf.random.set_seed(42)

#%% FUNCTIONS



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

def make_consensus_model(N_DIMS=8, LR=1e-4, layers=1, L2=0, last_kernel_size=(1,1,1), kernel_initializer=None, bias_initializer=None):
    input_layer = Input(shape=(None,None,None,22))
    if layers==1:
        if kernel_initializer is not None:
            final_layer = Conv3D(filters = 7, kernel_size=last_kernel_size,kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, activation='softmax', padding='same', kernel_regularizer=regularizers.l2(L2))(input_layer)
        else:
            final_layer = Conv3D(filters = 7, kernel_size=last_kernel_size, activation='softmax', padding='same', kernel_regularizer=regularizers.l2(L2))(input_layer)
    elif layers==2:
        conv_layer = Conv3D(filters = N_DIMS, kernel_size=(3,3,3), activation='sigmoid', padding='same', kernel_regularizer=regularizers.l2(L2))(input_layer)
        final_layer = Conv3D(filters = 7, kernel_size=last_kernel_size, activation='softmax', padding='same', kernel_regularizer=regularizers.l2(L2))(conv_layer)
    
    model = Model(inputs=input_layer, outputs=final_layer)
    model.summary()
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=tf.keras.losses.categorical_crossentropy)
    
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=Generalised_dice_coef_multilabel7)
    
    model.compile(loss=Generalised_dice_coef_multilabel7, 
              optimizer=Adam(lr=LR), 
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



def prepare_data_for_model(X_MRI, X_sagittal, X_coronal,X_axial, Y, use_probs=False):
    X_MRI = X_MRI/np.percentile(X_MRI, 95)
    if not use_probs:
        
        X_sagittal = np.argmax(X_sagittal, -1)
        X_coronal = np.argmax(X_coronal, -1)
        X_axial = np.argmax(X_axial, -1)
        
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




def load_data_for_training(train_subjects, sagittal_segmentations_path,coronal_segmentations_path, axial_segmentations_path, MRI_PATH, GT_PATH, BLOCK_SIZE, USE_PROBS):

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
    
    if not USE_PROBS:
        
        sagittal_blocks = np.argmax(sagittal_blocks,-1)
        coronal_blocks = np.argmax(coronal_blocks,-1)
        axial_blocks = np.argmax(axial_blocks,-1)
        
        
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


def make_vote_consensus_probs(subj_sagittal_seg, subj_coronal_seg, subj_axial_seg):
    print('Making vote consensus...')


    # Use probabilities for discerning clashes
    subj_sagittal_seg_bin = np.argmax(subj_sagittal_seg, -1)
    subj_coronal_seg_bin = np.argmax(subj_coronal_seg, -1)
    subj_axial_seg_bin = np.argmax(subj_axial_seg, -1)    

    vote_vol = np.zeros(subj_sagittal_seg_bin.shape, dtype=np.uint8)  # Initialize with zeros
    equals = np.logical_and((subj_sagittal_seg_bin == subj_coronal_seg_bin), 
                            (subj_axial_seg_bin == subj_coronal_seg_bin))
    
    

    # Assign values where models agree
    vote_vol[equals] = subj_sagittal_seg_bin[equals]


    
    # Get vectors that need consensus
    bin_sagittal_needs_consensus_vector = subj_sagittal_seg_bin[~equals]
    bin_axial_needs_consensus_vector = subj_axial_seg_bin[~equals]
    bin_coronal_needs_consensus_vector = subj_coronal_seg_bin[~equals]


    sagittal_needs_consensus_vector = subj_sagittal_seg[~equals]
    axial_needs_consensus_vector = subj_axial_seg[~equals]
    coronal_needs_consensus_vector = subj_coronal_seg[~equals]
    
    
    vote_vector = []
    for INDEX in range(len(sagittal_needs_consensus_vector)):
        sag_pred = bin_sagittal_needs_consensus_vector[INDEX]
        axial_pred = bin_axial_needs_consensus_vector[INDEX]
        coronal_pred = bin_coronal_needs_consensus_vector[INDEX]
        
        list_of_elements = [sag_pred,coronal_pred, axial_pred]
        if len(set(list_of_elements)) == 3:
            sag_v = sagittal_needs_consensus_vector[INDEX]
            ax_v = axial_needs_consensus_vector[INDEX]
            co_v = coronal_needs_consensus_vector[INDEX]
            
            
            most_confident_model = np.argmax([np.max(sag_v), np.max(ax_v),  np.max(co_v)])
            winning_prediction = list_of_elements[most_confident_model]

        else:
            winning_prediction = scipy.stats.mode(list_of_elements, axis=0)[0][0]

        vote_vector.append(winning_prediction)
        
    vote_vol[~equals] = vote_vector
    
    return vote_vol    

    # disagree = []
    # for INDEX in range(len(sagittal_needs_consensus_vector)):
    #     sag_pred = bin_sagittal_needs_consensus_vector[INDEX]
    #     axial_pred = bin_axial_needs_consensus_vector[INDEX]
    #     coronal_pred = bin_coronal_needs_consensus_vector[INDEX]
        
    #     list_of_elements = [sag_pred, axial_pred, coronal_pred]
    #     if len(set(list_of_elements)) == 3:
    #         disagree.append(INDEX)
    #     print(f'Disagreement in {len(disagree)} voxels. ({round(len(disagree)*100./INDEX,2)}%)')
        
 
    # INDEX = disagree[2]
    # plt.plot(sagittal_needs_consensus_vector[INDEX], label=f'sagittal: {sag_pred}')
    # plt.plot(axial_needs_consensus_vector[INDEX], label=f'axial: {axial_pred}')
    # plt.plot(coronal_needs_consensus_vector[INDEX], label=f'coronal: {coronal_pred}')
    # plt.legend()
      





#%% Pipeline

#-----------  Make Model

def custom_kernel_initializer(shape=(1,1,1,22,7), dtype='float32'):
  return tf.random.uniform(shape, minval=0.99, maxval=1.01, dtype=dtype)

def custom_bias_initializer(shape=(7,), dtype='float32'):
  return tf.random.uniform(shape, minval=-0.01, maxval=0.01, dtype=dtype)


# model = make_consensus_model(N_DIMS=8, LR=5e-4, layers=1, L2=0, last_kernel_size=(1,1,1), 
#                              kernel_initializer=custom_kernel_initializer,
#                              bias_initializer=custom_bias_initializer)

model = make_consensus_model(N_DIMS=8, LR=5e-4, layers=1, L2=0, last_kernel_size=(1,1,1))

# w = model.layers[1].get_weights()
# w[0]

#-----------  Prepare data

batch_size = 16#48  

BLOCK_SIZE = (64,64,64)


use_tf_Dataset = False
use_probabilities = True

FOLD = '2'
coronal_segmentations_path = f"/media/HDD/MultiAxial/Data/folds 2-7 val/fold{FOLD}/coronal/"
sagittal_segmentations_path = f"/media/HDD/MultiAxial/Data/folds 2-7 val/fold{FOLD}/sagittal/"
axial_segmentations_path = f"/media/HDD/MultiAxial/Data/folds 2-7 val/fold{FOLD}/axial/"


GT_PATH = "/media/HDD/MultiAxial/Data/All_heads/GT/"
MRI_PATH = "/media/HDD/MultiAxial/Data/All_heads/MRI/"

subjects = list(set([x.split('_')[0] for x in os.listdir(coronal_segmentations_path)]))

train_subjects = ['NM012',  'PA020','RM165', 'CM013', 'TO016','DS007','FJ003']#subjects[:5]#[:5]
val_subjects = ['P907', 'P908','P904']# subjects[5:]#[5:]

print(train_subjects)
print(val_subjects)

X,y = load_data_for_training(train_subjects,  
                             sagittal_segmentations_path,coronal_segmentations_path, axial_segmentations_path, MRI_PATH, GT_PATH, 
                             BLOCK_SIZE = BLOCK_SIZE, USE_PROBS=use_probabilities)

X_val, y_val = load_data_for_training(val_subjects,  
                             sagittal_segmentations_path,coronal_segmentations_path, axial_segmentations_path, MRI_PATH, GT_PATH, 
                             BLOCK_SIZE = BLOCK_SIZE, USE_PROBS=use_probabilities)


# X_fold2 = np.copy(X)
# Y_fold2 = np.copy(y)

# X_fold3 = np.copy(X)
# Y_fold3 = np.copy(y)

# X_fold4 = np.copy(X)
# Y_fold4 = np.copy(y)

# X_fold5 = np.copy(X)
# Y_fold5 = np.copy(y)

# X_fold6 = np.copy(X)
# Y_fold6 = np.copy(y)

# X_fold7 = np.copy(X)
# Y_fold7 = np.copy(y)


# X = np.concatenate([X_fold2, X_fold3, X_fold4, X_fold5, X_fold6, X_fold7], axis=0)
# y = np.concatenate([Y_fold2, Y_fold3, Y_fold4, Y_fold5, Y_fold6, Y_fold7], axis=0)

print(X.shape)
y.shape
X_val.shape
y_val.shape
X.dtype

if use_tf_Dataset:
        
    # Create training and validation datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X, y))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    
    train_dataset = train_dataset.batch(batch_size)
    val_dataset = val_dataset.batch(1) 
    
    history = model.fit(train_dataset,epochs=100, validation_data=val_dataset, 
                        callbacks=[EarlyStopping(monitor='val_loss', patience=20,min_delta=0.01,verbose=1 ),
                                   ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20, min_lr=0.001)])
else:
    history = model.fit(X,y,epochs=500, batch_size=batch_size, validation_data=(X_val,y_val), shuffle=True, 
                        callbacks=[EarlyStopping(monitor='val_loss', patience=20,min_delta=0.01,verbose=1 ),
                                   ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20, min_lr=0.001)])


plt.figure(figsize=(15,5))
plt.subplot(131)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()

plt.subplot(132)
for k in ['dice_coef_multilabel_bin0', 'dice_coef_multilabel_bin1', 'dice_coef_multilabel_bin2', 'dice_coef_multilabel_bin3', 'dice_coef_multilabel_bin4', 'dice_coef_multilabel_bin5', 'dice_coef_multilabel_bin6']:
    plt.plot(history.history[k], label=k)
# plt.legend(loc='lower right')

plt.subplot(133)
for k in 'val_dice_coef_multilabel_bin0', 'val_dice_coef_multilabel_bin1', 'val_dice_coef_multilabel_bin2', 'val_dice_coef_multilabel_bin3', 'val_dice_coef_multilabel_bin4', 'val_dice_coef_multilabel_bin5', 'val_dice_coef_multilabel_bin6':
    plt.plot(history.history[k], label=k)
# plt.legend(loc='lower right')



# from utils import save_model_and_weights
# save_model_and_weights(model, 'consensus_layer_noBuchwald', '/home/deeperthought/Projects/Multiaxial/Sessions/Consensus_model_sessions/')

#%%   ################# Full head prediction ##################3

LOAD_MODEL = False
if LOAD_MODEL:
    model = tf.keras.models.load_model('/home/deeperthought/Projects/Multiaxial/Sessions/Consensus_model_sessions/consensus_layer_fold2.h5')

val_results = {}

#w = model.get_weights()

#np.save('/home/deeperthought/Projects/Multiaxial/Sessions/Consensus_model_sessions/weights_noBuchwald.npy',w)

# val_subjects = train_subjects

for subj in val_subjects:
    print(subj)
    subj_sagittal_seg = load_scans(subj, sagittal_segmentations_path)
    subj_coronal_seg = load_scans(subj, coronal_segmentations_path)
    subj_axial_seg = load_scans(subj, axial_segmentations_path)
    subj_MRI = load_scans(subj, MRI_PATH)
    subj_GT = load_scans(subj, GT_PATH)
           
    X_test,y_test = prepare_data_for_model(subj_MRI, subj_sagittal_seg, subj_coronal_seg, subj_axial_seg, subj_GT, use_probs=use_probabilities)
    
    print('getting model consensus')

    yhat = model.predict(np.expand_dims(X_test,0))
    pred = np.argmax(yhat[0],-1)

  
    subj_sagittal_seg = np.argmax(subj_sagittal_seg, -1)
    subj_coronal_seg = np.argmax(subj_coronal_seg, -1)
    subj_axial_seg = np.argmax(subj_axial_seg, -1)

    vote_pred = make_vote_consensus(subj_sagittal_seg,subj_coronal_seg , subj_axial_seg)
    
    
    
    dice_vote = dice_coefficient(subj_GT, vote_pred) 
    dice_consensus = dice_coefficient(subj_GT, pred)
    dice_sagittal = dice_coefficient(subj_GT, subj_sagittal_seg)
    dice_coronal = dice_coefficient(subj_GT, subj_coronal_seg)
    dice_axial = dice_coefficient(subj_GT, subj_axial_seg)
    
    val_results[subj] = {'consensus':dice_consensus, 'vote':dice_vote,'sagittal':dice_sagittal, 'coronal':dice_coronal, 'axial':dice_axial}
        
    
    
    INDEX=105
    plt.figure(figsize=(15,10))
    plt.subplot(241); plt.imshow(np.rot90(subj_MRI[INDEX])); plt.title('MRI')
    plt.subplot(242); plt.imshow(np.rot90(subj_GT[INDEX])); plt.title('GT')
    plt.subplot(243); plt.imshow(np.rot90(pred[INDEX])); plt.title('Consensus')
    plt.subplot(245); plt.imshow(np.rot90(subj_sagittal_seg[INDEX])); plt.title('Sagittal')
    plt.subplot(246); plt.imshow(np.rot90(subj_coronal_seg[INDEX])); plt.title('Coronal')
    plt.subplot(247); plt.imshow(np.rot90(subj_axial_seg[INDEX])); plt.title('Axial')
    plt.subplot(248); plt.imshow(np.rot90(vote_pred[INDEX])); plt.title('Vote')
    plt.show()
    plt.close()
    
    # INDEX=140   
    # plt.figure(figsize=(15,10))
    # plt.subplot(241); plt.imshow(np.rot90(subj_MRI[:,:,INDEX])); plt.title('MRI')
    # plt.subplot(242); plt.imshow(np.rot90(subj_GT[:,:,INDEX])); plt.title('GT')
    # plt.subplot(243); plt.imshow(np.rot90(pred[:,:,INDEX])); plt.title('Consensus')
    # plt.subplot(245); plt.imshow(np.rot90(subj_sagittal_seg[:,:,INDEX])); plt.title('Sagittal')
    # plt.subplot(246); plt.imshow(np.rot90(subj_coronal_seg[:,:,INDEX])); plt.title('Coronal')
    # plt.subplot(247); plt.imshow(np.rot90(subj_axial_seg[:,:,INDEX])); plt.title('Axial')
    # plt.subplot(248); plt.imshow(np.rot90(vote_pred[:,:,INDEX])); plt.title('Vote')
    
    
    
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

plt.plot(np.random.normal(loc=0, scale=0.1,size=len(consensus_results)), sagittal_results, '.', markersize=15, alpha=0.8)
plt.plot(np.random.normal(loc=1, scale=0.1,size=len(consensus_results)), coronal_results, '.', markersize=15, alpha=0.8)
plt.plot(np.random.normal(loc=2, scale=0.1,size=len(consensus_results)), axial_results, '.', markersize=15, alpha=0.8)
plt.plot(np.random.normal(loc=3, scale=0.1,size=len(consensus_results)), vote_results, '.', markersize=15, alpha=0.8)
plt.plot(np.random.normal(loc=4, scale=0.1,size=len(consensus_results)), consensus_results, '.', markersize=15, alpha=0.8)
plt.plot([np.median(sagittal_results), np.median(coronal_results), np.median(axial_results), np.median(vote_results), np.median(consensus_results)], color='k')
plt.plot([np.mean(sagittal_results), np.mean(coronal_results), np.mean(axial_results), np.mean(vote_results), np.mean(consensus_results)], color='gray')

plt.xticks([0,1,2,3,4],['sagittal','coronal','axial','vote','consensus_3DCNN'])
plt.ylabel('Dice')
# plt.ylim([0.5,1])
plt.grid()

# m3 = [0.8915508122208011,
#   0.8710431544321071,
#   0.8880955135323095,
#   0.8852586753607063,
#   0.8777911921696024]

# m23 = [0.8867719566961542,
#   0.873515486662581,
#   0.8969770045250006,
#   0.8845136336794808,
#   0.8887021473085913]

# m4 = [0.8898699312188495,
#   0.8664861744136897,
#   0.8801249135702688,
#   0.8841473647321946,
#   0.8770179970448684]

# m24 = [0.8897810902696416,
#   0.8689967980764793,
#   0.8912767250646857,
#   0.8826626827042112,
#   0.8863489533193692]

# m34 = [0.8902268381963184,
#   0.866708417595942,
#   0.8809867116038193,
#   0.8847991783480543,
#   0.8785356067777071]

# m234567 = [0.888934376403281,
#  0.8692526497088394,
#  0.8837409780116973,
#  0.8828854004212048,
#  0.8754165206746495]

# plt.plot([m3, m23, m4, m24, m34, m234567], color='lightblue')
# plt.plot([np.median(m1), np.median(m2)], color='k')
