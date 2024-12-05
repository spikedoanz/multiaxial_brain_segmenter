#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 10:56:33 2022

Get ROIs from malignants in the training set, and compare with ground truth segmented locations (if available)

Get some metric for detection. 

@author: deeperthought
"""

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

import os
os.chdir('/home/deeperthought/Projects/Multiaxial/scripts/')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.optimizers import Adam

from utils import dice_loss, dice_coef, Generalised_dice_coef_multilabel7,dice_coef_multilabel_bin0,dice_coef_multilabel_bin1, dice_coef_multilabel_bin2,dice_coef_multilabel_bin3, dice_coef_multilabel_bin4,dice_coef_multilabel_bin5,dice_coef_multilabel_bin6
from utils import UNet_v0_2DTumorSegmenter_V2
from utils import MyHistory, my_model_checkpoint
from utils import DataGenerator2
# from utils import Generalised_dice_coef_neighbors

ORIENTATION = 'sagittal'
# ORIENTATION = 'axial'
# ORIENTATION = 'coronal'
# ORIENTATION = 'all'


PARTITIONS_PATH = '/home/deeperthought/Projects/Multiaxial/Data/data.npy' #'/media/HDD/MultiAxial/Data/partitions.npy'

OUTPUT_PATH = '/home/deeperthought/Projects/Multiaxial/Sessions/' 

STORED_SLICES_PATH = '/media/HDD/MultiAxial/Data/New_slices_coordinates/'
    
EPOCHS = 100
BATCH_SIZE = 6
DEPTH = 6
N_BASE_FILTERS = 16
LR=5e-5
L2=1e-3
activation_name = 'softmax'

DATA_AUGMENTATION = False
ADD_SPATIAL_PRIOR = True
LOAD_MODEL = False
SHUFFLE_VAL = True

AXIS = None#'all'

# NAME = f'NeighborPenalty_Segmenter_PositionalEncoding_{EPOCHS}epochs_depth{DEPTH}_baseFilters{N_BASE_FILTERS}_AndrewPartition_Step2'

NAME = f'HighL2_ShuffleVal_NoEmptySlices_OldDiceMetric_experiment_{ORIENTATION}Segmenter_PositionalEncoding_{EPOCHS}epochs_depth{DEPTH}_baseFilters{N_BASE_FILTERS}_AndrewPartition_Step2'



#%%  TRAINING SESSIONS

MODEL_SESSION_PATH = '/home/deeperthought/kirbyPRO_home/home/MultiAxial/Sessions/' + NAME

DATA_PATH = f'{STORED_SLICES_PATH}{ORIENTATION}/MRI/'
Label_PATH =  f'{STORED_SLICES_PATH}{ORIENTATION}/GT/'
COORDS_PATH = f'{STORED_SLICES_PATH}{ORIENTATION}/coords/'
   


if PARTITIONS_PATH is None:
    print('No pre-saved partition. Creating new partition..')

    available_data = os.listdir(DATA_PATH)
    segmented_data = os.listdir(Label_PATH)
    
    available_data.sort()
    segmented_data.sort()
    
    print('Found {} slices'.format(len(available_data)))
    
    subjects = list(set([x.split('_')[0] for x in available_data]))
    
    print('From {} subjects'.format(len(subjects)))
    
    N_TRAIN = int(len(subjects)*0.9)
    
    train_subj = np.random.choice(subjects, size=N_TRAIN, replace=False)
    
    test_subj = [x for x in subjects if x not in train_subj]
    
    val_subj = np.random.choice(train_subj, size=int(N_TRAIN*0.1), replace=False)
    
    train_subj = [x for x in train_subj if x not in val_subj]
    
    assert set(train_subj).intersection(set(test_subj)) == set()
    assert set(train_subj).intersection(set(val_subj)) == set()
    assert set(val_subj).intersection(set(test_subj)) == set()
    
    assert len(train_subj) + len(val_subj) + len(test_subj) == len(subjects)

    train_images = []
    val_images = []
    for subj in train_subj:
        train_images.extend([x for x in available_data if x.startswith(subj)])
        
    for subj in val_subj:
        val_images.extend([x for x in available_data if x.startswith(subj)])
    
    
    len(train_images), len(val_images)
    
    partition = {'train':train_images,'validation':val_images, 'test':test_subj}
    
    np.save('/media/HDD/MultiAxial/Data/Processed_New_MCS/partitions.npy', partition)



else:
    partition = np.load(PARTITIONS_PATH, allow_pickle=True).item()

    assert set([x[:6] for x in partition['train']]).intersection(set([x[:6] for x in partition['validation']])) == set()
    
    len(set([x[:6] for x in partition['train']])), len(set([x[:6] for x in partition['validation']]))
    len(partition['train']), len(partition['validation'])
    
    
    set([x.split('_')[0] for x in partition['train']])
    set([x.split('_')[0] for x in partition['validation']])
    
    all_data = partition['train'] + partition['validation']


#%%

# Parameters
params_train = {'dim': (256,256),
          'batch_size': BATCH_SIZE,
          'n_classes': 7,
          'n_channels': 1,
          'shuffledata': True,
          'data_path':DATA_PATH,
          'labels_path':Label_PATH,
          'coords_path':COORDS_PATH,
          'do_augmentation':False,
          'axis':AXIS,
          'debug':False}

params_val = {'dim': (256,256),
          'batch_size': BATCH_SIZE,
          'n_classes': 7,
          'n_channels': 1,
          'shuffledata': SHUFFLE_VAL,
          'data_path':DATA_PATH,
          'labels_path':Label_PATH,
          'coords_path':COORDS_PATH,
          'do_augmentation':False}


#%%
model = UNet_v0_2DTumorSegmenter_V2(input_shape =  (256,256,1), pool_size=(2, 2),
                                 initial_learning_rate=LR, 
                                 deconvolution=True, depth=DEPTH, n_base_filters=N_BASE_FILTERS,
                                 activation_name="softmax", L2=L2, use_batch_norm=False,
                                 add_spatial_prior=ADD_SPATIAL_PRIOR)
  

# from utils import Generalised_dice_coef_multilabel7, generalized_dice_coef_multilabel7_present_class, dice_metric0, dice_metric1, dice_metric2, dice_metric3, dice_metric4, dice_metric5, dice_metric6
# from tensorflow.keras.optimizers import Adam


# model.compile(loss=Generalised_dice_coef_multilabel7, 
#           optimizer=Adam(lr=1e-5), 
#           metrics=[dice_metric0, dice_metric1, dice_metric2, dice_metric3, dice_metric4, dice_metric5, dice_metric6]) 

        
# model.compile(loss=Generalised_dice_coef_neighbors, 
#           optimizer=Adam(lr=LR), 
#           metrics=[dice_coef_multilabel_bin0, 
#                    dice_coef_multilabel_bin1, 
#                    dice_coef_multilabel_bin2,
#                    dice_coef_multilabel_bin3,
#                    dice_coef_multilabel_bin4,
#                    dice_coef_multilabel_bin5,
#                    dice_coef_multilabel_bin6])
model.summary()

#%%

if not os.path.exists(OUTPUT_PATH + NAME):
    os.mkdir(OUTPUT_PATH + NAME)

Custom_History = MyHistory(OUTPUT_PATH, NAME)
my_custom_checkpoint = my_model_checkpoint(MODEL_PATH=OUTPUT_PATH+NAME, MODEL_NAME='/best_model')

if LOAD_MODEL:

    my_custom_objects = {'Generalised_dice_coef_multilabel7':Generalised_dice_coef_multilabel7,
                                     'dice_coef_multilabel_bin0':dice_coef_multilabel_bin0,
                                     'dice_coef_multilabel_bin1':dice_coef_multilabel_bin1,
                                     'dice_coef_multilabel_bin2':dice_coef_multilabel_bin2,
                                     'dice_coef_multilabel_bin3':dice_coef_multilabel_bin3,
                                     'dice_coef_multilabel_bin4':dice_coef_multilabel_bin4,
                                     'dice_coef_multilabel_bin5':dice_coef_multilabel_bin5,
                                     'dice_coef_multilabel_bin6':dice_coef_multilabel_bin6}
    
    model_pretrained = tf.keras.models.load_model(MODEL_SESSION_PATH + 'last_model.h5', 
                                       custom_objects = my_custom_objects)
    
    model.get_weights()
    
    model.set_weights(model_pretrained.get_weights())
    

    
    model.compile(loss=Generalised_dice_coef_multilabel7, 
                  optimizer=Adam(lr=1e-5), 
                  metrics=[dice_coef_multilabel_bin0, 
                           dice_coef_multilabel_bin1, 
                           dice_coef_multilabel_bin2,
                           dice_coef_multilabel_bin3,
                           dice_coef_multilabel_bin4,
                           dice_coef_multilabel_bin5,
                           dice_coef_multilabel_bin6])
    
    
    logger = pd.read_csv(OUTPUT_PATH+NAME + '/csvLogger.log')
    
    
    Custom_History = MyHistory(OUTPUT_PATH, NAME, loss=list(logger['loss'].values), val_loss=list(logger['val_loss'].values), 
                 dice_coef_multilabel_bin0=list(logger['dice_coef_multilabel_bin0'].values), dice_coef_multilabel_bin1=list(logger['dice_coef_multilabel_bin1'].values), 
                 dice_coef_multilabel_bin2=list(logger['dice_coef_multilabel_bin2'].values), dice_coef_multilabel_bin3=list(logger['dice_coef_multilabel_bin3'].values), 
                 dice_coef_multilabel_bin4=list(logger['dice_coef_multilabel_bin4'].values), dice_coef_multilabel_bin5=list(logger['dice_coef_multilabel_bin5'].values), 
                 dice_coef_multilabel_bin6=list(logger['dice_coef_multilabel_bin6'].values), 
                 val_dice_coef_multilabel_bin0=list(logger['val_dice_coef_multilabel_bin0'].values), val_dice_coef_multilabel_bin1=list(logger['val_dice_coef_multilabel_bin1'].values), 
                 val_dice_coef_multilabel_bin2=list(logger['val_dice_coef_multilabel_bin2'].values), val_dice_coef_multilabel_bin3=list(logger['val_dice_coef_multilabel_bin3'].values), 
                 val_dice_coef_multilabel_bin4=list(logger['val_dice_coef_multilabel_bin4'].values), val_dice_coef_multilabel_bin5=list(logger['val_dice_coef_multilabel_bin5'].values), 
                 val_dice_coef_multilabel_bin6=list(logger['val_dice_coef_multilabel_bin6'].values))
    
    
    my_custom_checkpoint = my_model_checkpoint(MODEL_PATH=OUTPUT_PATH+NAME, MODEL_NAME='/best_model', val_loss=list(logger['val_loss'].values))

    
    EPOCHS = EPOCHS - len(logger)

#%%

csv_logger = tf.keras.callbacks.CSVLogger(OUTPUT_PATH+NAME + '/csvLogger.log', 
                                     separator=',', 
                                     append=True)

myEarlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                min_delta=0, 
                                                patience=50, 
                                                verbose=1, 
                                                mode='min', 
                                                baseline=None, 
                                                restore_best_weights=False)


# # Center slices from each partition
# partition['train'] = [x for x in partition['train'] if int(x.split('slice')[-1].split('.npy')[0]) in range(110,140)]
# partition['validation'] = [x for x in partition['validation'] if int(x.split('slice')[-1].split('.npy')[0]) in range(110,140)]


# # Non-center slices from each partition
# partition['train'] = [x for x in partition['train'] if int(x.split('slice')[-1].split('.npy')[0]) not in range(50,200)][:300]
# partition['validation'] = [x for x in partition['validation'] if int(x.split('slice')[-1].split('.npy')[0]) not in range(50,200)][:300]


# inspect content of this validation set:
    


# len(partition['train'])

# # Random slices from train set
# parts = np.random.choice(partition['train'], size=600, replace=False)
# partition['train'] = parts[:300]
# partition['validation'] = parts[300:]




# # Generators
# training_generator = DataGenerator2(partition['train'], **params_train)
# validation_generator = DataGenerator2(partition['validation'], **params_val)



# subj_train = list(set([x.split('_')[0] for x in partition['train']]))
# subj_val = list(set([x.split('_')[0] for x in partition['validation']]))



###################### REMOVE BACKGROUND SLICES #####################

segfiles = os.listdir(Label_PATH)

background_slices = []
for f in partition['train']:
    seg = np.load(Label_PATH + f, allow_pickle=True)
    if np.sum(seg != 0) == 0:
        background_slices.append(f)
        
newtrain = [x for x in partition['train'] if x not in background_slices]        
        
background_slices = []
for f in partition['validation']:
    seg = np.load(Label_PATH + f, allow_pickle=True)
    if np.sum(seg != 0) == 0:
        background_slices.append(f)
        
newval = [x for x in partition['validation'] if x not in background_slices]        
    
partition['train'] = newtrain
partition['validation'] = newval
#-----------------------------------------------------------------------

# # inspect content of train and val sets
# val_classes = {0:0,1:0,2:0,3:0,4:0,5:0,6:0}
# for subj in subj_val:
#     print(subj)
#     for f in segfiles:
#         if f.startswith(subj):
            
#             seg = np.load(Label_PATH + f, allow_pickle=True)
            
#             if np.sum(seg != 0) == 0:
                
                
            
#             for cl in val_classes.keys():
                
#                 val_classes[cl] += np.sum(seg == cl)
    
    
# # inspect content of train and val sets
# train_classes = {0:0,1:0,2:0,3:0,4:0,5:0,6:0}
# for subj in subj_train:
#     print(subj)
#     for f in segfiles:
#         if f.startswith(subj):
            
#             seg = np.load(Label_PATH + f, allow_pickle=True)
#             for cl in train_classes.keys():
                
#                 train_classes[cl] += np.sum(seg == cl)


# N = float(train_classes[0])
# for cl in train_classes.keys():
#     train_classes[cl] /= N

# N = float(val_classes[0])
# for cl in val_classes.keys():
#     val_classes[cl] /= N
    
    
# Make validaiton set from train
# newVal = np.random.choice(subj_train, size=10, replace=False)
# newTrain = [x for x in subj_train if x not in newVal]

# part_val = []
# for subj in newVal:
#     part_val.extend([x for x in partition['train'] if x.startswith(subj)])

# part_train = []
# for subj in newTrain:
#     part_train.extend([x for x in partition['train'] if x.startswith(subj)])

# partition['train'] = np.random.choice(part_train, size=600, replace=False)
# partition['validation'] = np.random.choice(part_val, size=600, replace=False)


# partition['train'] = np.random.choice(partition['train'], size=600, replace=False)
# partition['validation'] = np.random.choice(partition['validation'], size=600, replace=False)



# partition['train'] = partition['train'][:600]
# partition['validation'] = partition['validation'][:600]


# Generators
training_generator = DataGenerator2(partition['train'], **params_train)
validation_generator = DataGenerator2(partition['validation'], **params_val)



tf.keras.utils.plot_model(model, to_file=OUTPUT_PATH + NAME + '/Model.png', show_shapes=True)
with open(OUTPUT_PATH + NAME + '/modelsummary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        
# Train model on dataset
history = model.fit_generator(generator=training_generator,
                            validation_data=validation_generator,   ### CHANGE TO VAL!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                            use_multiprocessing=True,
                            max_queue_size=12,
                            workers=8, 
                            verbose=1,
                            steps_per_epoch = len(partition['train']) // BATCH_SIZE,
                            epochs = EPOCHS,
                            shuffle=True,
                            callbacks=[Custom_History, csv_logger, my_custom_checkpoint, myEarlyStop])


df = pd.read_csv(OUTPUT_PATH+NAME + '/csvLogger.log')

plt.figure(figsize=(15,5))

plt.subplot(1,3,1); 
plt.title('Loss')
plt.plot(df['loss'])
plt.plot(df['val_loss'])
plt.legend(['Train','Val'])
plt.grid()

plt.subplot(1,3,2); plt.title('Dice')
plt.plot(df['dice_coef_multilabel_bin0'], label='Background')
plt.plot(df['dice_coef_multilabel_bin1'], label='Air')
plt.plot(df['dice_coef_multilabel_bin2'], label='GM')
plt.plot(df['dice_coef_multilabel_bin3'], label='WM')
plt.plot(df['dice_coef_multilabel_bin4'], label='CSF')
plt.plot(df['dice_coef_multilabel_bin5'], label='Bone')
plt.plot(df['dice_coef_multilabel_bin6'], label='Skin')
axis = plt.axis()
plt.legend(loc='upper left', bbox_to_anchor=(0.95, 0.5), ncol=1, fancybox=True, shadow=True)
plt.grid()


plt.subplot(1,3,3); plt.title('Dice Val')
plt.plot(df['val_dice_coef_multilabel_bin0'], label='Background')
plt.plot(df['val_dice_coef_multilabel_bin1'], label='Air')
plt.plot(df['val_dice_coef_multilabel_bin2'], label='GM')
plt.plot(df['val_dice_coef_multilabel_bin3'], label='WM')
plt.plot(df['val_dice_coef_multilabel_bin4'], label='CSF')
plt.plot(df['val_dice_coef_multilabel_bin5'], label='Bone')
plt.plot(df['val_dice_coef_multilabel_bin6'], label='Skin')
plt.axis(axis)
#plt.legend(loc='upper left', bbox_to_anchor=(0.95, 0.5), ncol=1, fancybox=True, shadow=True)
plt.grid()

plt.tight_layout()
plt.savefig(OUTPUT_PATH + NAME + '/Full_Training_Session.png')
plt.close()



