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
os.chdir('/media/HDD/MultiAxial/scripts/')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.optimizers import Adam

from utils import dice_loss, dice_coef, Generalised_dice_coef_multilabel7,dice_coef_multilabel_bin0,dice_coef_multilabel_bin1, dice_coef_multilabel_bin2,dice_coef_multilabel_bin3, dice_coef_multilabel_bin4,dice_coef_multilabel_bin5,dice_coef_multilabel_bin6
from utils import UNet_v0_2DTumorSegmenter, UNet_v0_2DTumorSegmenter_V2
from utils import MyHistory, my_model_checkpoint
from utils import DataGenerator, DataGenerator2

ORIENTATION = 'sagittal'
#ORIENTATION = 'axial'
#ORIENTATION = 'coronal'

# PARTITIONS_PATH = ''
PARTITIONS_PATH = '/media/HDD/MultiAxial/Data/partitions.npy'

OUTPUT_PATH = '/media/HDD/MultiAxial/Sessions/' 

EPOCHS = 2
BATCH_SIZE = 6
DEPTH = 6
N_BASE_FILTERS = 4
LR=5e-5
activation_name = 'softmax'

DATA_AUGMENTATION = False

ADD_SPATIAL_PRIOR = True
ADD_SPATIAL_EMBEDDING = False

LOAD_MODEL = False

NAME = f'{ORIENTATION}Segmenter_PositionalEncoding_{EPOCHS}epochs_depth{DEPTH}_baseFilters{N_BASE_FILTERS}'

MODEL_SESSION_PATH = f'/home/deeperthought/Projects/Others/2D_brain_segmenter/Sessions/{NAME}/'
     
DATA_PATH = f'/media/HDD/MultiAxial/Data/New_Slices_Coordinates/{ORIENTATION}/MRI/'
Label_PATH =  f'/media/HDD/MultiAxial/Data/New_Slices_Coordinates/{ORIENTATION}/GT/'
COORDS_PATH = f'/media/HDD/MultiAxial/Data/New_Slices_Coordinates/{ORIENTATION}/coords/'

#%% TEST GENERATOR

# partition = np.load(PARTITIONS_PATH, allow_pickle=True).item()
# # Parameters
# params_train = {'dim': (256,256),
#           'batch_size': 6,
#           'n_classes': 7,
#           'n_channels': 1,
#           'shuffledata': True,
#           'data_path':DATA_PATH,
#           'labels_path':Label_PATH,
#           'coords_path':COORDS_PATH,
#           'do_augmentation':False,
#           'use_slice_location':True,
#           'debug':True}

# params_val = {'dim': (256,256),
#           'batch_size': 6,
#           'n_classes': 7,
#           'n_channels': 1,
#           'shuffledata': False,
#           'data_path':DATA_PATH,
#           'labels_path':Label_PATH,
#           'coords_path':COORDS_PATH,
#           'do_augmentation':False,
#           'use_slice_location':True}



# # Generators
# training_generator = DataGenerator2(partition['train'], **params_train)

# training_generator.seed = 1

# training_generator.list_IDs

# output = training_generator.__getitem__(0)

# X,y, ids = output


# positional_encoding = X[1]

# slicenr = positional_encoding[:,:,:,0]
# xgrid = positional_encoding[:,:,:,1]
# ygrid = positional_encoding[:,:,:,2]


# plt.imshow(slicenr[0,:,:]); plt.colorbar()
# plt.imshow(xgrid[0,:,:]); plt.colorbar()
# plt.imshow(ygrid[0,:,:]); plt.colorbar()


# for INDEX in range(6):
#     plt.figure(INDEX, figsize=(15,15))
#     plt.subplot(2,2,1); plt.title('slice:' + str(slicenr[INDEX][0][0]*256))
#     plt.imshow(X[0][INDEX,:,:,0], cmap='gray')
#     plt.subplot(2,2,2)
#     plt.imshow(np.argmax(y[INDEX,:,:,:], -1))#; plt.colorbar()
#     plt.subplot(2,2,3)
#     plt.imshow(ygrid[INDEX])    
    
#     plt.imshow(y[INDEX,:,:,2])#; plt.colorbar()


# model = UNet_v0_2DTumorSegmenter_V2(input_shape =  (256,256,1), pool_size=(2, 2),
#                                   initial_learning_rate=0.001, 
#                                   deconvolution=True, depth=DEPTH, n_base_filters=16,
#                                   activation_name="softmax", L2=1e-5, use_batch_norm=False,
#                                   add_spatial_prior=ADD_SPATIAL_PRIOR)
   
# model.input

# X, Y, ids = training_generator.__getitem__(0)


# X[1] = X[1]/256.

# yhat = model.predict([X[0],X[1]])

# model.fit([X[0],X[1]], Y, epochs=100)

# yhat = model.predict([X[0],X[1]])


# plt.imshow(X[0][3,:,:,0])
# plt.imshow(np.argmax(Y,-1)[3,:,:])

# plt.imshow(np.argmax(yhat,-1)[3,:,:])





# for INDEX in range(6):
#     plt.figure(INDEX)
#     plt.subplot(1,3,1)
#     plt.imshow(X[0][INDEX,:,:,0])
#     plt.subplot(1,3,2)
#     plt.imshow(np.argmax(y[INDEX,:,:,:], -1))#; plt.colorbar()
# #
#     plt.subplot(1,3,3)
#     plt.imshow(np.argmax(y[INDEX,:,:,:], -1))#; plt.colorbar()




#%%  TRAINING SESSIONS


if PARTITIONS_PATH == '':
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
          'batch_size': 6,
          'n_classes': 7,
          'n_channels': 1,
          'shuffledata': True,
          'data_path':DATA_PATH,
          'labels_path':Label_PATH,
          'coords_path':COORDS_PATH,
          'do_augmentation':False,
          'use_slice_location':True,
          'debug':False}

params_val = {'dim': (256,256),
          'batch_size': 6,
          'n_classes': 7,
          'n_channels': 1,
          'shuffledata': False,
          'data_path':DATA_PATH,
          'labels_path':Label_PATH,
          'coords_path':COORDS_PATH,
          'do_augmentation':False,
          'use_slice_location':True}


#%%
model = UNet_v0_2DTumorSegmenter_V2(input_shape =  (256,256,1), pool_size=(2, 2),
                                 initial_learning_rate=LR, 
                                 deconvolution=True, depth=DEPTH, n_base_filters=N_BASE_FILTERS,
                                 activation_name="softmax", L2=1e-5, use_batch_norm=False,
                                 add_spatial_prior=ADD_SPATIAL_PRIOR)
   
model.summary()





# SLICE = 'NC016_slice62.npy'

# x = np.load(DATA_PATH + SLICE)
# y = np.load(Label_PATH + SLICE)
# p = np.load(COORDS_PATH + SLICE)

# x.shape
# y.shape
# p.shape

# plt.imshow(x)
# plt.imshow(y)
# plt.imshow(p[:,:,0])
# plt.imshow(p[:,:,1])
# plt.imshow(p[:,:,2])


# x = np.expand_dims(x,-1)
# x = np.expand_dims(x,0)
# p = np.expand_dims(p,0)


# yhat = model.predict([x, p])

# plt.imshow(np.argmax(yhat,-1)[0])


# y = np.expand_dims(y, -1)
# y = np.expand_dims(y, 0)
# y -= 1
# y = tf.keras.utils.to_categorical(y, num_classes=7)

# model.evaluate(np.array([np.expand_dims(x,-1)]), y)



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

# Generators
training_generator = DataGenerator2(partition['train'], **params_train)
validation_generator = DataGenerator2(partition['validation'], **params_val)

tf.keras.utils.plot_model(model, to_file=OUTPUT_PATH + NAME + '/Model.png', show_shapes=True)
with open(OUTPUT_PATH + NAME + '/modelsummary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        
        
        
# Train model on dataset
history = model.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
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



