#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: deeperthought
"""



import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    # Visible devices must be set at program startup
    print(e)

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


ORIENTATION = 'sagittal'
#ORIENTATION = 'axial'
#ORIENTATION = 'coronal'


PARTITIONS_PATH = '/DATA/data.npy'

OUTPUT_PATH = '/Sessions/' 

EPOCHS = 1000
NAME = '{}_segmenter_NoDataAug_1000epochs'.format(ORIENTATION)


BATCH_SIZE = 6
DEPTH = 7
N_BASE_FILTERS = 48
LR=5e-5
activation_name = 'softmax'
DATA_AUGMENTATION = True



LOAD_MODEL = False
MODEL_SESSION_PATH = '/Sessions/{}_segmenter_NoDataAug_1000epochs/'.format(ORIENTATION)

#%% METRICS AND LOSSES
        
    
def dice_loss(y_true, y_pred):
#  y_true = tf.cast(y_true, tf.float32)
#  y_pred = tf.math.sigmoid(y_pred)
  numerator = 2 * tf.math.reduce_sum(y_true * y_pred)
  denominator = tf.math.reduce_sum(y_true + y_pred)
  return 1 - numerator / denominator

def Generalised_dice_coef_multilabel7(y_true, y_pred, numLabels=7):
    """This is the loss function to MINIMIZE. A perfect overlap returns 0. Total disagreement returns numeLabels"""
    dice=0
    for index in range(numLabels):
        dice -= dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
    return numLabels + dice

def dice_coef(y_true, y_pred):
    smooth = 1e-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f**2) + tf.reduce_sum(y_pred_f**2) + smooth)


def dice_coef_multilabel_bin0(y_true, y_pred):
    dice = dice_coef(y_true[:,:,:,0], tf.math.round(y_pred[:,:,:,0]))
    return dice

def dice_coef_multilabel_bin1(y_true, y_pred):
    dice = dice_coef(y_true[:,:,:,1], tf.math.round(y_pred[:,:,:,1]))
    return dice

def dice_coef_multilabel_bin2(y_true, y_pred):
    dice = dice_coef(y_true[:,:,:,2], tf.math.round(y_pred[:,:,:,2]))
    return dice

def dice_coef_multilabel_bin3(y_true, y_pred):
    dice = dice_coef(y_true[:,:,:,3], tf.math.round(y_pred[:,:,:,3]))
    return dice

def dice_coef_multilabel_bin4(y_true, y_pred):
    dice = dice_coef(y_true[:,:,:,4], tf.math.round(y_pred[:,:,:,4]))
    return dice

def dice_coef_multilabel_bin5(y_true, y_pred):
    dice = dice_coef(y_true[:,:,:,5], tf.math.round(y_pred[:,:,:,5]))
    return dice

def dice_coef_multilabel_bin6(y_true, y_pred):
    dice = dice_coef(y_true[:,:,:,6], tf.math.round(y_pred[:,:,:,6]))
    return dice

def Generalised_dice_coef_multilabel2_numpy(y_true, y_pred, numLabels=2):
    """This is the loss function to MINIMIZE. A perfect overlap returns 0. Total disagreement returns numeLabels"""
    dice=0
    for index in range(numLabels):
        dice -= dice_coef_numpy(y_true[:,:,:,index], y_pred[:,:,:,index])
    return numLabels + dice

def dice_coef_multilabel_bin0_numpy(y_true, y_pred):
    dice = dice_coef_numpy(y_true[:,:,:,0], np.round(y_pred[:,:,:,0]))
    return dice
def dice_coef_multilabel_bin1_numpy(y_true, y_pred):
    dice = dice_coef_numpy(y_true[:,:,:,1], np.round(y_pred[:,:,:,1]))
    return dice

def dice_coef_numpy(y_true, y_pred):
    smooth = 1e-6
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f**2) + np.sum(y_pred_f**2) + smooth)


#%% 2D Unet

from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Activation, BatchNormalization, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.layers import concatenate



def double_conv_block(x, n_filters):
   # Conv2D then ReLU activation
   x = tf.keras.layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   # Conv2D then ReLU activation
   x = tf.keras.layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   return x

def downsample_block(x, n_filters):
   f = double_conv_block(x, n_filters)
   p = tf.keras.layers.MaxPool2D(2)(f)
   p = tf.keras.layers.Dropout(0)(p)
   return f, p

def upsample_block(x, conv_features, n_filters):
   # upsample
   x = tf.keras.layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
   # concatenate
   x = tf.keras.layers.concatenate([x, conv_features])
   # dropout
   x = tf.keras.layers.Dropout(0)(x)
   # Conv2D twice with ReLU activation
   x = double_conv_block(x, n_filters)
   return x



#%% MODEL 2



def create_convolution_block(input_layer, n_filters, kernel=(3, 3), padding='same', strides=(1, 1), L2=0, use_batch_norm=True):

    layer = Conv2D(n_filters, kernel, padding=padding, strides=strides, kernel_regularizer=regularizers.l2(L2))(input_layer)
    if use_batch_norm:
        layer = BatchNormalization()(layer)

    return Activation('relu')(layer)


def get_up_convolution(n_filters, pool_size=(2,2), kernel_size=(2,2), strides=(2, 2),
                       deconvolution=True, bilinear_upsampling=False, L2=0):
    if deconvolution:
        if bilinear_upsampling:
            return Conv2DTranspose(filters=n_filters, kernel_size=(3,3),
                                   strides=strides, trainable=False)#, kernel_initializer=make_bilinear_filter_5D(shape=(3,3,3,n_filters,n_filters)), trainable=False)
        else:
            return Conv2DTranspose(filters=n_filters, kernel_size=(2,2),
                                   strides=strides, kernel_regularizer=regularizers.l2(L2))            
    else:
        return UpSampling2D(size=pool_size)

def my_init(shape, dtype=None):
    return K.random_normal(shape, dtype=dtype)



def UNet_v0_2DTumorSegmenter(input_shape =  (256, 256,1), pool_size=(2, 2),initial_learning_rate=1e-5, deconvolution=True,
                      depth=6, n_base_filters=32, activation_name="softmax", L2=0, use_batch_norm=True):
        """ Simple version, padding 'same' on every layer, output size is equal to input size. Has border artifacts and checkerboard artifacts """
        inputs = Input(input_shape)
        levels = list()
        #current_layer = Conv2D(n_base_filters, (1, 1))(inputs)  # ???? needed??  Not even a nonlinear activation!!!
        current_layer = inputs
    
        # add levels with max pooling
        for layer_depth in range(depth):
            layer1 = create_convolution_block(input_layer=current_layer, kernel=(3,3), n_filters=n_base_filters*(layer_depth+1), padding='same', L2=L2, use_batch_norm=False)
            layer2 = create_convolution_block(input_layer=layer1, kernel=(3,3),  n_filters=n_base_filters*(layer_depth+1), padding='same', L2=L2, use_batch_norm=False)
            if layer_depth < depth - 1:
                current_layer = MaxPooling2D(pool_size=(2,2))(layer2)
                levels.append([layer1, layer2, current_layer])
            else:
                current_layer = layer2
                levels.append([layer1, layer2])

        for layer_depth in range(depth-2, -1, -1):
            
            up_convolution = get_up_convolution(pool_size=(2,2), deconvolution=deconvolution, n_filters=n_base_filters*(layer_depth+1), L2=L2)(current_layer)

            concat = concatenate([up_convolution, levels[layer_depth][1]] , axis=-1)
            current_layer = create_convolution_block(n_filters=n_base_filters*(layer_depth+1),kernel=(3,3), input_layer=concat, padding='same', L2=L2, use_batch_norm=False)
            current_layer = create_convolution_block(n_filters=n_base_filters*(layer_depth+1),kernel=(3,3), input_layer=current_layer, padding='same', L2=L2, use_batch_norm=False)


        current_layer = Conv2D(256, (1, 1), activation='relu')(current_layer)
        current_layer = Conv2D(512, (1, 1), activation='relu')(current_layer)
    
        final_convolution = Conv2D(7, (1, 1))(current_layer)              
        act = Activation(activation_name)(final_convolution)
        
        model = Model(inputs=[inputs], outputs=act)

        model.compile(loss=Generalised_dice_coef_multilabel7, 
                      optimizer=Adam(lr=initial_learning_rate), 
                      metrics=[dice_coef_multilabel_bin0, 
                               dice_coef_multilabel_bin1, 
                               dice_coef_multilabel_bin2,
                               dice_coef_multilabel_bin3,
                               dice_coef_multilabel_bin4,
                               dice_coef_multilabel_bin5,
                               dice_coef_multilabel_bin6])

        return model
    

#%% FROM DIRECTORY



"""
Then add contralateral, clinical, previous exam...

"""

class DataGenerator(tf.keras.utils.Sequence): # inheriting from Sequence allows for multiprocessing functionalities

    def __init__(self, list_IDs, batch_size=4, dim=(256,256,1), n_channels=3,
                 n_classes=2, shuffledata=True, data_path='', labels_path='', 
                 do_augmentation=True, debug=False):
        
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffledata = shuffledata
        self.on_epoch_end()
        self.data_path = data_path
        self.labels_path = labels_path
        self.seed = 0
        self.do_augmentation = do_augmentation
        self.debug = debug
        
        self.augmentor = tf.keras.preprocessing.image.ImageDataGenerator(
                    rotation_range=0,
                    shear_range=0.0,
                    zoom_range=0.0,
                    horizontal_flip=False,
                    vertical_flip=False,
                    fill_mode='nearest',  # ????
                    
                )
        
        self.augmentor_mask = tf.keras.preprocessing.image.ImageDataGenerator( 
                    rotation_range=0,
                    shear_range=0.0,
                    zoom_range=0.0,
                    horizontal_flip=False,
                    vertical_flip=False,
                    fill_mode='constant'
                    
#                    preprocessing_function = lambda x: np.where(x>0, 1, 0).astype('float32')
                    
#                    preprocessing_function = lambda x:np.where(x < 0.5, 0, 
#                                                                (np.where(x < 1.5, 1, 
#                                                                (np.where(x < 2.5, 2, 
#                                                                (np.where(x < 3.5, 3, 
#                                                                (np.where(x < 4.5, 4, 
#                                                                (np.where(x < 5.5, 5, 6))))))))))).astype('float32') 
                    )

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        
        if self.do_augmentation:
            for i in range(self.batch_size):
                X[i] *= np.random.uniform(low=0.9, high=1.1, size=1)                                                     
                    
        y = tf.keras.utils.to_categorical(y, num_classes=self.n_classes) 

        if self.debug:
            return X, y, list_IDs_temp
        else:
            return X, y


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffledata == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))#, dtype='float32')
        y = np.zeros((self.batch_size, self.dim[0], self.dim[1]))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):          
            X[i,:,:,0] = np.load(self.data_path + ID)   # Here we add the path. ID can be the path
            
            y[i] = np.load(self.labels_path + ID)

        y = y-1
        y = np.expand_dims(y, -1) #ImageDataGenerator needs arrays of rank 4. But wants channel dim = 1,3,4
        
        
        
        #y = np.array(y, dtype='float32')
        
        if self.do_augmentation:
            X_gen = self.augmentor.flow(X, batch_size=self.batch_size, shuffle=False, seed=self.seed)
            y_gen = self.augmentor_mask.flow(y, batch_size=self.batch_size, shuffle=False, seed=self.seed)


            return next(X_gen), next(y_gen)
        else:
            return X,y
    
#%% TEST GENERATOR
    
#
#Label_PATH = '/labels/{}/'.format(ORIENTATION)
#DATA_PATH = '/slices/{}/'.format(ORIENTATION)
#partition = np.load(PARTITIONS_PATH, allow_pickle=True).item()
# # Parameters
#params_train = {'dim': (256,256),
#          'batch_size': 6,
#          'n_classes': 7,
#          'n_channels': 1,
#          'shuffledata': True,
#          'data_path':DATA_PATH,
#          'labels_path':Label_PATH,
#           'do_augmentation':True}
#
#params_val = {'dim': (256,256),
#          'batch_size': 6,
#          'n_classes': 7,
#          'n_channels': 1,
#          'shuffledata': False,
#          'data_path':DATA_PATH,
#          'labels_path':Label_PATH,
#           'do_augmentation':False}
#
#
#
# # Generators
#training_generator = DataGenerator(partition['train'], **params_train)
#
#training_generator.seed = 1
#
#X, y = training_generator.__getitem__(2)
#
#for INDEX in range(6):
#    plt.figure(INDEX, figsize=(15,15))
#    plt.subplot(1,2,1)
#    plt.imshow(X[INDEX,:,:,0])
#    plt.subplot(1,2,2)
##    plt.imshow(np.argmax(y[INDEX,:,:,:], -1))#; plt.colorbar()
#    plt.imshow(y[INDEX,:,:,0])#; plt.colorbar()
#


#y = model.predict(X)
#for INDEX in range(6):
#    plt.figure(INDEX)
#    plt.subplot(1,2,1)
#    plt.imshow(X[INDEX,:,:,0])
#    plt.subplot(1,2,2)
#    plt.imshow(np.argmax(y[INDEX,:,:,:], -1))#; plt.colorbar()
#

#%%


class MyHistory(tf.keras.callbacks.Callback):
    def __init__(self, OUT, NAME, loss=[], val_loss=[], 
                 dice_coef_multilabel_bin0=[], dice_coef_multilabel_bin1=[], 
                 dice_coef_multilabel_bin2=[], dice_coef_multilabel_bin3=[], 
                 dice_coef_multilabel_bin4=[], dice_coef_multilabel_bin5=[], dice_coef_multilabel_bin6=[], 
                 val_dice_coef_multilabel_bin0=[], val_dice_coef_multilabel_bin1=[], 
                 val_dice_coef_multilabel_bin2=[], val_dice_coef_multilabel_bin3=[], 
                 val_dice_coef_multilabel_bin4=[], val_dice_coef_multilabel_bin5=[], val_dice_coef_multilabel_bin6=[]):
        
        self.OUT = OUT
        self.NAME = NAME  
        
        self.loss = loss
        self.dice_coef_multilabel_bin0 = dice_coef_multilabel_bin0
        self.dice_coef_multilabel_bin1 = dice_coef_multilabel_bin1
        self.dice_coef_multilabel_bin2 = dice_coef_multilabel_bin2
        self.dice_coef_multilabel_bin3 = dice_coef_multilabel_bin3
        self.dice_coef_multilabel_bin4 = dice_coef_multilabel_bin4
        self.dice_coef_multilabel_bin5 = dice_coef_multilabel_bin5
        self.dice_coef_multilabel_bin6 = dice_coef_multilabel_bin6

        self.val_loss = val_loss
        self.val_dice_coef_multilabel_bin0 = val_dice_coef_multilabel_bin0
        self.val_dice_coef_multilabel_bin1 = val_dice_coef_multilabel_bin1
        self.val_dice_coef_multilabel_bin2 = val_dice_coef_multilabel_bin2
        self.val_dice_coef_multilabel_bin3 = val_dice_coef_multilabel_bin3
        self.val_dice_coef_multilabel_bin4 = val_dice_coef_multilabel_bin4
        self.val_dice_coef_multilabel_bin5 = val_dice_coef_multilabel_bin5
        self.val_dice_coef_multilabel_bin6 = val_dice_coef_multilabel_bin6        
        
#    def on_train_begin(self, logs={}):
#
#        self.loss = []
#        self.dice_coef_multilabel_bin0 = []
#        self.dice_coef_multilabel_bin1 = []
#        self.dice_coef_multilabel_bin2 = []
#        self.dice_coef_multilabel_bin3 = []
#        self.dice_coef_multilabel_bin4 = []
#        self.dice_coef_multilabel_bin5 = []
#        self.dice_coef_multilabel_bin6 = []
#
#        self.val_loss = []
#        self.val_dice_coef_multilabel_bin0 = []
#        self.val_dice_coef_multilabel_bin1 = []
#        self.val_dice_coef_multilabel_bin2 = []
#        self.val_dice_coef_multilabel_bin3 = []
#        self.val_dice_coef_multilabel_bin4 = []
#        self.val_dice_coef_multilabel_bin5 = []
#        self.val_dice_coef_multilabel_bin6 = []

#    def on_batch_end(self, batch, logs={}):
#        self.loss.append(logs.get('loss'))
#        self.dice_coef_multilabel_bin1.append(logs.get('dice_coef_multilabel_bin1'))
#        
    def on_epoch_end(self, epoch, logs={}):
        self.loss.append(logs.get('loss'))
        self.dice_coef_multilabel_bin0.append(logs.get('dice_coef_multilabel_bin0'))
        self.dice_coef_multilabel_bin1.append(logs.get('dice_coef_multilabel_bin1'))
        self.dice_coef_multilabel_bin2.append(logs.get('dice_coef_multilabel_bin2'))
        self.dice_coef_multilabel_bin3.append(logs.get('dice_coef_multilabel_bin3'))
        self.dice_coef_multilabel_bin4.append(logs.get('dice_coef_multilabel_bin4'))
        self.dice_coef_multilabel_bin5.append(logs.get('dice_coef_multilabel_bin5'))
        self.dice_coef_multilabel_bin6.append(logs.get('dice_coef_multilabel_bin6'))


        self.val_loss.append(logs.get('val_loss'))
        self.val_dice_coef_multilabel_bin0.append(logs.get('val_dice_coef_multilabel_bin0'))
        self.val_dice_coef_multilabel_bin1.append(logs.get('val_dice_coef_multilabel_bin1'))
        self.val_dice_coef_multilabel_bin2.append(logs.get('val_dice_coef_multilabel_bin2'))
        self.val_dice_coef_multilabel_bin3.append(logs.get('val_dice_coef_multilabel_bin3'))
        self.val_dice_coef_multilabel_bin4.append(logs.get('val_dice_coef_multilabel_bin4'))
        self.val_dice_coef_multilabel_bin5.append(logs.get('val_dice_coef_multilabel_bin5'))
        self.val_dice_coef_multilabel_bin6.append(logs.get('val_dice_coef_multilabel_bin6'))


        plt.figure(figsize=(15,5))
        
        plt.subplot(1,3,1); 
        plt.title('Loss')
        plt.plot(self.loss)
        plt.plot(self.val_loss)
        plt.legend(['Train','Val'])
        plt.grid()
        
        plt.subplot(1,3,2); plt.title('Dice')
        plt.plot(self.dice_coef_multilabel_bin0, label='Bg')
        plt.plot(self.dice_coef_multilabel_bin1, label='Air')
        plt.plot(self.dice_coef_multilabel_bin2, label='GM')
        plt.plot(self.dice_coef_multilabel_bin3, label='WM')
        plt.plot(self.dice_coef_multilabel_bin4, label='CSF')
        plt.plot(self.dice_coef_multilabel_bin5, label='Bone')
        plt.plot(self.dice_coef_multilabel_bin6, label='Skin')
        plt.ylim([0,1])
        plt.legend(loc='upper left', bbox_to_anchor=(0.95, 0.5), ncol=1, fancybox=True, shadow=True)
        plt.grid()
        
        
        plt.subplot(1,3,3); plt.title('Dice Val')
        plt.plot(self.val_dice_coef_multilabel_bin0, label='Bg')
        plt.plot(self.val_dice_coef_multilabel_bin1, label='Air')
        plt.plot(self.val_dice_coef_multilabel_bin2, label='GM')
        plt.plot(self.val_dice_coef_multilabel_bin3, label='WM')
        plt.plot(self.val_dice_coef_multilabel_bin4, label='CSF')
        plt.plot(self.val_dice_coef_multilabel_bin5, label='Bone')
        plt.plot(self.val_dice_coef_multilabel_bin6, label='Skin')
        #plt.legend(loc='upper left', bbox_to_anchor=(0.95, 0.5), ncol=1, fancybox=True, shadow=True)
        plt.grid()
        plt.ylim([0,1])
        
        plt.tight_layout()   
        
        plt.savefig(self.OUT + self.NAME + '/Training_curves.png')
        plt.close()

def freeze_layers(model):
    model_type = type(model) 

    for i in model.layers:
        i.trainable = False
        if type(i) == model_type:
            freeze_layers(i)
    return model


def save_model_and_weights(model, NAME, FOLDER):    
    model_to_save = tf.keras.models.clone_model(model)
    model_to_save.set_weights(model.get_weights())
    model_to_save = freeze_layers(model_to_save)
    model_to_save.save_weights(FOLDER + NAME + '_weights.h5')
    model_to_save.save(FOLDER + NAME + '.h5')       

class my_model_checkpoint(tf.keras.callbacks.Callback):
    
    def __init__(self, MODEL_PATH, MODEL_NAME, val_loss = [999]):
        self.MODEL_PATH = MODEL_PATH
        self.MODEL_NAME = MODEL_NAME    
        self.val_loss = val_loss

    def on_epoch_end(self, epoch, logs={}):
        min_val_loss = min(self.val_loss)
        current_val_loss = logs.get('val_loss')
        self.val_loss.append(current_val_loss)
        print('Min loss so far: {}, new loss: {}'.format(min_val_loss, current_val_loss))
        if current_val_loss < min_val_loss :
            print('New best model! Epoch: {}'.format(epoch))
            save_model_and_weights(self.model, self.MODEL_NAME, self.MODEL_PATH)
        else :
            save_model_and_weights(self.model, '/last_model', self.MODEL_PATH)            




#%%  TRAINING SESSIONS

Label_PATH = '/labels/{}/'.format(ORIENTATION)
DATA_PATH = '/slices/{}/'.format(ORIENTATION)


#available_data = os.listdir(DATA_PATH)
#segmented_data = os.listdir(Label_PATH)
#
#available_data.sort()
#segmented_data.sort()
#
#print('Found {} slices'.format(len(available_data)))
#
#N = len(available_data)
#N_train = int(N*0.95)
#N_val = N - N_train
#
#train_indexes = range(0,N)
#val_indexes = np.random.choice(range(0,N), replace=False, size=N_val)
#train_indexes = list(set(train_indexes) - set(val_indexes))
#
#assert set(train_indexes).intersection(set(val_indexes)) == set()
#len(train_indexes), len(val_indexes)
#
#train_images = [available_data[i] for i in train_indexes]
#val_images = [available_data[i] for i in val_indexes]
#
#partition = {'train':train_images,'validation':val_images}
#
#len(partition['train']), len(partition['validation'])
#
#np.save('/home/deeperthought/Projects/Others/2D_brain_segmenter/Sessions/DATA/data.npy', partition)
#



partition = np.load(PARTITIONS_PATH, allow_pickle=True).item()


#%%

# Parameters
params_train = {'dim': (256,256),
          'batch_size': BATCH_SIZE,
          'n_classes': 7,
          'n_channels': 1,
          'shuffledata': True,
          'data_path':DATA_PATH,
          'labels_path':Label_PATH,
           'do_augmentation':DATA_AUGMENTATION}

params_val = {'dim': (256,256),
          'batch_size': BATCH_SIZE,
          'n_classes': 7,
          'n_channels': 1,
          'shuffledata': False,
          'data_path':DATA_PATH,
          'labels_path':Label_PATH,
           'do_augmentation':False}


model = UNet_v0_2DTumorSegmenter(input_shape =  (256,256,1), pool_size=(2, 2),
                                 initial_learning_rate=LR, 
                                 deconvolution=True, depth=DEPTH, n_base_filters=N_BASE_FILTERS,
                                 activation_name="softmax", L2=1e-5, use_batch_norm=False)
   
model.summary()


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
training_generator = DataGenerator(partition['train'], **params_train)
validation_generator = DataGenerator(partition['validation'], **params_val)

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



