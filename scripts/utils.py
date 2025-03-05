#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 12:20:18 2024

@author: deeperthought
"""


import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


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
        dice -= dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
        
    return numLabels + dice

def dice_coef_multilabel_bin0(y_true, y_pred):
  numerator = 2 * tf.math.reduce_sum(y_true[:,:,:,0] * y_pred[:,:,:,0])
  denominator = tf.math.reduce_sum(y_true[:,:,:,0] + y_pred[:,:,:,0])
  return numerator / denominator

def dice_coef_multilabel_bin1(y_true, y_pred):
  numerator = 2 * tf.math.reduce_sum(y_true[:,:,:,1] * y_pred[:,:,:,1])
  denominator = tf.math.reduce_sum(y_true[:,:,:,1] + y_pred[:,:,:,1])
  return numerator / denominator

def dice_coef_multilabel_bin2(y_true, y_pred):
    
  numerator = 2 * tf.math.reduce_sum(y_true[:,:,:,2] * y_pred[:,:,:,2])
  denominator = tf.math.reduce_sum(y_true[:,:,:,2] + y_pred[:,:,:,2])
  return numerator / denominator

def dice_coef_multilabel_bin3(y_true, y_pred):
  numerator = 2 * tf.math.reduce_sum(y_true[:,:,:,3] * y_pred[:,:,:,3])
  denominator = tf.math.reduce_sum(y_true[:,:,:,3] + y_pred[:,:,:,3])
  return numerator / denominator

def dice_coef_multilabel_bin4(y_true, y_pred):
  numerator = 2 * tf.math.reduce_sum(y_true[:,:,:,4] * y_pred[:,:,:,4])
  denominator = tf.math.reduce_sum(y_true[:,:,:,4] + y_pred[:,:,:,4])
  return numerator / denominator

def dice_coef_multilabel_bin5(y_true, y_pred):
  numerator = 2 * tf.math.reduce_sum(y_true[:,:,:,5] * y_pred[:,:,:,5])
  denominator = tf.math.reduce_sum(y_true[:,:,:,5] + y_pred[:,:,:,5])
  return numerator / denominator

def dice_coef_multilabel_bin6(y_true, y_pred):
  numerator = 2 * tf.math.reduce_sum(y_true[:,:,:,6] * y_pred[:,:,:,6])
  denominator = tf.math.reduce_sum(y_true[:,:,:,6] + y_pred[:,:,:,6])
  return numerator / denominator

def segment_MRI(img, coords, model_sagittal=None, model_axial=None, model_coronal=None, consensus_model=None):
    
    model_segmentation_sagittal = None
    model_segmentation_coronal = None
    model_segmentation_axial = None
    
    if model_sagittal is not None:
        yhat_sagittal = model_sagittal.predict([np.expand_dims(img,-1), coords], batch_size=1, verbose=1)
        model_segmentation_sagittal = yhat_sagittal 
        
    if model_coronal is not None:
        yhat_coronal = model_coronal.predict([np.expand_dims(np.swapaxes(img, 0, 1),-1), np.swapaxes(coords, 0, 1)], batch_size=1, verbose=1)
        model_segmentation_coronal = np.swapaxes(yhat_coronal,0,1)          

    if model_axial is not None:
        yhat_axial = model_axial.predict([np.expand_dims(np.swapaxes(np.swapaxes(img, 1,2), 0,1),-1), np.swapaxes(np.swapaxes(coords, 1,2), 0,1)], batch_size=1, verbose=1)
        model_segmentation_axial = np.swapaxes(np.swapaxes(yhat_axial,0,1), 1,2)


    # Add Consensus Here
       
    X = np.concatenate([np.expand_dims(img,-1), model_segmentation_sagittal,model_segmentation_coronal,model_segmentation_axial ],-1)
    print('getting model consensus')
    yhat = consensus_model.predict(np.expand_dims(X,0))
    print(f"Consensus model output shape: {yhat.shape}")
    
    # Handle different possible output shapes from the consensus model
    if len(yhat.shape) == 5:  # Shape: (1, 256, 256, 256, 7)
        pred = np.argmax(yhat[0], axis=-1)  # Convert to shape (256, 256, 256)
    else:  # Shape: (1, 256, 256, 256)
        pred = yhat[0]  # Already in the correct shape

    
    return pred
