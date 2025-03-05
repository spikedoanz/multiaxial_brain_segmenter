#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 12:20:18 2024

@author: deeperthought
"""


import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


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
