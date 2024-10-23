# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 10:47:55 2024

@author: Andrew
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import resize
from keras.utils import to_categorical
from tensorflow.keras.layers import Input, Conv3D
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras import backend as K
from utils_head_segmenter import vol_to_patches, patches_to_vol, plot_comparison
import gc
gc.enable()
if tf.__version__[0] == '1':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list="0"
    tf.keras.backend.set_session(tf.Session(config=config))

elif tf.__version__[0] == '2':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      # Restrict TensorFlow to only use the first GPU
      try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
      except RuntimeError as e:
        # Visible devices must be set at program startup
        print(e)
        
 #%% METRICS AND LOSSES
         
def dice_coef_numpy(y_true, y_pred):
    smooth = 1e-6
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    
def Generalised_dice_coef_multilabel_numpy(y_true, y_pred, numLabels=2):
    dice=0
    for index in range(numLabels):
        dice += dice_coef_numpy(y_true == index, y_pred == index)
    return  dice/float(numLabels)


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


def dice_coef_multilabel_bin0_numpy(y_true, y_pred):
    dice = dice_coef_numpy(y_true[:,:,:,0], np.round(y_pred[:,:,:,0]))
    return dice

def dice_coef_multilabel_bin1_numpy(y_true, y_pred):
    dice = dice_coef_numpy(y_true[:,:,:,1], np.round(y_pred[:,:,:,1]))
    return dice


def segment_in_axis(mri_padded, model, orientation='sagittal'):
    INDEX = 0
    SLICES = mri_padded.shape[0]
    segmentation = np.zeros((mri_padded.shape))
    segmentation_probs = np.zeros((256,256,256,7))
    
    for INDEX in range(SLICES):
        #print('{}/{}'.format(INDEX,SLICES))
        if orientation == 'sagittal':
            x = mri_padded[INDEX]
        elif orientation == 'coronal':
            x = mri_padded[:,INDEX]
        elif orientation == 'axial':
            x = mri_padded[:,:,INDEX]
            
        x = np.expand_dims(np.expand_dims(x,0),-1)    
        yhat = model.predict(x)
        
        seg_slices = np.argmax(yhat,-1)
        
        if orientation == 'sagittal':
            segmentation[INDEX] = seg_slices
            segmentation_probs[INDEX] = yhat
        elif orientation == 'coronal':
            segmentation[:,INDEX] = seg_slices
            segmentation_probs[:,INDEX] = yhat
        elif orientation == 'axial':
            segmentation[:,:,INDEX] = seg_slices
            segmentation_probs[:,:,INDEX] = yhat
    
    return segmentation, segmentation_probs

def segment_and_dice(mri_padded, model, orientation='sagittal', seg_padded = np.array([])):
    INDEX = 0
    SLICES = mri_padded.shape[0]
    segmentation = np.zeros((mri_padded.shape))
    segmentation_probs = np.zeros((256,256,256,7))
    
    for INDEX in range(SLICES):
        #print('{}/{}'.format(INDEX,SLICES))
        if orientation == 'sagittal':
            x = mri_padded[INDEX]
        elif orientation == 'coronal':
            x = mri_padded[:,INDEX]
        elif orientation == 'axial':
            x = mri_padded[:,:,INDEX]
            
        x = np.expand_dims(np.expand_dims(x,0),-1)    
        yhat = model.predict(x)
        
        seg_slices = np.argmax(yhat,-1)
        
        if orientation == 'sagittal':
            segmentation[INDEX] = seg_slices
            segmentation_probs[INDEX] = yhat
        elif orientation == 'coronal':
            segmentation[:,INDEX] = seg_slices
            segmentation_probs[:,INDEX] = yhat
        elif orientation == 'axial':
            segmentation[:,:,INDEX] = seg_slices
            segmentation_probs[:,:,INDEX] = yhat
    
    if len(seg_padded) == 0:
        print('no segmentation given. No dice.')
        return segmentation, 0
            
    return segmentation, Generalised_dice_coef_multilabel_numpy(seg_padded, segmentation, 7), segmentation_probs

    
#%% USER INPUT
# subjects = ['Andy','Bikson','Parra','Edwards']  # Add your subjects here
# processing_steps = ['','.deface','.reface','.reface_plus']
# X_train = []
# Y_train = []
# X_val = []
# Y_val = []

all_MRI = r'C:\Users\Andrew\Documents\All_Heads_Training\all_MRI.txt'
all_MRI_table = pd.read_csv(all_MRI, header=None, delimiter='\t')
all_MRI_paths = all_MRI_table.values.flatten().tolist() 
all_SEG = r'C:\Users\Andrew\Documents\All_Heads_Training\all_LABEL.txt'
all_SEG_table = pd.read_csv(all_SEG, header=None, delimiter='\t')
all_SEG_paths = all_SEG_table.values.flatten().tolist() 

all_subjects = []

for path in all_MRI_paths:
    subject = path.split('\\')[-2]
    all_subjects.append(subject)

# data = np.load(r'C:\Users\Andrew\Documents\All_Heads_Training\DATA\data.npy', allow_pickle=True).item() #data file of train/val/test splits with slices

# subjects = []
# train_subjects = ['GU008','GU011','GU012','Parra']
# val_subjects = ['GU021','NC008','NC025','NC029',]

# Filter MRI paths based on common subjects

#subjects.append(selected_paths)
# def process_subjects(subject):
subjects = [path for path in all_MRI_paths if path.split('\\')[-2] in all_subjects]
labels = [path for path in all_SEG_paths if path.split('\\')[-2] in all_subjects]
# PATH = r'C:\Users\Andrew\Documents\All_Heads_Training\Sessions\Fold 1\\'
# PATH = r'C:\Users\Andrew\Documents\All_Heads_Training\NewRun_removedLogBug\\'
SAGITTAL_MODEL_SESSION_PATH = 'sagittal_best_model.h5'
AXIAL_MODEL_SESSION_PATH = 'axial_best_model.h5'
CORONAL_MODEL_SESSION_PATH =  'coronal_best_model.h5'


   
#%%
print('Loading models...')
my_custom_objects = {'Generalised_dice_coef_multilabel7':Generalised_dice_coef_multilabel7,
                                'dice_coef_multilabel_bin0':dice_coef_multilabel_bin0,
                                'dice_coef_multilabel_bin1':dice_coef_multilabel_bin1,
                                'dice_coef_multilabel_bin2':dice_coef_multilabel_bin2,
                                'dice_coef_multilabel_bin3':dice_coef_multilabel_bin3,
                                'dice_coef_multilabel_bin4':dice_coef_multilabel_bin4,
                                'dice_coef_multilabel_bin5':dice_coef_multilabel_bin5,
                                'dice_coef_multilabel_bin6':dice_coef_multilabel_bin6}


model_sagittal = tf.keras.models.load_model(SAGITTAL_MODEL_SESSION_PATH, 
                                custom_objects = my_custom_objects)

model_axial = tf.keras.models.load_model(AXIAL_MODEL_SESSION_PATH,
                                  custom_objects = my_custom_objects)

model_coronal = tf.keras.models.load_model(CORONAL_MODEL_SESSION_PATH,
                                  custom_objects = my_custom_objects)
X = []
Y = []
for subject, label in zip(subjects, labels):
    SUBJECT_PATH = subject
    print(subject)
    SEGMENTATION_PATH = label  # Set SEGMENTATION_PATH to the corresponding label for the current subject
    

    SEGMENTATION_METHOD = 2
    
    #SUBJECT_NAME = SUBJECT_PATH.split(os.sep)[-1].split('.')[0]
    SUBJECT_NAME = subject.split('\\')[-2]
    # OUTPUT_PATH = r'C:\Users\Andrew\Documents\multiaxial_brain_segmenter-main'
    # output_file_path = OUTPUT_PATH + os.sep + '{}_CONSENSUS.nii'.format(SUBJECT_NAME)
    
    # if os.path.exists(output_file_path):
    #     print("File already exists. Skipping...")
    # else:
        
  
    #%%



    print('Loading and preprocessing {} ...'.format(SUBJECT_NAME))
    nii = nib.load(SUBJECT_PATH)
    affine = nii.affine
    mri = nii.get_fdata()
    mri_shape = mri.shape
    mri = resize(mri, output_shape=((mri.shape[0], 256, 256)),  anti_aliasing=True, preserve_range=True)    
    mri = mri / np.percentile(mri, 95)

    if len(SEGMENTATION_PATH) > 0:
        seg = nib.load(SEGMENTATION_PATH).get_fdata()     
        seg = np.array(seg, dtype=float)
        np.unique(seg)
        seg = seg - 1    #see if this is the problem
        
        if seg.shape[-1] < 300:
            seg = resize(seg, output_shape=(seg.shape[0], 256, 256), order=0, anti_aliasing=False)    
        
        else:
            # Gradual resizing seems to work better somehow...
            seg = resize(seg, output_shape=(seg.shape[0], 256, 300), order=0, anti_aliasing=False)    
            seg = resize(seg, output_shape=(seg.shape[0], 256, 275), order=0, anti_aliasing=False)    
            seg = resize(seg, output_shape=(seg.shape[0], 256, 256), order=0, anti_aliasing=False)    
        
        seg.max()
        seg = np.array(seg, dtype=int)

    TARGET_WIDTH = 256
    padding_width = TARGET_WIDTH - mri.shape[0]

    mri_padded = np.pad(mri, ((int(padding_width/2),int(padding_width/2) + padding_width%2),(0,0),(0,0)), 'constant')
    if len(SEGMENTATION_PATH) > 0:
        seg_padded = np.pad(seg, ((int(padding_width/2),int(padding_width/2) + padding_width%2),(0,0),(0,0)), 'minimum')
    else:
        seg_padded = np.array([])
        
    if SUBJECT_NAME in['Andy','Bikson','Edwards','Parra']:
        seg_padded = seg_padded + 1 
        
    print('Segmenting scan in 3 axis...')
    if SEGMENTATION_METHOD == 1:
        sagittal_segmentation, sagittal_segmentation_probs = segment_in_axis(mri_padded, model_sagittal, 'sagittal')
        axial_segmentation, axial_segmentation_probs = segment_in_axis(mri_padded, model_axial, 'axial')
        coronal_segmentation, coronal_segmentation_probs = segment_in_axis(mri_padded, model_coronal, 'coronal')
        
    elif SEGMENTATION_METHOD == 2:
        sagittal_segmentation = model_sagittal.predict(np.expand_dims(mri_padded,-1), batch_size=1)
        coronal_segmentation = model_coronal.predict(np.expand_dims(np.swapaxes(mri_padded, 0, 1),-1), batch_size=1)
        axial_segmentation = model_axial.predict(np.expand_dims(np.swapaxes(np.swapaxes(mri_padded, 1,2), 0,1),-1), batch_size=1)
      
        
        # sagittal_segmentation_probs = np.amax(sagittal_segmentation, axis=-1)
        # sagittal_segmentation = np.argmax(sagittal_segmentation, axis=-1)
        
        # coronal_segmentation_probs = np.amax(coronal_segmentation, axis=-1)
        # coronal_segmentation = np.argmax(coronal_segmentation, axis=-1)
        
        # axial_segmentation_probs = np.amax(axial_segmentation, axis=-1)
        # axial_segmentation = np.argmax(axial_segmentation, axis=-1)


        # # Remove padding
        # if padding_width > 0:
        #     sagittal_segmentation = sagittal_segmentation[int(padding_width/2):-(int(padding_width/2) + padding_width%2)]
        #     axial_segmentation = axial_segmentation[int(padding_width/2):-(int(padding_width/2) + padding_width%2)]
        #     coronal_segmentation = coronal_segmentation[int(padding_width/2):-(int(padding_width/2) + padding_width%2)]
    
        # # Resize to original
        # sagittal_segmentation = resize(sagittal_segmentation, output_shape=mri_shape, order=0, anti_aliasing=True, preserve_range=True)    
        # axial_segmentation = resize(axial_segmentation, output_shape=mri_shape, order=0, anti_aliasing=True, preserve_range=True)    
        # coronal_segmentation = resize(coronal_segmentation, output_shape=mri_shape, order=0, anti_aliasing=True, preserve_range=True)    
        # #resize probs
        # sagittal_segmentation_probs = resize(sagittal_segmentation_probs, output_shape=mri_shape, order=0, anti_aliasing=True, preserve_range=True)    
        # axial_segmentation_probs = resize(axial_segmentation_probs, output_shape=mri_shape, order=0, anti_aliasing=True, preserve_range=True)    
        # coronal_segmentation_probs = resize(coronal_segmentation_probs, output_shape=mri_shape, order=0, anti_aliasing=True, preserve_range=True)  
        
        # seg = resize(seg, output_shape=mri_shape, order=0, anti_aliasing=True, preserve_range=True)  
        
        
        # sagittal_segmentation = np.array(sagittal_segmentation, dtype='int8')
        # axial_segmentation = np.array(axial_segmentation, dtype='int8')
        # coronal_segmentation = np.array(coronal_segmentation, dtype='int8')
 
        axial_segmentation = np.transpose(axial_segmentation, (1, 2, 0, 3))
        coronal_segmentation = np.transpose(coronal_segmentation, (1, 0, 2, 3))
        sagittal_segmentation = sagittal_segmentation
        
        # np.savez('NC029',axial = axial_segmentation, coronal= coronal_segmentation, sagittal=sagittal_segmentation)
        # np.savez('NC029_label',seg = seg_padded)
        data = vol_to_patches(axial_segmentation,coronal_segmentation, sagittal_segmentation,seg_padded)
                    
        patches_dir = 'Best_Patches'
        if not os.path.exists(patches_dir):
            os.makedirs(patches_dir)
        np.savez(f'Best_Patches/{SUBJECT_NAME}', input_data_patches =  data[0] ,truth_data_patches =  data[1] )

        # y = model.predict(data[0], batch_size=1) #takes 1 at a time 64x64x64x7 cube and outputs 64x64x64x1
        # pred, label = patches_to_vol(y,data[1])
        # plot_comparison(axial_segmentation,coronal_segmentation, sagittal_segmentation, pred, label,SUBJECT_NAME, slice_num=120)
        
        # pred = np.argmax(pred,axis=-1)
        # pred = np.array(pred, dtype='int8')
        # nii_out = nib.Nifti1Image(pred, affine)
        # nib.save(nii_out,f'Patches/{SUBJECT_NAME}_pred.nii')
        # nib.save(nii_out,fr'C:\Users\Andrew\Documents\ROAST\P909\{SUBJECT_NAME}_pred.nii')
        
        # label = np.argmax(label,axis=-1)
        # label = np.array(label, dtype='int8')
        # nii_out = nib.Nifti1Image(label, affine)
        # nib.save(nii_out,f'Patches/{SUBJECT_NAME}_label.nii')
        # nib.save(nii_out,fr'C:\Users\Andrew\Documents\ROAST\P909\{SUBJECT_NAME}_label.nii')
        
        X.append(data[0])
        Y.append(data[1])
            
    # return X, Y


    
def load_data(subject):
    X = []
    Y = []
    i = 0
    for subject in subject:
        i = i+1
        print(f'Subject {i}: ',subject)     
        data = np.load(f'Best_Patches/{subject}.npz')
        x = data['input_data_patches']
        y = data['truth_data_patches']
    X.append(x)
    Y.append(y)
    return X,Y

# partition = np.load(r'C:\Users\Andrew\Documents\All_Heads_Training\DATA\data.npy', allow_pickle=True).item()
folds = 1
for i in range(folds):
    train_subjects = all_subjects
    # train_subjects = partition[f'Fold {i+1}']['train_names']
    # val_subjects = partition[f'Fold {i+1}']['validation_names']
          
    print('Train Heads: ',train_subjects)
    
    # X_train,Y_train = process_subjects(train_subjects)
    X_train,Y_train = load_data(train_subjects)
        
    # print('Validation Heads: ',val_subjects)
    
    # X_val,Y_val = process_subjects(val_subjects)
    # X_val,Y_val = load_data(val_subjects)
    
    X_train = np.concatenate(X_train, axis=0)
    Y_train = np.concatenate(Y_train, axis=0)
    # X_val = np.concatenate(X_val, axis=0)
    # Y_val = np.concatenate(Y_val, axis=0)
    
    print('Train Shapes:', X_train.shape, Y_train.shape)
    # print('Val Shapes:', X_val.shape, Y_val.shape)
    
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    with tf.device('/GPU:0'):    
        
        tf.keras.backend.clear_session()
        os.chdir(r'C:\Users\Andrew\Documents\multiaxial_brain_segmenter-main')
         
        input_layer = Input(shape=(None,None,None,21))
        
        conv_layer = Conv3D(filters = 7, kernel_size=(3,3,3), activation='softmax', padding='same')(input_layer)
        
        
        model = Model(inputs=input_layer, outputs=conv_layer)
        model.summary()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5), loss=tf.keras.losses.categorical_crossentropy)
        # tf.keras.utils.plot_model(model, 'Model.png', show_shapes=True)   
        
        BATCH_SIZE = 4
           
        history = model.fit(X_train,Y_train,epochs=100, batch_size=BATCH_SIZE) #, validation_data=(X_val, Y_val), validation_batch_size=BATCH_SIZE)
            
        plt.title(f"Train {len(train_subjects)} heads, Learning Rate 5e-5, Batch Size 4")
        plt.plot(history.history['loss'], label='Train')
        # plt.plot(history.history['val_loss'], label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()  # No need to pass labels here
        plt.savefig('consensus_best_model.png')
        plt.show()

        model.save('consensus_best_model.h5')
            # y = model.predict(data[0], batch_size=1) #takes 1 at a time 64x64x64x7 cube and outputs 64x64x64x1
            # pred, label = patches_to_vol(y,data[1])
            # y.shape
            
            # label = np.argmax(label,axis=-1)
            # label = np.array(label, dtype='int8')
            # nii_out = nib.Nifti1Image(label, affine)
            # nib.save(nii_out,'GU024_label.nii')
                
            # # input_data_patches and truth_data_patches should be defined as numpy arrays of shape (64, 64, 64, 64, 7)
            # pred, label = patches_to_vol(y,data[1])
            
            # # Example usage:
            # plot_comparison(axial, coronal, sagittal, pred, label, 'Parra', slice_num=120)
                    
            # # plt.figure(figsize=(15,10))
            # # plt.subplot(2,3,1); plt.imshow(np.rot90(mri[100]), cmap='gray')
            # # plt.subplot(2,3,2); plt.imshow(np.rot90(sagittal_segmentation[100])); plt.title('Sagittal segmenter')
            # # plt.subplot(2,3,3); plt.imshow(np.rot90(axial_segmentation[100])); plt.title('Axial segmenter')
            # # plt.subplot(2,3,5); plt.imshow(np.rot90(coronal_segmentation[100])); plt.title('Coronal segmenter')
            # # plt.subplot(2,3,6); plt.imshow(np.rot90(vote_vol[100])); plt.title('Majority vote segmenter')
        
        
            # if len(OUTPUT_PATH) > 0:
            #     print('Saving segmentation in {} ...'.format(OUTPUT_PATH + os.sep + '{}.nii'.format(SUBJECT_NAME)))
            #     if not os.path.exists(OUTPUT_PATH):
            #         os.mkdir(OUTPUT_PATH)
            #     nii_out = nib.Nifti1Image(sagittal_segmentation, affine)
            #     nib.save(nii_out, OUTPUT_PATH + os.sep + '{}_sagittal.nii'.format(SUBJECT_NAME))
                
            #     nii_out = nib.Nifti1Image(coronal_segmentation, affine)
            #     nib.save(nii_out, OUTPUT_PATH + os.sep + '{}_coronal.nii'.format(SUBJECT_NAME))
                
            #     nii_out = nib.Nifti1Image(axial_segmentation, affine)
            #     nib.save(nii_out, OUTPUT_PATH + os.sep + '{}_axial.nii'.format(SUBJECT_NAME))
        
            #     nii_out = nib.Nifti1Image(vote_vol, affine)
            #     nib.save(nii_out, OUTPUT_PATH + os.sep + '{}_CONSENSUS.nii'.format(SUBJECT_NAME))

