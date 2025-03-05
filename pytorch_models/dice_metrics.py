"""
PyTorch implementation of Dice coefficient metrics used in the brain segmentation model.
"""

import torch
import torch.nn.functional as F

def dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Calculate Dice coefficient between two tensors.
    
    Args:
        y_true: Ground truth tensor
        y_pred: Predicted tensor
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice coefficient (scalar)
    """
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    intersection = torch.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (torch.sum(y_true_f**2) + torch.sum(y_pred_f**2) + smooth)

def generalized_dice_coef_multilabel7(y_true, y_pred, num_labels=7):
    """
    Calculate generalized Dice coefficient for multi-label segmentation.
    This is the loss function to MINIMIZE. A perfect overlap returns 0. 
    Total disagreement returns numLabels.
    
    Args:
        y_true: Ground truth tensor [batch, h, w, d, num_labels]
        y_pred: Predicted tensor [batch, h, w, d, num_labels]
        num_labels: Number of segmentation classes
        
    Returns:
        Generalized Dice coefficient (scalar)
    """
    dice = 0
    for index in range(num_labels):
        dice -= dice_coef(y_true[..., index], y_pred[..., index])
        
    return num_labels + dice

def dice_coef_multilabel_bin(y_true, y_pred, class_idx):
    """
    Calculate Dice coefficient for a specific class in multi-label segmentation.
    
    Args:
        y_true: Ground truth tensor [batch, h, w, d, num_labels]
        y_pred: Predicted tensor [batch, h, w, d, num_labels]
        class_idx: Class index to calculate Dice coefficient for
        
    Returns:
        Dice coefficient for the specified class (scalar)
    """
    numerator = 2 * torch.sum(y_true[..., class_idx] * y_pred[..., class_idx])
    denominator = torch.sum(y_true[..., class_idx] + y_pred[..., class_idx])
    return numerator / (denominator + 1e-6)  # Add small epsilon to avoid division by zero

# Individual class Dice metrics for each of the 7 classes
def dice_coef_multilabel_bin0(y_true, y_pred):
    return dice_coef_multilabel_bin(y_true, y_pred, 0)

def dice_coef_multilabel_bin1(y_true, y_pred):
    return dice_coef_multilabel_bin(y_true, y_pred, 1)

def dice_coef_multilabel_bin2(y_true, y_pred):
    return dice_coef_multilabel_bin(y_true, y_pred, 2)

def dice_coef_multilabel_bin3(y_true, y_pred):
    return dice_coef_multilabel_bin(y_true, y_pred, 3)

def dice_coef_multilabel_bin4(y_true, y_pred):
    return dice_coef_multilabel_bin(y_true, y_pred, 4)

def dice_coef_multilabel_bin5(y_true, y_pred):
    return dice_coef_multilabel_bin(y_true, y_pred, 5)

def dice_coef_multilabel_bin6(y_true, y_pred):
    return dice_coef_multilabel_bin(y_true, y_pred, 6)