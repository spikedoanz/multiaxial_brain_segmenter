#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch implementation of the brain segmentation model.
"""

import os
import argparse
import numpy as np
import nibabel as nib
import torch
from safetensors.torch import load_file

# Import PyTorch models and utilities
from pytorch_models.models import ViewSpecificModel, ConsensusModel
from pytorch_models.utils import segment_MRI_pytorch
from scripts.preprocessing_lib import preprocess_head_MRI, reshape_back_to_original

# Default paths
SAGITTAL_MODEL_PATH = 'pytorch_models/sagittal_model.safetensors'
AXIAL_MODEL_PATH = 'pytorch_models/axial_model.safetensors'
CORONAL_MODEL_PATH = 'pytorch_models/coronal_model.safetensors'
CONSENSUS_LAYER_PATH = 'pytorch_models/consensus_layer.safetensors'

INPUT_PATH = 'input.nii.gz'
OUTPUT_PATH = 'output_pytorch.nii.gz'

def load_model(model_path, model_type='view', input_channels=1, coord_channels=3, output_channels=7):
    """
    Load a PyTorch model from a SafeTensors file.
    
    Args:
        model_path: Path to the SafeTensors file
        model_type: Type of model ('view' or 'consensus')
        
    Returns:
        Loaded PyTorch model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Create model
    if model_type == 'view':
        model = ViewSpecificModel(
            input_channels=input_channels,
            coord_channels=coord_channels,
            output_channels=output_channels
        )
    else:  # consensus
        model = ConsensusModel(
            input_channels=1 + 3 * output_channels,  # original image + 3 view predictions
            output_channels=output_channels
        )
    
    # Load weights
    weights = load_file(model_path)
    model.load_state_dict(weights, strict=False)
    
    return model

def segment_brain(input_path, output_path, sagittal_model_path=None, 
                 axial_model_path=None, coronal_model_path=None, 
                 consensus_model_path=None, segmentation_path=None, 
                 anterior_commissure=None, device='cuda'):
    """
    Segment a brain MRI using PyTorch models.
    
    Args:
        input_path: Path to the input NIfTI file
        output_path: Path to save the segmentation output
        sagittal_model_path, axial_model_path, coronal_model_path: Paths to view-specific models
        consensus_model_path: Path to the consensus model
        segmentation_path: Path to ground truth segmentation (optional)
        anterior_commissure: Coordinates of the anterior commissure (optional)
        device: Device to run inference on ('cuda' or 'cpu')
    """
    # Set device
    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif device == 'mps' or (device == 'cuda' and not torch.cuda.is_available() and torch.backends.mps.is_available()):
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Load models
    model_sagittal = None
    model_coronal = None
    model_axial = None
    consensus_model = None
    
    if sagittal_model_path and os.path.exists(sagittal_model_path):
        model_sagittal = load_model(sagittal_model_path, 'view')
        model_sagittal = model_sagittal.to(device)
        model_sagittal.eval()
        print(f"Loaded sagittal model from {sagittal_model_path}")
    
    if coronal_model_path and os.path.exists(coronal_model_path):
        model_coronal = load_model(coronal_model_path, 'view')
        model_coronal = model_coronal.to(device)
        model_coronal.eval()
        print(f"Loaded coronal model from {coronal_model_path}")
    
    if axial_model_path and os.path.exists(axial_model_path):
        model_axial = load_model(axial_model_path, 'view')
        model_axial = model_axial.to(device)
        model_axial.eval()
        print(f"Loaded axial model from {axial_model_path}")
    
    if consensus_model_path and os.path.exists(consensus_model_path):
        consensus_model = load_model(consensus_model_path, 'consensus')
        consensus_model = consensus_model.to(device)
        consensus_model.eval()
        print(f"Loaded consensus model from {consensus_model_path}")
    
    # Load input data
    nii = nib.load(input_path)
    if segmentation_path is not None:
        nii_seg = nib.load(segmentation_path)
    else:
        nii_seg = None
    
    print(f"Input shape: {nii.shape}")
    
    # Preprocess
    nii_out, nii_seg_out, coords, anterior_commissure, reconstruction_parms = preprocess_head_MRI(
        nii, nii_seg, anterior_commissure=anterior_commissure, keep_parameters_for_reconstruction=True
    )
    
    # Segment
    with torch.no_grad():
        segmentation = segment_MRI_pytorch(
            nii_out.get_fdata(),
            coords,
            model_sagittal=model_sagittal,
            model_coronal=model_coronal,
            model_axial=model_axial,
            consensus_model=consensus_model,
            device=device
        )
    
    # Save output
    nii_out_pred = nib.Nifti1Image(np.array(segmentation, dtype='int16'), nii_out.affine)
    nib.save(nii_out_pred, output_path)
    
    print(f"Segmentation saved to {output_path}")
    
    return segmentation

def main():
    parser = argparse.ArgumentParser(description='Brain segmentation using PyTorch models')
    parser.add_argument('--input', type=str, default=INPUT_PATH, help='Path to input NIfTI file')
    parser.add_argument('--output', type=str, default=OUTPUT_PATH, help='Path to save output segmentation')
    parser.add_argument('--sagittal', type=str, default=SAGITTAL_MODEL_PATH, help='Path to sagittal model')
    parser.add_argument('--coronal', type=str, default=CORONAL_MODEL_PATH, help='Path to coronal model')
    parser.add_argument('--axial', type=str, default=AXIAL_MODEL_PATH, help='Path to axial model')
    parser.add_argument('--consensus', type=str, default=CONSENSUS_LAYER_PATH, help='Path to consensus model')
    parser.add_argument('--segmentation', type=str, default=None, help='Path to ground truth segmentation')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'mps', 'cpu'], help='Device to run on')
    
    args = parser.parse_args()
    
    segment_brain(
        args.input,
        args.output,
        args.sagittal,
        args.axial,
        args.coronal,
        args.consensus,
        args.segmentation,
        device=args.device
    )

if __name__ == '__main__':
    main()