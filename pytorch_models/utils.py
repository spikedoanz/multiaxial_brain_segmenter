"""
Utility functions for PyTorch model conversion and inference.
"""

import os
import numpy as np
import torch
import h5py
import nibabel as nib
from safetensors.torch import save_file
from scipy.ndimage import map_coordinates

def create_coordinate_matrix(shape, anterior_commissure):
    """
    Create a coordinate matrix relative to the anterior commissure.
    
    Args:
        shape: The shape of the output coordinate matrix (x, y, z)
        anterior_commissure: The coordinates of the anterior commissure
        
    Returns:
        A coordinate matrix with shape [x, y, z, 3]
    """
    x, y, z = shape
    meshgrid = np.meshgrid(
        np.linspace(0, x - 1, x), 
        np.linspace(0, y - 1, y), 
        np.linspace(0, z - 1, z), 
        indexing='ij'
    )
    coordinates = np.stack(meshgrid, axis=-1) - np.array(anterior_commissure)
    
    # Normalize coordinates
    coordinates = coordinates / 256.0
    
    return coordinates[..., :3]  # Return only x, y, z coordinates

def convert_h5_to_safetensors(h5_path, output_path, model_type='view'):
    """
    Convert H5 model weights to SafeTensors format.
    This is a simplified implementation since we don't know the exact architecture.
    
    Args:
        h5_path: Path to H5 model file
        output_path: Path to save SafeTensors file
        model_type: Type of model ('view' or 'consensus')
    """
    try:
        # Open the H5 file
        with h5py.File(h5_path, 'r') as h5_file:
            # Initialize state dict for PyTorch model
            state_dict = {}
            
            # Extract weights from H5 file (simplified example)
            # This mapping needs to be adjusted based on the actual model architecture
            for name, layer in h5_file.items():
                if 'weight' in name or 'bias' in name:
                    weight = np.array(layer)
                    
                    # Convert from NHWC to NCHW format if needed (for Conv layers)
                    if len(weight.shape) == 5:
                        # Assuming TensorFlow format: [filter_height, filter_width, filter_depth, in_channels, out_channels]
                        # Convert to PyTorch format: [out_channels, in_channels, filter_depth, filter_height, filter_width]
                        weight = np.transpose(weight, (4, 3, 0, 1, 2))
                    
                    # Convert to PyTorch tensor
                    state_dict[name] = torch.tensor(weight)
            
            # Save as SafeTensors
            save_file(state_dict, output_path)
            
            print(f"Successfully converted {h5_path} to {output_path}")
            
    except Exception as e:
        print(f"Error converting H5 to SafeTensors: {e}")

def swapaxes_3d(data, axis1, axis2):
    """
    Swap axes of 3D data with batch and channel dimensions.
    
    Args:
        data: Input tensor with shape [batch, channels, d1, d2, d3]
        axis1, axis2: Axes to swap (2, 3, or 4 for the spatial dimensions)
        
    Returns:
        Tensor with swapped axes
    """
    if axis1 == 2 and axis2 == 3:
        # Swap d1 and d2
        return data.permute(0, 1, 3, 2, 4)
    elif axis1 == 2 and axis2 == 4:
        # Swap d1 and d3
        return data.permute(0, 1, 4, 3, 2)
    elif axis1 == 3 and axis2 == 4:
        # Swap d2 and d3
        return data.permute(0, 1, 2, 4, 3)
    else:
        raise ValueError(f"Invalid axes: {axis1}, {axis2}. Must be 2, 3, or 4.")

def segment_MRI_pytorch(img, coords, model_sagittal=None, model_coronal=None, model_axial=None, consensus_model=None, device=None):
    """
    Segment an MRI using PyTorch models.
    
    Args:
        img: Input image as a NumPy array with shape [256, 256, 256]
        coords: Coordinate matrix as a NumPy array with shape [256, 256, 256, 3]
        model_sagittal, model_coronal, model_axial: View-specific segmentation models
        consensus_model: Consensus model
        device: Device to run inference on ('cuda', 'mps', or 'cpu')
        
    Returns:
        Segmentation as a NumPy array with shape [256, 256, 256]
    """
    # Set device if not provided
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    # Move models to device
    if model_sagittal is not None:
        model_sagittal = model_sagittal.to(device)
        model_sagittal.eval()
    
    if model_coronal is not None:
        model_coronal = model_coronal.to(device)
        model_coronal.eval()
    
    if model_axial is not None:
        model_axial = model_axial.to(device)
        model_axial.eval()
    
    if consensus_model is not None:
        consensus_model = consensus_model.to(device)
        consensus_model.eval()
    
    with torch.no_grad():
        # Convert to PyTorch tensors
        img_tensor = torch.from_numpy(img).float().to(device)
        coords_tensor = torch.from_numpy(coords).float().to(device)
        
        # Add batch and channel dimensions
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 256, 256, 256]
        coords_tensor = coords_tensor.permute(3, 0, 1, 2).unsqueeze(0)  # [1, 3, 256, 256, 256]
        
        model_segmentation_sagittal = None
        model_segmentation_coronal = None
        model_segmentation_axial = None
        
        # Sagittal view
        if model_sagittal is not None:
            # No need to swap axes for sagittal view
            yhat_sagittal = model_sagittal(img_tensor, coords_tensor)
            model_segmentation_sagittal = yhat_sagittal  # [1, 7, 256, 256, 256]
            print("Sagittal segmentation complete")
        
        # Coronal view
        if model_coronal is not None:
            # Swap axes 0 and 1 (sagittal to coronal)
            img_coronal = swapaxes_3d(img_tensor, 2, 3)
            coords_coronal = swapaxes_3d(coords_tensor, 2, 3)
            
            yhat_coronal = model_coronal(img_coronal, coords_coronal)
            # Swap back
            model_segmentation_coronal = swapaxes_3d(yhat_coronal, 2, 3)
            print("Coronal segmentation complete")
        
        # Axial view
        if model_axial is not None:
            # Swap axes to get axial view
            img_axial = swapaxes_3d(swapaxes_3d(img_tensor, 3, 4), 2, 3)
            coords_axial = swapaxes_3d(swapaxes_3d(coords_tensor, 3, 4), 2, 3)
            
            yhat_axial = model_axial(img_axial, coords_axial)
            # Swap back
            model_segmentation_axial = swapaxes_3d(swapaxes_3d(yhat_axial, 2, 3), 3, 4)
            print("Axial segmentation complete")
        
        # Consensus model
        if consensus_model is not None and all([model_segmentation_sagittal, model_segmentation_coronal, model_segmentation_axial]):
            # Concatenate all predictions
            X = torch.cat([
                img_tensor,
                model_segmentation_sagittal,
                model_segmentation_coronal,
                model_segmentation_axial
            ], dim=1)  # [1, 22, 256, 256, 256]
            
            print("Getting model consensus")
            yhat = consensus_model(X)  # [1, 7, 256, 256, 256]
            
            # Convert to numpy and take argmax
            pred = torch.argmax(yhat, dim=1).cpu().numpy()[0]  # [256, 256, 256]
            print(f"Consensus model output shape: {pred.shape}")
        else:
            # Use one of the individual predictions if consensus model is not available
            if model_segmentation_sagittal is not None:
                pred = torch.argmax(model_segmentation_sagittal, dim=1).cpu().numpy()[0]
            elif model_segmentation_coronal is not None:
                pred = torch.argmax(model_segmentation_coronal, dim=1).cpu().numpy()[0]
            elif model_segmentation_axial is not None:
                pred = torch.argmax(model_segmentation_axial, dim=1).cpu().numpy()[0]
            else:
                raise ValueError("No models provided for segmentation")
        
        return pred