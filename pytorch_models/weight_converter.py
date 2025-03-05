"""
Tool for converting TensorFlow model weights to PyTorch format.
"""

import os
import argparse
import torch
import h5py
import numpy as np
from safetensors.torch import save_file

from .models import ViewSpecificModel, ConsensusModel

def extract_keras_weights(h5_path):
    """
    Extract weights from a Keras H5 model file.
    
    Args:
        h5_path: Path to the H5 model file
        
    Returns:
        Dictionary of layer names and their weights
    """
    weights = {}
    
    with h5py.File(h5_path, 'r') as f:
        model_weights = f['model_weights']
        
        for layer_name in model_weights:
            if 'conv' in layer_name.lower() or 'dense' in layer_name.lower() or 'batch_normalization' in layer_name.lower():
                layer = model_weights[layer_name]
                
                if 'kernel:0' in layer:
                    # Convert convolution kernel from TensorFlow format to PyTorch format
                    # TF: [filter_height, filter_width, filter_depth, in_channels, out_channels]
                    # PyTorch: [out_channels, in_channels, filter_depth, filter_height, filter_width]
                    kernel = np.array(layer['kernel:0'])
                    
                    if len(kernel.shape) == 5:  # 3D convolution
                        kernel = np.transpose(kernel, (4, 3, 0, 1, 2))
                    elif len(kernel.shape) == 4:  # 2D convolution
                        kernel = np.transpose(kernel, (3, 2, 0, 1))
                        
                    weights[f"{layer_name}.weight"] = torch.tensor(kernel)
                
                if 'bias:0' in layer:
                    bias = np.array(layer['bias:0'])
                    weights[f"{layer_name}.bias"] = torch.tensor(bias)
                
                # BatchNormalization parameters
                if 'gamma:0' in layer:
                    gamma = np.array(layer['gamma:0'])
                    weights[f"{layer_name}.weight"] = torch.tensor(gamma)
                
                if 'beta:0' in layer:
                    beta = np.array(layer['beta:0'])
                    weights[f"{layer_name}.bias"] = torch.tensor(beta)
                
                if 'moving_mean:0' in layer:
                    mean = np.array(layer['moving_mean:0'])
                    weights[f"{layer_name}.running_mean"] = torch.tensor(mean)
                
                if 'moving_variance:0' in layer:
                    var = np.array(layer['moving_variance:0'])
                    weights[f"{layer_name}.running_var"] = torch.tensor(var)
    
    return weights

def map_weights_to_model(weights, model, model_type='view'):
    """
    Map extracted weights to a PyTorch model.
    This is a simplified implementation and would need to be adjusted 
    based on the actual model architecture.
    
    Args:
        weights: Dictionary of layer names and weights
        model: PyTorch model instance
        model_type: Type of model ('view' or 'consensus')
        
    Returns:
        Model with mapped weights
    """
    state_dict = model.state_dict()
    mapped_weights = {}
    
    if model_type == 'view':
        # Map weights for view-specific model
        # This is highly simplified and would need to be adjusted based on actual architecture
        weight_mapping = {
            # Image branch
            'conv1_img.weight': 'conv1_img/kernel:0',
            'conv1_img.bias': 'conv1_img/bias:0',
            'conv2_img.weight': 'conv2_img/kernel:0',
            'conv2_img.bias': 'conv2_img/bias:0',
            'batch_norm_img.weight': 'batch_norm_img/gamma:0',
            'batch_norm_img.bias': 'batch_norm_img/beta:0',
            'batch_norm_img.running_mean': 'batch_norm_img/moving_mean:0',
            'batch_norm_img.running_var': 'batch_norm_img/moving_variance:0',
            
            # Coordinate branch
            'conv1_coord.weight': 'conv1_coord/kernel:0',
            'conv1_coord.bias': 'conv1_coord/bias:0',
            'conv2_coord.weight': 'conv2_coord/kernel:0',
            'conv2_coord.bias': 'conv2_coord/bias:0',
            'batch_norm_coord.weight': 'batch_norm_coord/gamma:0',
            'batch_norm_coord.bias': 'batch_norm_coord/beta:0',
            'batch_norm_coord.running_mean': 'batch_norm_coord/moving_mean:0',
            'batch_norm_coord.running_var': 'batch_norm_coord/moving_variance:0',
            
            # Combined branch
            'conv_combined1.weight': 'conv_combined1/kernel:0',
            'conv_combined1.bias': 'conv_combined1/bias:0',
            'conv_combined2.weight': 'conv_combined2/kernel:0',
            'conv_combined2.bias': 'conv_combined2/bias:0',
            'batch_norm_combined.weight': 'batch_norm_combined/gamma:0',
            'batch_norm_combined.bias': 'batch_norm_combined/beta:0',
            'batch_norm_combined.running_mean': 'batch_norm_combined/moving_mean:0',
            'batch_norm_combined.running_var': 'batch_norm_combined/moving_variance:0',
            
            # Output layer
            'conv_output.weight': 'conv_output/kernel:0',
            'conv_output.bias': 'conv_output/bias:0',
        }
    else:  # consensus model
        # Map weights for consensus model
        weight_mapping = {
            # Consensus layers
            'conv1.weight': 'conv1/kernel:0',
            'conv1.bias': 'conv1/bias:0',
            'batch_norm1.weight': 'batch_norm1/gamma:0',
            'batch_norm1.bias': 'batch_norm1/beta:0',
            'batch_norm1.running_mean': 'batch_norm1/moving_mean:0',
            'batch_norm1.running_var': 'batch_norm1/moving_variance:0',
            
            'conv2.weight': 'conv2/kernel:0',
            'conv2.bias': 'conv2/bias:0',
            'batch_norm2.weight': 'batch_norm2/gamma:0',
            'batch_norm2.bias': 'batch_norm2/beta:0',
            'batch_norm2.running_mean': 'batch_norm2/moving_mean:0',
            'batch_norm2.running_var': 'batch_norm2/moving_variance:0',
            
            'conv3.weight': 'conv3/kernel:0',
            'conv3.bias': 'conv3/bias:0',
            'batch_norm3.weight': 'batch_norm3/gamma:0',
            'batch_norm3.bias': 'batch_norm3/beta:0',
            'batch_norm3.running_mean': 'batch_norm3/moving_mean:0',
            'batch_norm3.running_var': 'batch_norm3/moving_variance:0',
            
            # Output layer
            'conv_output.weight': 'conv_output/kernel:0',
            'conv_output.bias': 'conv_output/bias:0',
        }
    
    # Map weights
    for pytorch_key, keras_key in weight_mapping.items():
        if keras_key in weights:
            # Handle shape mismatches with warning
            if pytorch_key in state_dict and weights[keras_key].shape != state_dict[pytorch_key].shape:
                print(f"Warning: Shape mismatch for {pytorch_key}. "
                      f"PyTorch: {state_dict[pytorch_key].shape}, "
                      f"Keras: {weights[keras_key].shape}")
                continue
                
            mapped_weights[pytorch_key] = weights[keras_key]
    
    # Load weights into model
    model.load_state_dict(mapped_weights, strict=False)
    
    return model

def convert_model(h5_path, output_path, model_type='view', input_channels=1, coord_channels=3, output_channels=7):
    """
    Convert a Keras H5 model to PyTorch SafeTensors format.
    
    Args:
        h5_path: Path to the Keras H5 model file
        output_path: Path to save the SafeTensors file
        model_type: Type of model ('view' or 'consensus')
        input_channels: Number of input channels
        coord_channels: Number of coordinate channels
        output_channels: Number of output channels
    """
    # Create appropriate model based on type
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
    
    # Extract weights from Keras model
    weights = extract_keras_weights(h5_path)
    
    # Map weights to PyTorch model
    model = map_weights_to_model(weights, model, model_type)
    
    # Save model to SafeTensors format
    save_file(model.state_dict(), output_path)
    
    print(f"Successfully converted {h5_path} to {output_path}")
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Convert Keras H5 models to PyTorch SafeTensors format')
    parser.add_argument('--input', type=str, required=True, help='Path to Keras H5 model file')
    parser.add_argument('--output', type=str, required=True, help='Path to save SafeTensors file')
    parser.add_argument('--type', type=str, choices=['view', 'consensus'], default='view', 
                        help='Type of model to convert')
    
    args = parser.parse_args()
    
    convert_model(args.input, args.output, args.type)

if __name__ == '__main__':
    main()