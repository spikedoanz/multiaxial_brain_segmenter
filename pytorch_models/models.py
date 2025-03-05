"""
PyTorch implementation of brain segmentation models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViewSpecificModel(nn.Module):
    """
    PyTorch implementation of view-specific segmentation model.
    This is a simplified model structure since we don't know the exact 
    architecture of the original models.
    """
    def __init__(self, input_channels=1, coord_channels=3, output_channels=7, kernel_size=3):
        super(ViewSpecificModel, self).__init__()
        
        # Process the image branch
        self.conv1_img = nn.Conv3d(input_channels, 16, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2_img = nn.Conv3d(16, 32, kernel_size=kernel_size, padding=kernel_size//2)
        self.batch_norm_img = nn.BatchNorm3d(32)
        
        # Process the coordinates branch
        self.conv1_coord = nn.Conv3d(coord_channels, 16, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2_coord = nn.Conv3d(16, 32, kernel_size=kernel_size, padding=kernel_size//2)
        self.batch_norm_coord = nn.BatchNorm3d(32)
        
        # Combined processing
        self.conv_combined1 = nn.Conv3d(64, 64, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv_combined2 = nn.Conv3d(64, 32, kernel_size=kernel_size, padding=kernel_size//2)
        self.batch_norm_combined = nn.BatchNorm3d(32)
        
        # Output layer
        self.conv_output = nn.Conv3d(32, output_channels, kernel_size=1)
        
    def forward(self, image, coords):
        # Process image branch
        x_img = F.relu(self.conv1_img(image))
        x_img = F.relu(self.batch_norm_img(self.conv2_img(x_img)))
        
        # Process coordinates branch
        x_coord = F.relu(self.conv1_coord(coords))
        x_coord = F.relu(self.batch_norm_coord(self.conv2_coord(x_coord)))
        
        # Combine branches
        x_combined = torch.cat([x_img, x_coord], dim=1)
        x_combined = F.relu(self.conv_combined1(x_combined))
        x_combined = F.relu(self.batch_norm_combined(self.conv_combined2(x_combined)))
        
        # Output segmentation
        output = self.conv_output(x_combined)
        
        # Apply softmax along the channel dimension
        output = F.softmax(output, dim=1)
        
        return output


class ConsensusModel(nn.Module):
    """
    PyTorch implementation of consensus model that combines predictions 
    from three view-specific models.
    """
    def __init__(self, input_channels=22, output_channels=7, kernel_size=3):
        super(ConsensusModel, self).__init__()
        
        # Input: original image (1) + sagittal predictions (7) + coronal predictions (7) + axial predictions (7)
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=kernel_size, padding=kernel_size//2)
        self.batch_norm1 = nn.BatchNorm3d(32)
        
        self.conv2 = nn.Conv3d(32, 32, kernel_size=kernel_size, padding=kernel_size//2)
        self.batch_norm2 = nn.BatchNorm3d(32)
        
        self.conv3 = nn.Conv3d(32, 16, kernel_size=kernel_size, padding=kernel_size//2)
        self.batch_norm3 = nn.BatchNorm3d(16)
        
        # Output layer
        self.conv_output = nn.Conv3d(16, output_channels, kernel_size=1)
        
    def forward(self, x):
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = F.relu(self.batch_norm3(self.conv3(x)))
        
        # Output segmentation
        output = self.conv_output(x)
        
        # Apply softmax along the channel dimension
        output = F.softmax(output, dim=1)
        
        return output