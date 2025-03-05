# PyTorch Brain Segmentation Models

This directory contains PyTorch implementations of the brain segmentation models originally written in TensorFlow/Keras.

## Overview

The PyTorch implementation includes:

1. **Model Architecture**:
   - View-specific models (sagittal, coronal, axial) for multi-view brain segmentation
   - Consensus model to combine the predictions from all views
   - Models use 3D convolutions with coordinate-aware inputs

2. **Loss Functions**:
   - Dice coefficient metrics for 7-class segmentation
   - Class-specific Dice coefficients

3. **Utilities**:
   - TensorFlow to PyTorch model conversion
   - SafeTensors format for model weights
   - Coordinate matrix generation
   - Axis swapping for different views

## Usage

### Converting TensorFlow Models to PyTorch

```bash
python convert_models.py --output_dir pytorch_models
```

This will convert all four TensorFlow models (`sagittal_model.h5`, `coronal_model.h5`, `axial_model.h5`, and `consensus_layer.h5`) to PyTorch models in SafeTensors format.

### Running Inference

```bash
python pytorch_evaluate_model_on_scan.py --input input.nii.gz --output output_pytorch.nii.gz
```

### Testing Models

For testing with small input sizes:

```bash
python test_pytorch_models.py
```

For testing with minimal kernel sizes:

```bash
python test_small_kernels.py
```

## Architecture Details

### View-Specific Models

- Input: 3D brain scan + coordinate information
- Dual-branch architecture processing image and coordinates separately
- Merged branches for combined processing
- Output: 7-class segmentation predictions

### Consensus Model

- Input: Original scan + predictions from all three view-specific models
- Several convolutional layers to combine and refine predictions
- Output: Final 7-class segmentation prediction

## Notes

- The PyTorch implementation preserves the original model's architecture while adapting to PyTorch's conventions
- The coordinate system is maintained for spatial awareness
- The axis swapping for different views is handled within the segmentation function
- SafeTensors format provides improved security and loading speed