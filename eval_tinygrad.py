#!/usr/bin/env python3
import os
import sys
import nibabel as nib
import numpy as np
from tqdm import tqdm
from skimage.transform import resize
from nibabel.orientations import axcodes2ornt, ornt_transform
from tinygrad.tensor import Tensor
from tinyonnx import OnnxRunner

# ---------------------------
# Image Orientation Functions
# ---------------------------
def reorient(nii, orientation) -> nib.Nifti1Image:
    """Reorients a nifti image to specified orientation."""
    orig_ornt = nib.io_orientation(nii.affine)
    targ_ornt = axcodes2ornt(orientation)
    transform = ornt_transform(orig_ornt, targ_ornt)
    reoriented_nii = nii.as_reoriented(transform)
    return reoriented_nii

def create_coordinate_matrix(shape, anterior_commissure):
    """Creates a coordinate matrix based on the image shape and anterior commissure."""
    x, y, z = shape
    meshgrid = np.meshgrid(np.linspace(0, x - 1, x), np.linspace(0, y - 1, y), np.linspace(0, z - 1, z), indexing='ij')
    coordinates = np.stack(meshgrid, axis=-1) - np.array(anterior_commissure)
    matrix_with_ones = np.concatenate([coordinates, np.ones((coordinates.shape[0], coordinates.shape[1], coordinates.shape[2], 1))], axis=-1)
    return matrix_with_ones

# ----------------------
# Preprocessing Functions
# ----------------------
def preprocess_head_MRI(nii: nib.Nifti1Image, nii_seg: nib.Nifti1Image = None, anterior_commissure: tuple = None, keep_parameters_for_reconstruction: bool = False):
    """Preprocesses a head MRI image."""    
    if nii_seg is not None:
        assert nii.shape == nii_seg.shape
    if anterior_commissure is None:
        print('No anterior commissure location given.. centering to center of image..')
        anterior_commissure = [nii.shape[0]//2, nii.shape[1]//2, nii.shape[2]//2]
    else:
        print(f'anterior commissure given: {anterior_commissure}')
    orientation = nib.aff2axcodes(nii.affine)
    
    if ''.join(orientation) != 'RAS':
        print(f'Image orientation : {orientation}. Changing to RAS..')
        nii = reorient(nii, "RAS")
        if nii_seg is not None:
            nii_seg = reorient(nii_seg, "RAS")
   
    if nii_seg is not None:
        img_seg = nii_seg.get_fdata()
    else:
        print('Segmentation image not given')
                
    # Make image isotropic
    res = nii.header['pixdim'][1:4]
    img = nii.get_fdata()
    new_shape = np.array(np.array(nii.shape)*res, dtype='int')
    if np.any(np.array(nii.shape) != new_shape):
        img = resize(img, new_shape, anti_aliasing=True, preserve_range=True)
        if nii_seg is not None: 
            img_seg = resize(img_seg, new_shape, order=0, anti_aliasing=True, preserve_range=True)
       
    nii.affine[0][0] = 1.
    nii.affine[1][1] = 1.
    nii.affine[2][2] = 1.
    nii.header['pixdim'][1:4] = np.diag(nii.affine)[0:3]
    
    # Crop/Pad to make shape 256,256,256
    d1, d2, d3 = new_shape
    start = None
    end = None    
    
    if d1 < 256:
        pad1 = 256-d1
        img = np.pad(img, ((pad1//2, pad1//2+pad1%2),(0,0),(0,0)))
        if nii_seg is not None: 
            img_seg = np.pad(img_seg, ((pad1//2, pad1//2+pad1%2),(0,0),(0,0)))
        anterior_commissure[0] += pad1//2
    
    if d2 > 256: 
        crop2 = d2-256
        img = img[:,crop2//2:-(crop2//2+crop2%2)]
        if nii_seg is not None: 
            img_seg = img_seg[:,crop2//2:-(crop2//2+crop2%2)]
        anterior_commissure[1] -= crop2//2
            
    elif d2 < 256:
        pad2 = 256-d2
        img = np.pad(img, ((0,0),(pad2//2, pad2//2+pad2%2),(0,0)))
        if nii_seg is not None: 
            img_seg = np.pad(img_seg, ((0,0),(pad2//2, pad2//2+pad2%2),(0,0)))
        anterior_commissure[1] += pad2//2
        
    if d3 > 256: 
        # Head start
        proj = np.max(img,(0,1))
        proj[proj < np.percentile(proj, 50) ] = 0
        proj[proj > 0] = 1
        end = np.max(np.argwhere(proj == 1))
        end = np.min([end + 20, d3]) # Leave some space above the head
        start = end-256
        if start < 0:
            crop3 = d3 - 256
            img = img[:,:,crop3:]
            if nii_seg is not None: 
                img_seg = img_seg[:,:,crop3:]
            anterior_commissure[2] -= crop3
         
        else:
            img = img[:,:,start:end]
            if nii_seg is not None: 
                img_seg = img_seg[:,:,start:end]
            anterior_commissure[2] -= start
    elif d3 < 256:
        pad3 = 256-d3
        img = np.pad(img, ((0,0),(0,0),(pad3//2, pad3//2+pad3%2)))
        if nii_seg is not None: 
            img_seg = np.pad(img_seg, ((0,0),(0,0),(pad3//2, pad3//2+pad3%2)))
        anterior_commissure[2] += pad3//2
    
    coords = create_coordinate_matrix(img.shape, anterior_commissure)        
    
    # Intensity normalization
    p95 = np.percentile(img, 95)
    img = img/p95
    
    coords = coords[:,:,:,:3]
    coords = coords/256.
    
    result = [nib.Nifti1Image(img, nii.affine)]
    
    if nii_seg is not None:
        result.append(nib.Nifti1Image(np.array(img_seg, dtype='int8'), nii.affine))
    else:
        result.append(None)
        
    result.extend([np.array(coords, dtype='float32'), np.array(anterior_commissure, dtype='int')])
    
    if keep_parameters_for_reconstruction:
        reconstruction_parms = d1, d2, d3, start, end
        result.append(reconstruction_parms)
    
    return tuple(result)
        
def reshape_back_to_original(img, nii_original, reconstruction_parms, resample_order=1):
    """Reshape a processed image back to its original dimensions."""
    d1, d2, d3, start, end = reconstruction_parms
    
    # Pad or crop z axis
    if d3 > 256: 
        if start < 0:
            crop3 = d3 - 256
            img = np.pad(img, ((0,0),(0,0),(crop3,0)))
        else:
            img = np.pad(img, ((0,0),(0,0),(start,end)))          
    elif d3 < 256:
        pad3 = 256-d3
        img = img[:,:,pad3//2:-(pad3//2+pad3%2)]
    
    # Pad or crop y axis
    if d2 > 256: 
        crop2 = d2-256
        img = np.pad(img, ((0,0),(crop2//2,crop2//2+crop2%2),(0,0)))
    elif d2 < 256:
        pad2 = 256-d2
        img = img[:,pad2//2:-(pad2//2+pad2%2),:]
        
    # Pad or crop x axis
    if d1 < 256:
        pad1 = 256-d1
        img = img[pad1//2:-(pad1//2+pad1%2),:,:]
    
    # Resample to original resolution
    img = resize(img, nii_original.shape, anti_aliasing=True, preserve_range=True, order=resample_order)
    
    nii_out = nib.Nifti1Image(img, nii_original.affine)
    return nii_out

# ------------------
# ONNX Model Inspection
# ------------------
def inspect_onnx_model(model_path):
    """
    Load and inspect an ONNX model to understand its input and output structure.
    """
    print(f"Inspecting ONNX model: {model_path}")
    
    try:
        import onnx
        # Load the model
        model = onnx.load(model_path)
        
        # Check model metadata
        print(f"Model IR version: {model.ir_version}")
        print(f"Producer name: {model.producer_name}")
        print(f"Producer version: {model.producer_version}")
        print(f"Domain: {model.domain}")
        print(f"Model version: {model.model_version}")
        
        # Analyze inputs
        print("\nInputs:")
        for i, input_info in enumerate(model.graph.input):
            print(f"  Input #{i}: {input_info.name}")
            
            # Get input shape
            shape = []
            for dim in input_info.type.tensor_type.shape.dim:
                if dim.dim_param:
                    shape.append(dim.dim_param)  # Dynamic dimension
                else:
                    shape.append(dim.dim_value)  # Static dimension
            
            print(f"    Shape: {shape}")
            
            # Get data type
            dtype = input_info.type.tensor_type.elem_type
            print(f"    Data Type: {dtype}")
        
        # Analyze outputs
        print("\nOutputs:")
        for i, output_info in enumerate(model.graph.output):
            print(f"  Output #{i}: {output_info.name}")
            
            # Get output shape
            shape = []
            for dim in output_info.type.tensor_type.shape.dim:
                if dim.dim_param:
                    shape.append(dim.dim_param)  # Dynamic dimension
                else:
                    shape.append(dim.dim_value)  # Static dimension
            
            print(f"    Shape: {shape}")
            
            # Get data type
            dtype = output_info.type.tensor_type.elem_type
            print(f"    Data Type: {dtype}")
        
        # Count the number of nodes
        nodes_count = len(model.graph.node)
        print(f"\nTotal nodes in the model: {nodes_count}")
        
        # List of node types (operators)
        op_types = set(node.op_type for node in model.graph.node)
        print(f"Operator types used: {', '.join(sorted(op_types))}")
    except ImportError:
        print("ONNX library not found. Install with 'pip install onnx' to use this function.")
    except Exception as e:
        print(f"Error inspecting model: {e}")

# ------------------
# TinyGrad ONNX Segmentation Logic
# ------------------
def process_sagittal_slices(runner, img, coords, input_names=None):
    """Process all sagittal slices through the model and return the predictions."""
    # Initialize output array (7 classes per slice prediction)
    output = np.zeros((img.shape[0], img.shape[1], img.shape[2], 7), dtype=np.float32)
    
    print(f"Processing {img.shape[0]} sagittal slices...")
    
    # If input_names not provided, attempt to detect from the first run
    if input_names is None:
        input_names = {"img": "input_1", "coords": "input_2"}
        print(f"Using default input names: {input_names}")
    
    img_input_name = input_names["img"]
    coords_input_name = input_names["coords"]
    
    # Process each slice
    for i in tqdm(range(img.shape[0])):
        # Extract 2D slice
        img_slice = img[i, :, :]
        coords_slice = coords[i, :, :, :]
        
        # Prepare inputs with correct shapes
        img_input = np.expand_dims(np.expand_dims(img_slice, -1), 0).astype(np.float32)
        coords_input = np.expand_dims(coords_slice, 0).astype(np.float32)
        
        # Print shapes for the first slice to verify
        if i == 0:
            print(f"  Input image shape: {img_input.shape}")
            print(f"  Input coords shape: {coords_input.shape}")
        
        # Create TinyGrad tensors
        img_tensor = Tensor(img_input, requires_grad=False)
        coords_tensor = Tensor(coords_input, requires_grad=False)
        
        # Run inference using TinyGrad's OnnxRunner with correct input names
        outputs = runner({
            img_input_name: img_tensor, 
            coords_input_name: coords_tensor
        })
        
        # Get output tensor (assuming the output has a standard name like 'output')
        # Extract the first key if only one output is present
        output_tensor = list(outputs.values())[0]
        
        # Convert to numpy and store prediction (remove batch dimension)
        output[i, :, :, :] = output_tensor.numpy()[0]
    
    return output

def process_coronal_slices(runner, img, coords, input_names=None):
    """Process all coronal slices through the model and return the predictions."""
    # Initialize output array (7 classes per slice prediction)
    output = np.zeros((img.shape[0], img.shape[1], img.shape[2], 7), dtype=np.float32)
    
    print(f"Processing {img.shape[1]} coronal slices...")
    
    # If input_names not provided, attempt to detect from the first run
    if input_names is None:
        input_names = {"img": "input_1", "coords": "input_2"}
        print(f"Using default input names: {input_names}")
    
    img_input_name = input_names["img"]
    coords_input_name = input_names["coords"]
    
    # Process each slice
    for i in tqdm(range(img.shape[1])):
        # Extract 2D slice
        img_slice = img[:, i, :]
        coords_slice = coords[:, i, :, :]
        
        # Prepare inputs with correct shapes
        img_input = np.expand_dims(np.expand_dims(img_slice, -1), 0).astype(np.float32)
        coords_input = np.expand_dims(coords_slice, 0).astype(np.float32)
        
        # Print shapes for the first slice to verify
        if i == 0:
            print(f"  Input image shape: {img_input.shape}")
            print(f"  Input coords shape: {coords_input.shape}")
        
        # Create TinyGrad tensors
        img_tensor = Tensor(img_input, requires_grad=False)
        coords_tensor = Tensor(coords_input, requires_grad=False)
        
        # Run inference using TinyGrad's OnnxRunner with correct input names
        outputs = runner({
            img_input_name: img_tensor, 
            coords_input_name: coords_tensor
        })
        
        # Get output tensor (assuming the output has a standard name like 'output')
        # Extract the first key if only one output is present
        output_tensor = list(outputs.values())[0]
        
        # Convert to numpy and store prediction (remove batch dimension)
        output[:, i, :, :] = output_tensor.numpy()[0]
    
    return output

def process_axial_slices(runner, img, coords, input_names=None):
    """Process all axial slices through the model and return the predictions."""
    # Initialize output array (7 classes per slice prediction)
    output = np.zeros((img.shape[0], img.shape[1], img.shape[2], 7), dtype=np.float32)
    
    print(f"Processing {img.shape[1]} coronal slices...")
    
    # If input_names not provided, attempt to detect from the first run
    if input_names is None:
        input_names = {"img": "input_1", "coords": "input_2"}
        print(f"Using default input names: {input_names}")
    
    img_input_name = input_names["img"]
    coords_input_name = input_names["coords"]
    print(f"Processing {img.shape[2]} axial slices...")
    
    # Process each slice
    for i in tqdm(range(img.shape[2])):
        # Extract 2D slice
        img_slice = img[:, :, i]
        coords_slice = coords[:, :, i, :]
        
        # Prepare inputs with correct shapes
        img_input = np.expand_dims(np.expand_dims(img_slice, -1), 0).astype(np.float32)
        coords_input = np.expand_dims(coords_slice, 0).astype(np.float32)
        
        # Print shapes for the first slice to verify
        if i == 0:
            print(f"  Input image shape: {img_input.shape}")
            print(f"  Input coords shape: {coords_input.shape}")
        
        # Create TinyGrad tensors
        img_tensor = Tensor(img_input, requires_grad=False)
        coords_tensor = Tensor(coords_input, requires_grad=False)
        
        # Run inference using TinyGrad's OnnxRunner with correct input names
        outputs = runner({
            img_input_name: img_tensor, 
            coords_input_name: coords_tensor
        })
        
        # Get output tensor (assuming the output has a standard name like 'output')
        # Extract the first key if only one output is present
        output_tensor = list(outputs.values())[0]
        
        # Convert to numpy and store prediction (remove batch dimension)
        output[:, :, i, :] = output_tensor.numpy()[0]
    
    return output

def segment_MRI_tinygrad(img, coords, model_sagittal_path=None, model_axial_path=None, model_coronal_path=None, consensus_model_path=None):
    # Import necessary models
    import onnx
    
    model_segmentation_sagittal = None
    model_segmentation_coronal = None
    model_segmentation_axial = None
    
    # Debug helper function
    def print_model_inputs(model_path, model_name):
        """Print information about model inputs to help with debugging"""
        model = onnx.load(model_path)
        print(f"\n{model_name} Input Details:")
        input_names = {}
        for i, input_info in enumerate(model.graph.input):
            print(f"  Input #{i}")
            print(f"    Name: {input_info.name}")
            # Get shape
            shape = []
            for dim in input_info.type.tensor_type.shape.dim:
                if dim.dim_param:
                    shape.append(dim.dim_param)
                else:
                    shape.append(dim.dim_value)
            print(f"    Shape: {shape}")
            print(f"    Type: {input_info.type.tensor_type.elem_type}")
            
            # Store input names
            if i == 0:
                input_names["img"] = input_info.name
            elif i == 1:
                input_names["coords"] = input_info.name
                
        print(f"  Detected input names: {input_names}")
        print("")
        return input_names

    # Process slice-by-slice for each model
    if model_sagittal_path is not None:
        print("Running sagittal model inference...")
        sagittal_model = onnx.load(model_sagittal_path)
        sagittal_runner = OnnxRunner(sagittal_model)
        sagittal_input_names = print_model_inputs(model_sagittal_path, "Sagittal Model")
        model_segmentation_sagittal = process_sagittal_slices(
            sagittal_runner, img, coords, input_names=sagittal_input_names
        )
    
    if model_coronal_path is not None:
        print("Running coronal model inference...")
        coronal_model = onnx.load(model_coronal_path)
        coronal_runner = OnnxRunner(coronal_model)
        coronal_input_names = print_model_inputs(model_coronal_path, "Coronal Model")
        model_segmentation_coronal = process_coronal_slices(
            coronal_runner, img, coords, input_names=coronal_input_names
        )
    
    if model_axial_path is not None:
        print("Running axial model inference...")
        axial_model = onnx.load(model_axial_path)
        axial_runner = OnnxRunner(axial_model)
        axial_input_names = print_model_inputs(model_axial_path, "Axial Model")
        model_segmentation_axial = process_axial_slices(
            axial_runner, img, coords, input_names=axial_input_names
        )
    
    # Create empty outputs for any models that didn't run
    if model_segmentation_sagittal is None:
        model_segmentation_sagittal = np.zeros((img.shape[0], img.shape[1], img.shape[2], 7), dtype=np.float32)
    if model_segmentation_coronal is None:
        model_segmentation_coronal = np.zeros((img.shape[0], img.shape[1], img.shape[2], 7), dtype=np.float32)
    if model_segmentation_axial is None:
        model_segmentation_axial = np.zeros((img.shape[0], img.shape[1], img.shape[2], 7), dtype=np.float32)
    
    # Run consensus model
    print("Running consensus model...")
    consensus_model = onnx.load(consensus_model_path)
    consensus_runner = OnnxRunner(consensus_model)
    consensus_input_names = print_model_inputs(consensus_model_path, "Consensus Model")
    
    # Get the input name for the consensus model
    consensus_input_name = list(consensus_input_names.values())[0] if consensus_input_names else "input_1"
    print(f"Using consensus input name: {consensus_input_name}")
    
    # Prepare input for consensus model
    img_expanded = np.expand_dims(img, -1).astype(np.float32)
    
    # Concatenate along the channel dimension
    combined_data = np.concatenate([
        img_expanded, 
        model_segmentation_sagittal,
        model_segmentation_coronal,
        model_segmentation_axial
    ], axis=-1)
    
    print(f"Combined data shape: {combined_data.shape}")
    
    # Initialize output array
    height, width, depth = img.shape
    output = np.zeros((height, width, depth), dtype=np.int64)
    
    # Process each voxel individually
    print("Running consensus model inference voxel by voxel...")
    
    # Using a progress tracker to give feedback
    total_voxels = height * width * depth
    voxels_processed = 0
    last_percent = -1
    
    for i in range(height):
        for j in range(width):
            for k in range(depth):
                # Extract features for this voxel
                voxel_features = combined_data[i, j, k]
                
                # Reshape to match expected input: [batch, 1, 1, 1, channels]
                X = np.expand_dims(voxel_features, axis=0)  # Add batch dimension
                X = np.expand_dims(X, axis=1)  # Add height dimension
                X = np.expand_dims(X, axis=1)  # Add width dimension
                X = np.expand_dims(X, axis=1)  # Add depth dimension
                
                # Create TinyGrad tensor
                X_tensor = Tensor(X.astype(np.float32), requires_grad=False)
                
                # Run inference on single voxel with the correct input name
                outputs = consensus_runner({consensus_input_name: X_tensor})
                
                # Get output tensor
                output_tensor = list(outputs.values())[0]
                output_data = output_tensor.numpy()
                
                # Get prediction for this voxel
                if len(output_data.shape) == 5:  # Shape: [1, 1, 1, 1, N]
                    pred = np.argmax(output_data[0, 0, 0, 0])
                else:
                    pred = output_data[0, 0, 0, 0]
                
                # Store prediction
                output[i, j, k] = pred
                
                # Update progress
                voxels_processed += 1
                percent_complete = (voxels_processed * 100) // total_voxels
                if percent_complete > last_percent and percent_complete % 10 == 0:
                    print(f"Progress: {percent_complete}% complete")
                    last_percent = percent_complete
    
    print("Consensus model processing complete.")
    
    return output

# ---------------------
# Main Execution Logic
# ---------------------
def main():
    # Configuration
    OUTPUT_PATH = 'output.nii.gz'
    SCAN_PATH = 'input.nii.gz'
    SAGITTAL_MODEL_PATH = 'models/sagittal_model.onnx'
    AXIAL_MODEL_PATH = 'models/axial_model.onnx'
    CORONAL_MODEL_PATH = 'models/coronal_model.onnx'
    CONSENSUS_LAYER_PATH = 'models/consensus_layer.onnx'
    SEGMENTATION_PATH = None
    ANTERIOR_COMMISSURE = None
    INSPECT_ONLY = False  # Set to True to only inspect models without running inference
    
    # Set working directory to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Print script information
    print("MRI Segmentation with TinyGrad ONNX Runner")
    print("===========================================")
    print(f"Working directory: {script_dir}")
    print(f"Output path: {OUTPUT_PATH}")
    print(f"Scan path: {SCAN_PATH}")
    print(f"Sagittal model: {os.path.basename(SAGITTAL_MODEL_PATH)}")
    print(f"Axial model: {os.path.basename(AXIAL_MODEL_PATH)}")
    print(f"Coronal model: {os.path.basename(CORONAL_MODEL_PATH)}")
    print(f"Consensus model: {os.path.basename(CONSENSUS_LAYER_PATH)}")
    print("===========================================\n")
    
    # Check if we should just inspect the models
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        inspect_onnx_model(sys.argv[1])
        return
    
    # Optionally inspect all models first
    if INSPECT_ONLY:
        print("Inspecting ONNX models...")
        inspect_onnx_model(SAGITTAL_MODEL_PATH)
        inspect_onnx_model(CORONAL_MODEL_PATH)
        inspect_onnx_model(AXIAL_MODEL_PATH)
        inspect_onnx_model(CONSENSUS_LAYER_PATH)
        return
    
    # Load scan
    nii = nib.load(SCAN_PATH)
    nii_seg = nib.load(SEGMENTATION_PATH) if SEGMENTATION_PATH else None
    print(nii.shape)
    
    # Extract subject ID
    subject = SCAN_PATH.split('/')[-1].replace('.nii', '')
    if subject.startswith('r'):
        subject = subject[1:]
    
    # Preprocess the MRI
    nii_out, nii_seg_out, coords, anterior_commissure, reconstruction_parms = preprocess_head_MRI(
        nii, 
        nii_seg, 
        anterior_commissure=ANTERIOR_COMMISSURE, 
        keep_parameters_for_reconstruction=True
    )     
    
    # Segment the MRI using TinyGrad's ONNX runtime
    segmentation = segment_MRI_tinygrad(
        nii_out.get_fdata(), 
        coords, 
        SAGITTAL_MODEL_PATH,
        AXIAL_MODEL_PATH, 
        CORONAL_MODEL_PATH, 
        CONSENSUS_LAYER_PATH
    )
    
    # Save the segmentation result
    nii_out_pred = nib.Nifti1Image(np.array(segmentation, dtype='int16'), nii_out.affine)
    nib.save(nii_out_pred, OUTPUT_PATH)
    print(f"Segmentation saved to {OUTPUT_PATH}")

if __name__ == '__main__':
    main()
