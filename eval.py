#!/usr/bin/env python3
import os
import sys
import nibabel as nib
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
from skimage.transform import resize
from nibabel.orientations import axcodes2ornt, ornt_transform

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
# ONNX Segmentation Logic
# ------------------
def process_sagittal_slices(session, img, coords):
    """Process all sagittal slices through the model and return the predictions."""
    # Get input names
    input_name_img = session.get_inputs()[0].name
    input_name_coords = session.get_inputs()[1].name
    
    # Initialize output array (7 classes per slice prediction)
    output = np.zeros((img.shape[0], img.shape[1], img.shape[2], 7), dtype=np.float32)
    
    print(f"Processing {img.shape[0]} sagittal slices...")
    
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
        
        # Run inference
        outputs = session.run(None, {
            input_name_img: img_input,
            input_name_coords: coords_input
        })
        
        # Store prediction (remove batch dimension)
        output[i, :, :, :] = outputs[0][0]
    
    return output

def process_coronal_slices(session, img, coords):
    """Process all coronal slices through the model and return the predictions."""
    # Get input names
    input_name_img = session.get_inputs()[0].name
    input_name_coords = session.get_inputs()[1].name
    
    # Initialize output array (7 classes per slice prediction)
    output = np.zeros((img.shape[0], img.shape[1], img.shape[2], 7), dtype=np.float32)
    
    print(f"Processing {img.shape[1]} coronal slices...")
    
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
        
        # Run inference
        outputs = session.run(None, {
            input_name_img: img_input,
            input_name_coords: coords_input
        })
        
        # Store prediction (remove batch dimension)
        output[:, i, :, :] = outputs[0][0]
    
    return output

def process_axial_slices(session, img, coords):
    """Process all axial slices through the model and return the predictions."""
    # Get input names
    input_name_img = session.get_inputs()[0].name
    input_name_coords = session.get_inputs()[1].name
    
    # Initialize output array (7 classes per slice prediction)
    output = np.zeros((img.shape[0], img.shape[1], img.shape[2], 7), dtype=np.float32)
    
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
        
        # Run inference
        outputs = session.run(None, {
            input_name_img: img_input,
            input_name_coords: coords_input
        })
        
        # Store prediction (remove batch dimension)
        output[:, :, i, :] = outputs[0][0]
    return output

def segment_MRI_onnx(img, coords, model_sagittal_path=None, model_axial_path=None, model_coronal_path=None, consensus_model_path=None):
    """
    Segment an MRI image using the provided ONNX models.
    
    This function processes 2D slices of a 3D brain using the sagittal, coronal, and axial models,
    then combines the predictions using the consensus model.
    
    Each coordinate model (sagittal, coronal, axial) expects 2D inputs with shapes:
    - Image: [1, 256, 256, 1] (batch, height, width, channels)
    - Coordinates: [1, 256, 256, 3] (batch, height, width, coordinate channels)
    """
    import onnxruntime as ort
    import numpy as np
    
    model_segmentation_sagittal = None
    model_segmentation_coronal = None
    model_segmentation_axial = None
    
    # Create ONNX Runtime session options with optimizations
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Debug helper function
    def print_model_inputs(session, model_name):
        """Print information about model inputs to help with debugging"""
        print(f"\n{model_name} Input Details:")
        for i, input_info in enumerate(session.get_inputs()):
            print(f"  Input #{i}")
            print(f"    Name: {input_info.name}")
            print(f"    Shape: {input_info.shape}")
            print(f"    Type: {input_info.type}")
        print("")

    # Process slice-by-slice for each model
    if model_sagittal_path is not None:
        print("Running sagittal model inference...")
        sagittal_session = ort.InferenceSession(model_sagittal_path, session_options)
        print_model_inputs(sagittal_session, "Sagittal Model")
        model_segmentation_sagittal = process_sagittal_slices(sagittal_session, img, coords)
    
    if model_coronal_path is not None:
        print("Running coronal model inference...")
        coronal_session = ort.InferenceSession(model_coronal_path, session_options)
        print_model_inputs(coronal_session, "Coronal Model")
        model_segmentation_coronal = process_coronal_slices(coronal_session, img, coords)
    
    if model_axial_path is not None:
        print("Running axial model inference...")
        axial_session = ort.InferenceSession(model_axial_path, session_options)
        print_model_inputs(axial_session, "Axial Model")
        model_segmentation_axial = process_axial_slices(axial_session, img, coords)
    
    # Create empty outputs for any models that didn't run
    if model_segmentation_sagittal is None:
        model_segmentation_sagittal = np.zeros((img.shape[0], img.shape[1], img.shape[2], 7), dtype=np.float32)
    if model_segmentation_coronal is None:
        model_segmentation_coronal = np.zeros((img.shape[0], img.shape[1], img.shape[2], 7), dtype=np.float32)
    if model_segmentation_axial is None:
        model_segmentation_axial = np.zeros((img.shape[0], img.shape[1], img.shape[2], 7), dtype=np.float32)
    
    # Run consensus model
    print("Running consensus model...")
    consensus_session = ort.InferenceSession(consensus_model_path, session_options)
    print_model_inputs(consensus_session, "Consensus Model")
    
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
    input_name = consensus_session.get_inputs()[0].name
    
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
                
                # Run inference on single voxel
                outputs = consensus_session.run(None, {input_name: X})
                
                # Get prediction for this voxel
                if len(outputs[0].shape) == 5:  # Shape: [1, 1, 1, 1, N]
                    pred = np.argmax(outputs[0][0, 0, 0, 0])
                else:
                    pred = outputs[0][0, 0, 0, 0]
                
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
    
    # Segment the MRI using ONNX models
    segmentation = segment_MRI_onnx(
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
