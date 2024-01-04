# multiaxial_brain_segmenter


# INPUT:

SUBJECT_PATH = Path to T1w MRI to segment in format nifti file (.nii )
SEGMENTATION_PATH = [optional] segmentation of MRI to use as ground truth

OUTPUT_PATH = Path to folder to which output segmentations will be saved (it will create folder if nonexistent)

SAGITTAL_MODEL_SESSION_PATH = Path to sagittal segmenter model (included in repository), in format .h5
AXIAL_MODEL_SESSION_PATH = Path to axial segmenter model (included in repository), in format .h5
CORONAL_MODEL_SESSION_PATH = Path to coronal segmenter model (included in repository), in format .h5

# OUTPUT

In OUTPUT_PATH you will find 4 files: Individual segmentations along each axis by each respective model, and the consensus segmentation made by majority vote.

Tested on Tensorflow 2 (2.0.0) and Tensorflow 1 (1.14)

