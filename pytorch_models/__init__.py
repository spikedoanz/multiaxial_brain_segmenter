"""
PyTorch implementation of brain segmentation models.
"""

from .models import ViewSpecificModel, ConsensusModel
from .dice_metrics import (
    dice_coef, 
    generalized_dice_coef_multilabel7,
    dice_coef_multilabel_bin0,
    dice_coef_multilabel_bin1,
    dice_coef_multilabel_bin2,
    dice_coef_multilabel_bin3,
    dice_coef_multilabel_bin4,
    dice_coef_multilabel_bin5,
    dice_coef_multilabel_bin6
)
from .utils import segment_MRI_pytorch, create_coordinate_matrix
from .weight_converter import convert_model