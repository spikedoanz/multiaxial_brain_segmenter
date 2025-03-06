#!/usr/bin/env python3
import nibabel as nib
import numpy as np

correct_img = nib.load('correct.nii.gz')
output_img = nib.load('output.nii.gz')
diff_data = correct_img.get_fdata() - output_img.get_fdata()
diff_img = nib.Nifti1Image(diff_data, correct_img.affine, correct_img.header)
nib.save(diff_img, 'diff.nii.gz')
print("Difference saved to diff.nii.gz")
