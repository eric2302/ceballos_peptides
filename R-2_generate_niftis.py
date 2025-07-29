# %%
import numpy as np
import pandas as pd
import nibabel as nib
from abagen import get_interpolated_map

receptors = pd.read_csv('data/receptor_filtered.csv')['gene'].tolist()
mask = 'data/atlas_space-MNI152_den-2mm_desc-T1_GM.nii.gz'
data_dir = 'data/annotations/abagen'

maps = get_interpolated_map(receptors, mask=mask, data_dir=data_dir, n_neighbors=5, 
                            lr_mirror='bidirectional', ibf_threshold=0.2, 
                            norm_matched=True)

aff = nib.load(mask).affine
for receptor in receptors:
    fn = f"{data_dir}/template_space-MNI152_den-2mm_desc-{receptor}_gene_expression.nii.gz"
    nii = nib.Nifti1Image(maps[receptor], aff)
    nib.save(nii, fn)
