#%%
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.image import load_nifti

# %%
hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames(name='stanford_hardi')
bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab=gradient_table(bvals=bvals, bvecs=bvecs)

# %%
data, affine = load_nifti(hardi_fname)
# %%
