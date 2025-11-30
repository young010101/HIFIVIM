#%%
%run -i ../bart/startup

# %%
import numpy as np
import matplotlib.pyplot as plt
import itertools
import matplotlib
import multiprocessing
import os
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.image import load_nifti_data
from dipy.viz import window, actor
import sigpy.plot as pl


#%%
fraw, fbval,fbvec = get_fnames(name='ivim')
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
gtab = gradient_table(bvals=bvals, bvecs=bvecs)

#%%
print(bvals.shape)
print(bvecs.shape)

# %%
data=load_nifti_data(fraw)
print(data.shape)
# %%
pl.ImagePlot(data[:,:,27,0])

# %%
bart(0, 'show -h')

# %%
size = 90
fs = np.linspace(0, 0.4, size)  # 0, 0.4
Ds = np.linspace(0.3e-3, 3e-3, size)  # try it
Dstars = np.linspace(5e-3, 60e-3, size)
size = int(size * size * size)
ivim_dicc = np.zeros((size, len(bvals)))
# %%
args2 = [(f, D, Dstar, bvals, i) for i, (f, D, Dstar) in enumerate(itertools.product(fs, Ds, Dstars))]
# %%
import sys
# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# current_dir = os.path.join(current_dir, 'code')
# 或者获取项目根目录（往上一级）
project_dir = os.path.dirname(current_dir)
sys.path.insert(0, project_dir)  # 或者 current_dir

from HIFIVIM.code.utils import ivim_model
from HIFIVIM.code.dictionary_gen import get_dicc
# %%
# from mytools.utils import ivim_model

# %%

def get_dicc(f, D, Dstar, bvals, i):
    signal = ivim_model(f, D, Dstar, bvals)
    return i, signal, (f, D, Dstar)

with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    results = pool.starmap(get_dicc, args2)
    pool.close()
    pool.join()
    results = np.asarray(results, dtype=object)
    for i in range(results.shape[0]):
        idx, signal, params = results[i]
        ivim_dicc[idx, :] = signal
# %%
print("Now extracting basis set ...")
dicc = np.transpose(ivim_dicc, (1, 0))
U, S, V = bart(3, 'svd -e', dicc)
S1 = np.cumsum(S)
S1 = S1 / S1.max()
# %%
basis = bart(1, 'extract 1 0 {}'.format(5), U)  # extract basis
basis = bart(1, 'transpose 1 6', basis)  # place into correct bart format
basis = bart(1, 'transpose 0 5', basis)
# %%

import nibabel as nib
img=nib.load('Phantom/sens_maps.nii.gz')
data=img.get_fdata(dtype='complex128')
# %%
