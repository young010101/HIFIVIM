# %%
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# %%
img=nib.load('t1_icbm_normal_1mm_pn3_rf20.mnc.gz')
data=img.get_fdata()

# %%
import sigpy.plot as pl
pl.ImagePlot(data[90,:,:])

# %%
print(data.shape)
print(data.dtype)

# %%
plt.imshow(data[90,:,:],cmap='gray')

# %%
from dipy.sims.voxel import single_tensor  # 或 multi_tensor 对于复杂组织
from dipy.data import fetcher  # 但用 BrainWeb 数据替换
from dipy.core.gradients import gradient_table

# 示例：为 tissue map 生成 DWI
b_vals = np.array([0, 500, 1000, 2000])  # s/mm²
gtab = gradient_table(b_vals, bvecs=None)  # 需 dipy.io
data = single_tensor(gtab, S0=100, diff=0.001, eigvals=(0.001, 0.0005, 0.0005))  # 调整为 phantom
# 应用到整个体积：循环体素，使用 tissue-specific diff
# %%
from dipy.data import get_fnames
fraw, fbval, fbvec = get_fnames(name='small_64D')
# %%
from dipy.io import read_bvals_bvecs
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
# %%
from dipy.viz import window, actor
scene = window.Scene()
scene.add(actor.point(bvecs, window.colors.red, point_radius=0.02))
window.show(scene)
# %%
gtab = gradient_table(bvals=bvals, bvecs=bvecs)
gtab.info
# %%
from dipy.sims.voxel import single_tensor
data=single_tensor(gtab=gtab)

# %%
