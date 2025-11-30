# %% 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import sigpy.plot as pl
from dipy.sims.voxel import single_tensor  # 或 multi_tensor 对于复杂组织
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti_data
from dipy.viz import window, actor

# %% 导入基础数据，可能是组织，也可能是T2图像
img=nib.load('t1_icbm_normal_1mm_pn3_rf20.mnc.gz')
data=img.get_fdata()

pl.ImagePlot(data[90,:,:])

print(data.shape)
print(data.dtype)

# %% !fail!!
# 示例：为 tissue map 生成 DWI
b_vals = np.array([0., 500., 1000., 2000.])  # s/mm²
b_vecs=np.array([[0.,0.,1.] for _ in b_vals])  # 简单示例，所有方向相同
gtab = gradient_table(bvals=b_vals, bvecs=b_vecs)  # 需 dipy.io
data_sin = single_tensor(gtab, S0=10)  # 调整为 phantom
# 应用到整个体积：循环体素，使用 tissue-specific diff

# %%
fraw, fbval, fbvec = get_fnames(name='ivim')
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
gtab = gradient_table(bvals=bvals, bvecs=bvecs)
gtab.info
# %%
single_ten=single_tensor(gtab=gtab)

# %%
ivim_data=load_nifti_data(fraw)
fig,ax=plt.subplots(ncols=10,nrows=2,figsize=(30,6))
plt.subplots_adjust(wspace=0, hspace=0)
# plt.subplots_adjust(left=0., right=1., top=1., bottom=0.0)
for i in range(10):
    ax[0][i].imshow(ivim_data[:,:,27,i],cmap='gray')
    ax[0][i].set_title(f'Slc 27, Vol {i}', fontweight='bold')
    ax[0][i].axis('off')
    
    ax[1][i].imshow(ivim_data[:,:,27,i+10],cmap='gray')
    ax[1][i].axis('off')
    if i in [4,6,8]:
        ax[0][i].annotate('', xy=(0.72, 0.78), xytext=(0.82, 0.95),
                    arrowprops=dict(arrowstyle='->', color='yellow', lw=2.5),
                    xycoords='axes fraction', textcoords='axes fraction')
    # 可选：移除子图之间的边框线
    for axx in ax.flat:
        axx.tick_params(which='both', direction='in')
# %%
