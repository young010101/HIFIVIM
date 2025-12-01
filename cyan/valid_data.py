# %%
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
# Phantom 数据值范围：如果是T1,应该在 0.02s 到 1s之间
'''
fat 300
wm 800
gm 1400
CSF 4000
water 4000
'''
# %% ===================================================
phan_t1_img=nib.load('../Phantom/Phantom_T1.nii.gz')
phan_t1_data = phan_t1_img.get_fdata()

# %%
print(np.max(phan_t1_data))
print(phan_t1_data.min())

plt.imshow(phan_t1_data[:,:,90])
# %%
plt.hist(phan_t1_data[:,:,0])
# %%===================================================
t1_icbm_normal_img = nib.load('../local/t1_icbm_normal_1mm_pn3_rf20.mnc.gz')
t1_icbm_normal=t1_icbm_normal_img.get_fdata()
#%%
plt.imshow(abs(t1_icbm_normal[90,:,:]))
# %%=============================================
phase_esti_img=nib.load('../Phantom/phase_estimates.nii.gz')
phase_esti = phase_esti_img.get_fdata()
print(phase_esti.shape)
print(phase_esti.max())
print(phase_esti.min())
plt.imshow(abs(phase_esti[:,:,0]))
plt.colorbar()
fig, axes = plt.subplots(ncols=5, nrows=3)
for i, ax in enumerate(axes.flat):
    ax.imshow(abs(phase_esti[:,:,i]))
    ax.axis('off')
plt.tight_layout()
plt.subplots_adjust(wspace=0,hspace=0)

# %%================================================
sens_img=nib.load('../Phantom/sens_maps.nii.gz')
sens_data =sens_img.get_fdata()
print(sens_data.shape)
print(sens_data.max())
print(sens_data.min())
plt.imshow(abs(sens_data[:,:,0,0]))
plt.imshow(np.angle(sens_data[:,:,0,0]))
plt.colorbar()
# %%
