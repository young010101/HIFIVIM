# %%
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
# %% Phantom 数据值范围：如果是T1,应该在 0.02s 到 1s之间
'''
fat 300
wm 800
gm 1400
CSF 4000
water 4000
'''
phan_t1_img=nib.load('../Phantom/Phantom_T1.nii.gz')
phan_t1_data = phan_t1_img.get_fdata()

# %%
print(np.max(phan_t1_data))
print(phan_t1_data.min())

plt.imshow(phan_t1_data[:,:,90])
# %%
plt.hist(phan_t1_data[:,:,0])
# %%
