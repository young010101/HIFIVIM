# %%
%run ../bart/startup.py
#%%
import os
import nibabel as nib
import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append('./code')

from utils import *
# %%

slice = 90


bvals = [0, 5, 7, 10, 15, 20, 30, 40, 50, 60, 100, 200, 400, 700, 1000]
phantom = nib.load(os.path.join('./Phantom', 'Phantom_T1.nii.gz')).get_fdata()[..., slice]

Dt, Fp, Dp = np.zeros_like(phantom), np.zeros_like(phantom), np.zeros_like(phantom)
phantom_data = np.zeros((phantom.shape[0], phantom.shape[1], 15))
WMmask, GMmask, CSFmask, WMH_mask1,WMH_mask2, BGmask = np.zeros_like(Dt), np.zeros_like(Dt), \
                                            np.zeros_like(Dt), np.zeros_like(Dt), np.zeros_like(Dt), np.zeros_like(Dt)

center = (45,100)
height, width = phantom.shape

WMH_mask1 = np.zeros_like(Dt)
y, x = np.ogrid[:height, :width]
radius = 5
WMH_mask1[(x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2] = 1

WMH_mask2 = np.zeros_like(Dt)
center, radius = (53,60), 3.5
WMH_mask2[(x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2] = 1

WMH_mask3 = np.zeros_like(Dt)
center, radius = (60,100), 3
WMH_mask3[(x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2] = 1


# %%
plt.imshow(phantom * (np.ones_like(Dt) - WMH_mask1 - WMH_mask2 - WMH_mask3))
plt.colorbar()
# %% !! 这里需要注意仿真的 threshold, 可能会影响仿真图

for i in range(phantom.shape[0]):
    for j in range(phantom.shape[1]):
        if WMH_mask1[i,j] == 1:
            Dt[i, j] = 0.0012
            Fp[i, j] = 0.16
            Dp[i, j] = 0.025
        elif WMH_mask2[i, j] == 1:
            Dt[i, j] = 0.0014
            Fp[i, j] = 0.17
            Dp[i, j] = 0.028
        elif WMH_mask3[i, j] == 1:
            Dt[i, j] = 0.0013
            Fp[i, j] = 0.165
            Dp[i, j] = 0.027
        else:
            if j < 101 and j > 65 and i > 62 and i < 103:
                Dt[i, j] = 0.0006 if phantom[i, j] > 2.5 else 0.0005 if phantom[i, j] > 1.7 else 0.003 if phantom[
                                                                                                                i, j] > 0 else 0
                Fp[i, j] = 0.07 if phantom[i, j] > 2.5 else 0.06 if phantom[i, j] > 1.7 else 0.25 if phantom[
                                                                                                            i, j] > 0 else 0
                Dp[i, j] = 0.045 if phantom[i, j] > 2.5 else 0.055 if phantom[i, j] > 1.7 else 0.02 if phantom[ i, j] > 0 else 0
            else:
                Dt[i, j] = 0.0006 if phantom[i, j] > 2.35 else 0.0009 if phantom[i, j] > 1.7 else 0.003 if phantom[
                                                                                                                i, j] > 0 else 0
                Fp[i, j] = 0.07 if phantom[i, j] > 2.35 else 0.14 if phantom[i, j] > 1.7 else 0.2 if phantom[
                                                                                                        i, j] > 0 else 0
                Dp[i, j] = 0.045 if phantom[i, j] > 2.35 else 0.03 if phantom[i, j] > 1.7 else 0.02 if phantom[
                                                                                                            i, j] > 0 else 0
        if Dt[i,j] == 0.0006:
            WMmask[i,j] = 1
        elif Dt[i, j] == 0.0005:
            BGmask[i, j] = 1
        elif Dt[i, j] == 0.0009:
            GMmask[i, j] = 1
        elif Dt[i, j] == 0.003:
            CSFmask[i, j] = 1


        phantom_data[i,j] = ivim_model(Fp[i,j], Dt[i,j], Dp[i,j], bvals)
        phantom_data[i,j][Dt[i,j] == 0] =0

# %%
phantom_data.shape
# %%
plt.figure()
plt.subplot(231), plt.imshow(np.rot90(WMmask), cmap='gray'), plt.axis('off')
plt.subplot(232),  plt.imshow(np.rot90(GMmask), cmap='gray'), plt.axis('off')
plt.subplot(233), plt.imshow(np.rot90(BGmask), cmap='gray'), plt.axis('off')
plt.subplot(234), plt.imshow(np.rot90(WMH_mask1), cmap='gray'), plt.axis('off')
plt.subplot(235), plt.imshow(np.rot90(WMH_mask2), cmap='gray'), plt.axis('off')
plt.subplot(236), plt.imshow(np.rot90(WMH_mask3), cmap='gray'), plt.axis('off')

cmap = 'turbo'
fig, axes = plt.subplots(1, 4)
axes[0].imshow(np.rot90(phantom), cmap='gray'), axes[0].set_xticks([]), axes[0].set_yticks([])
axes[0].set_title('Phantom', fontsize=16, fontweight='bold')

im = axes[1].imshow(np.rot90(Dt), cmap=cmap)
axes[1].set_xticks([]), axes[1].set_yticks([])
axes[1].set_title('D', fontsize=16, fontweight='bold'), im.set_clim(0.0003, 0.0015)
cax = fig.add_axes([axes[1].get_position().x1 + 0.005,
                    axes[1].get_position().y0, 0.01, axes[1].get_position().height])
cbar = plt.colorbar(axes[1].images[0], cax=cax)


im = axes[2].imshow(np.rot90(Fp), cmap=cmap)
axes[2].set_xticks([]), axes[2].set_yticks([])
axes[2].set_title('f', fontsize=16, fontweight='bold'), im.set_clim(0.04, 0.2)
cax = fig.add_axes([axes[2].get_position().x1 + 0.005,
                    axes[2].get_position().y0, 0.01, axes[2].get_position().height])
cbar = plt.colorbar(axes[2].images[0], cax=cax)

im = axes[3].imshow(np.rot90(Dp), cmap=cmap)
axes[3].set_xticks([]), axes[3].set_yticks([])
axes[3].set_title('D*', fontsize=16, fontweight='bold'), im.set_clim(0.01, 0.06)
cax = fig.add_axes([axes[3].get_position().x1 + 0.005,
                    axes[3].get_position().y0, 0.01, axes[3].get_position().height])
cbar = plt.colorbar(axes[3].images[0], cax=cax)

plt.subplots_adjust(wspace=0.5)

plt.figure()
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(np.rot90(phantom_data[...,(i+3)*2]), cmap='gray'), plt.clim(0.2,1), plt.axis('off')
    plt.title("b = {} s/mm$^2$".format(bvals[(i+3)*2]), fontsize=14, fontweight='bold')

plt.show()
# %%
Dt,Fp,Dp, ivim, masks= Dt, Fp, Dp, phantom_data, (np.rot90(WMmask), np.rot90(GMmask), np.rot90(CSFmask),
                                      np.rot90(BGmask), np.rot90(WMH_mask1), np.rot90(WMH_mask2), np.rot90(WMH_mask3))
# %%
def plot_demo(img):
    _, axes = plt.subplots(nrows=1, ncols=5, figsize=(15, 3))
    for i, ax in enumerate(axes.flat):
        ax.imshow(img[..., i * 3].real, cmap='gray')
        ax.set_title('b = {} s/mm$^2$'.format(bvals[i * 3]), fontsize=14, fontweight='bold')
        ax.axis('off')
plot_demo(ivim)
# %%
plt.imshow(phantom)

composite_ivim = ivim.astype(np.complex64)
# %%
composite_ivim = add_noise(composite_ivim, 20, return_img=False)
# %%
print(composite_ivim.shape)
# %%
_, axes = plt.subplots(nrows=1, ncols=5, figsize=(15, 3))
for i, ax in enumerate(axes.flat):
    ax.imshow(composite_ivim[..., i * 3].real, cmap='gray')
    ax.set_title('b = {} s/mm$^2$'.format(bvals[i * 3]), fontsize=14, fontweight='bold')
    ax.axis('off')

# %%
