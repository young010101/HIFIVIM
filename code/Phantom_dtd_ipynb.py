# %% run ../../../src/bart/startup.py
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import datetime
from argparse import ArgumentParser
import cfl
import nibabel as nib    
from bart import bart
from Phantom_utils import add_phase, add_sens_maps, get_fft, \
    pub_figure, bland_altman_image
from utils import add_noise, get_initial_sens, lowres_phaseremoval, \
    get_composite_sens, ivim_fit_segmented, llr_recon, median_otsu
# from utils import dtd_gamma_model as ivim_model


def parser(argv=None):
    parse = ArgumentParser()
    parse.add_argument('--phantom_dir', type=str, default='phantom_1.0mm_normal_fuzzy')
    parse.add_argument('--outdir', type=str)
    return parse.parse_args(argv)


# %%
args = parser(["--outdir", "../Phantom"])


# Save key recon results as NIfTI (.nii.gz)
def save_nifti(volume, out_path, dtype=np.float32, affine=None):
    import nibabel as nib
    vol = np.asarray(volume, dtype=dtype)
    if vol.ndim == 2:
        vol = vol[..., np.newaxis]
    if affine is None:
        affine = np.eye(4)
    img = nib.Nifti1Image(vol, affine)
    nib.save(img, out_path)

outdir_nifti = os.path.join(args.outdir, 'phan_cyan/DATA/brain/NII')


# args.__dict__
# args = SimpleNamespace(phantom_dir='phantom_1.0mm_normal_fuzzy', outdir='/tmp/out')
# %%
if os.path.exists(args.outdir):
    pass
else:
    os.mkdir(args.outdir)
    print(f"Created directory {args.outdir}")

# bvals = [0, 5, 7, 10, 15, 20, 30, 40, 50, 60, 100, 200, 400, 700, 1000]
bvals = np.asarray([0, 7, 10, 15, 20, 40, 50, 60, 100, 200, 400, 700, 1000, 1400, 2000])  # s/mm^2


# %% make phantom
# debugging
def dtd_gamma_model(
    s0,
    d_iso,
    mu2_iso,
    mu2_aniso,
    bvals,
    b_delta=None,
    b_eta=None,
    rs=None,
    s_ind=None,
):
    """
    Python equivalent of dtd_gamma_1d_fit2data

    Parameters
    ----------
    bvals : array
        b-values (ms/um^2)
    s0 : float
        Baseline signal
    d_iso : float
        Mean diffusivity (MD) (um^2/ms)
    mu2_iso : float
        Isotropic variance
    mu2_aniso : float
        Anisotropic variance
    b_delta : array or None
        b-tensor anisotropy
    b_eta : array or None
        Asymmetry parameter
    rs : array or None
        Relative signal scaling across series
    s_ind : array or None
        Series index

    Returns
    -------
    s : array
        Signal S(b)
    """

    bvals = np.asarray(bvals) * 1e-3  # convert to ms/um^2, identical to MATLAB code in md-dmri

    # ---- baseline weighting (series-dependent S0) ----
    if rs is not None and s_ind is not None:
        rs = np.asarray([1.0] + list(rs))
        sw = s0 * np.sum(
            (rs[None, :] * (s_ind[:, None] == np.arange(1, len(rs) + 1))),
            axis=1,
        )
    else:
        sw = s0

    # ---- total diffusional variance ----
    if b_delta is None:
        mu2 = mu2_iso
    else:
        if b_eta is None:
            b_eta = 0
        mu2 = mu2_iso + mu2_aniso * b_delta**2 * (b_eta**2 + 3) / 3

    # ---- gamma model signal ----
    s = sw * (1 + bvals * mu2 / d_iso) ** (-d_iso**2 / mu2)
    
    # if np.isnan(s).any():
    #     print("NaN encountered in dtd_gamma_model with parameters:")
    #     print(f"s0={s0}, d_iso={d_iso}, mu2_iso={mu2_iso}, mu2_aniso={mu2_aniso}")
    #     print(f"bvals={bvals}")
    #     raise ValueError("NaN encountered in dtd_gamma_model output.")
    if (s > s0).any():
        print("Warning: Signal greater than baseline encountered in dtd_gamma_model.")

    return np.real(s)


ivim_model = dtd_gamma_model


def make_phantom(args, show=True):
    import nibabel as nib
    import matplotlib.pyplot as plt
    slice = 90


    phantom = nib.load(os.path.join(args.outdir, 'Phantom_T1.nii.gz')).get_fdata()[..., slice]
    phantom[phantom < 200] = 0
    phantom = phantom / phantom.max() * 4.  # scale to 0-3.

    md, vi, va = np.zeros_like(phantom), np.zeros_like(phantom), np.zeros_like(phantom)
    phantom_data = np.zeros((phantom.shape[0], phantom.shape[1], bvals.shape[0]))
    phantom_data_b_delta_1 = np.zeros_like(phantom_data)
    WMmask, GMmask, CSFmask, WMH_mask1,WMH_mask2, BGmask = np.zeros_like(md), np.zeros_like(md), \
                                              np.zeros_like(md), np.zeros_like(md), np.zeros_like(md), np.zeros_like(md)

    center = (45,100)
    height, width = phantom.shape

    WMH_mask1 = np.zeros_like(md)
    y, x = np.ogrid[:height, :width]
    radius = 5
    WMH_mask1[(x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2] = 1

    WMH_mask2 = np.zeros_like(md)
    center, radius = (53,60), 3.5
    WMH_mask2[(x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2] = 1

    WMH_mask3 = np.zeros_like(md)
    center, radius = (60,100), 3
    WMH_mask3[(x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2] = 1


    for i in range(phantom.shape[0]):
        for j in range(phantom.shape[1]):
            if WMH_mask1[i,j] == 1:
                # md[i, j] = 0.0012
                md[i, j] = 1.2
                vi[i, j] = 0.96
                va[i, j] = 0.925
            elif WMH_mask2[i, j] == 1:
                md[i, j] = 1.4
                vi[i, j] = 0.97
                va[i, j] = 0.928
            elif WMH_mask3[i, j] == 1:
                md[i, j] = 1.3
                vi[i, j] = 0.965
                va[i, j] = 0.927
            else:
                if j < 101 and j > 65 and i > 62 and i < 103:
                    md[i, j] = 0.6 if phantom[i, j] > 2.5 else 0.5 if phantom[i, j] > 1.7 else 3 if phantom[
                                                                                                                  i, j] > 0 else 0
                    vi[i, j] = 0.7 if phantom[i, j] > 2.5 else 0.6 if phantom[i, j] > 1.7 else 2.5 if phantom[
                                                                                                            i, j] > 0 else 0
                    va[i, j] = 0.45 if phantom[i, j] > 2.5 else 0.55 if phantom[i, j] > 1.7 else 0.2 if phantom[ i, j] > 0 else 0
                else:
                    md[i, j] = 0.6 if phantom[i, j] > 2.35 else 0.9 if phantom[i, j] > 1.7 else 3 if phantom[
                                                                                                                  i, j] > 0 else 0
                    vi[i, j] = 0.7 if phantom[i, j] > 2.35 else 1.4 if phantom[i, j] > 1.7 else 2 if phantom[
                                                                                                            i, j] > 0 else 0
                    va[i, j] = 0.45 if phantom[i, j] > 2.35 else 0.3 if phantom[i, j] > 1.7 else 0.2 if phantom[
                                                                                                              i, j] > 0 else 0
            if md[i,j] == 0.6:
                WMmask[i,j] = 1
            elif md[i, j] == 0.5:
                BGmask[i, j] = 1
            elif md[i, j] == 0.9:
                GMmask[i, j] = 1
            elif md[i, j] == 3:
                CSFmask[i, j] = 1


            phantom_data[i,j] = ivim_model(10, md[i,j], vi[i,j], va[i,j], bvals)
            phantom_data[i,j][md[i,j] == 0] =0
            phantom_data_b_delta_1[i,j] = ivim_model(10, md[i,j], vi[i,j], va[i,j], bvals, b_delta=np.ones_like(bvals))
            phantom_data_b_delta_1[i,j][md[i,j] == 0] =0


    if show:
        plt.figure()
        plt.subplot(231), plt.imshow(np.rot90(WMmask), cmap='gray'), plt.axis('off')
        plt.subplot(232),  plt.imshow(np.rot90(GMmask), cmap='gray'), plt.axis('off')
        plt.subplot(233), plt.imshow(np.rot90(BGmask), cmap='gray'), plt.axis('off')
        plt.subplot(234), plt.imshow(np.rot90(WMH_mask1), cmap='gray'), plt.axis('off')
        plt.subplot(235), plt.imshow(np.rot90(WMH_mask2), cmap='gray'), plt.axis('off')
        plt.subplot(236), plt.imshow(np.rot90(WMH_mask3), cmap='gray'), plt.axis('off')

        cmap = 'turbo'
        fig, axes = plt.subplots(1, 4, figsize=(16,4))
        axes[0].imshow(np.rot90(phantom), cmap='gray'), axes[0].set_xticks([]), axes[0].set_yticks([])
        axes[0].set_title('Phantom', fontsize=16, fontweight='bold')
        plt.colorbar(axes[0].images[0], ax=axes[0], fraction=0.046, pad=0.04)

        im = axes[1].imshow(np.rot90(md), cmap=cmap)
        axes[1].set_xticks([]), axes[1].set_yticks([])
        axes[1].set_title('MD', fontsize=16, fontweight='bold'), im.set_clim(0.0003, 0.0015)
        cax = fig.add_axes([axes[1].get_position().x1 + 0.005,
                            axes[1].get_position().y0, 0.01, axes[1].get_position().height])
        cbar = plt.colorbar(axes[1].images[0], cax=cax)


        im = axes[2].imshow(np.rot90(vi), cmap=cmap)
        axes[2].set_xticks([]), axes[2].set_yticks([])
        axes[2].set_title('$V_I$', fontsize=16, fontweight='bold'), im.set_clim()
        cax = fig.add_axes([axes[2].get_position().x1 + 0.005,
                            axes[2].get_position().y0, 0.01, axes[2].get_position().height])
        cbar = plt.colorbar(axes[2].images[0], cax=cax)

        im = axes[3].imshow(np.rot90(va), cmap=cmap)
        axes[3].set_xticks([]), axes[3].set_yticks([])
        axes[3].set_title('$V_A$', fontsize=16, fontweight='bold'), im.set_clim()
        cax = fig.add_axes([axes[3].get_position().x1 + 0.005,
                            axes[3].get_position().y0, 0.01, axes[3].get_position().height])
        cbar = plt.colorbar(axes[3].images[0], cax=cax)

        plt.subplots_adjust(wspace=0.5)

        # ####################################################
        plt.figure(figsize=(15,3))
        for i in range(5):
            plt.subplot(1,5,i+1)
            plt.imshow(np.rot90(phantom_data[...,(i+3)*2]), cmap='gray'), plt.clim(), plt.axis('off')
            plt.title("b = {} s/mm$^2$".format(bvals[(i+3)*2]), fontsize=14, fontweight='bold')
            plt.colorbar()

        plt.show()

        # ########### select some point to plot signal curve ########################
        plt.figure(figsize=(15,3))
        points = [(82,82), (50,50), (30,130), (100,60)]
        for idx, (i,j) in enumerate(points):
            plt.subplot(1, len(points), idx+1)
            plt.plot(bvals, phantom_data[i,j], 'o-')
            plt.plot(bvals, phantom_data_b_delta_1[i,j], 'x--')
            plt.xlabel('b-values (s/mm$^2$)', fontsize=14, fontweight='bold')
            plt.ylabel('Signal Intensity', fontsize=14, fontweight='bold')
            plt.title('Signal Curve at ({},{})'.format(i,j), fontsize=14, fontweight='bold')
            plt.grid()
        

    return md, vi, va, phantom_data, phantom_data_b_delta_1, (np.rot90(WMmask), np.rot90(GMmask), np.rot90(CSFmask),
                                      np.rot90(BGmask), np.rot90(WMH_mask1), np.rot90(WMH_mask2), np.rot90(WMH_mask3))


mean_diff, var_iso, var_aniso, dtd_gamma, ivim_b_delta_1, masks = make_phantom(args, show=True)  # 164 x 164 x 15 for ivim
# %%
dtd_gamma = dtd_gamma
points = [(82, 82), (50, 50), (30, 130), (100, 60)]
plt.figure(figsize=(15,3))
for idx, (i, j) in enumerate(points):
    plt.subplot(1, len(points), idx+1)
    plt.plot(bvals, dtd_gamma[i,j], 'o-')
    plt.plot(bvals, ivim_b_delta_1[i,j], 'x--')
    plt.xlabel('b-values (s/mm$^2$)', fontsize=14, fontweight='bold')
    plt.ylabel('Signal Intensity', fontsize=14, fontweight='bold')
    plt.title('Signal Curve at ({},{})'.format(i,j), fontsize=14, fontweight='bold')
    plt.grid()
    plt.legend(['$b_{\Delta}=0$', '$b_{\Delta}=1$'])
# %%
phantom = nib.load(os.path.join(args.outdir, 'Phantom_T1.nii.gz'))
affine = phantom.affine
# %%
if dtd_gamma.ndim == 4:
    save_nifti(dtd_gamma, os.path.join(outdir_nifti, 'ivim_phantom.nii.gz'), dtype=np.float32, affine=affine)
    save_nifti(ivim_b_delta_1, os.path.join(outdir_nifti, 'ivim_phantom_b_delta_1.nii.gz'), dtype=np.float32, affine=affine)
elif dtd_gamma.ndim == 3:
    ivim_expand = dtd_gamma[:, :, np.newaxis, :]
    save_nifti(ivim_expand, os.path.join(outdir_nifti, 'ivim_phantom.nii.gz'), dtype=np.float32, affine=affine)
    ivim_b_delta_1_expand = ivim_b_delta_1[:, :, np.newaxis, :]
    save_nifti(ivim_b_delta_1_expand, os.path.join(outdir_nifti, 'ivim_phantom_b_delta_1.nii.gz'), dtype=np.float32, affine=affine)
else:
    raise ValueError(f"ivim data does not have expected number of dimensions {dtd_gamma.shape}.")
# %%
print(dtd_gamma.shape)
if True:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15,3))
    for i in range(5):
        plt.subplot(1,5,i+1)
        plt.imshow(np.rot90(dtd_gamma[...,(i+3)*2]), cmap='gray'), plt.axis('off')
        plt.title("b = {} s/mm$^2$".format(bvals[(i+3)*2]), fontsize=14, fontweight='bold')
        plt.colorbar()

    plt.show()
# %%
tmp_list = [mean_diff, var_iso, var_aniso]
WMmask, GMmask, CSFmask, BGmask, WMH_mask1, WMH_mask2, WMH_mask3 = masks
masks = [WMmask, GMmask, BGmask, WMH_mask1, WMH_mask2, WMH_mask3]
# WMmask.shape, WMmask.max()
# GMmask.shape, GMmask.max()
# BGmask.shape, BGmask.max()
# WMH_mask1.shape, WMH_mask1.max(), plt.hist(WMH_mask1.ravel(), bins=2), plt.imshow(WMH_mask1)
# WMH_mask2.shape, WMH_mask2.max()
# WMH_mask3.shape, WMH_mask3.max()

# %%
composite_ivim = add_phase(args, dtd_gamma,
                            show=True)  # 164 x 164 x 15 - but now with different phase for each b-value.
print(composite_ivim.shape)

composite_ivim_b_delta_1 = add_phase(args, ivim_b_delta_1, show=True)
print(composite_ivim_b_delta_1.shape)

# %%
composite_ivim = add_noise(composite_ivim, 20, return_img=False)
composite_ivim_b_delta_1 = add_noise(composite_ivim_b_delta_1, 20, return_img=False)
# %%
composite_ivim_sens, sens_maps = add_sens_maps(args, composite_ivim, show=True)  # add_sens_maps... 164x164x16x15
sens_maps_expand = np.expand_dims(sens_maps, axis=2)

composite_ivim_sens_b_delta_1, sens_maps_b_delta_1 = add_sens_maps(args, composite_ivim_b_delta_1, show=True)
sens_maps_b_delta_1_expand = np.expand_dims(sens_maps_b_delta_1, axis=2)
# %%
def get_fft(args, composite_ivim_sens, show=True):

    fft_ivim = bart(1, 'fft -u 3', composite_ivim_sens)
    fft_ivim[:, ::2] = 0 # undersample R=2
    if show:
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        fig2, axes2 = plt.subplots(4, 4, figsize=(10, 10))
        for row in range(4):
            for col in range(4):
                im1 = axes[row, col].imshow(np.rot90(abs(fft_ivim[:,:, row + col, 9])**.2), cmap='gray')
                axes[row, col].set_xticks([]), axes[row, col].set_yticks([])

                im2 = axes2[row, col].imshow(np.rot90(abs(fft_ivim[:, :, row + col, 12])**.2), cmap='gray')
                axes2[row, col].set_xticks([]), axes2[row, col].set_yticks([])

                im1.set_clim(0, 5)
                im2.set_clim(0,5)

        plt.show()
    return fft_ivim
fft_ivim = np.expand_dims(get_fft(args, composite_ivim_sens, show=True), axis=2)  # 164x164x1x16x15
fft_ivim_b_delta_1 = np.expand_dims(get_fft(args, composite_ivim_sens_b_delta_1, show=True), axis=2)
# %% sens_maps.shape
sens_maps = np.expand_dims(sens_maps, axis=2)

# %% sense_prelim.shape
# sense_prelim = get_initial_sens(fft_ivim, sens_maps)  # 164x164x15
sense_prelim = np.zeros((1,1,15))
# %%
print(sense_prelim.shape)
print(fft_ivim.shape)
print(sens_maps.shape)
# %%
fig, axes = plt.subplots(3, sense_prelim.shape[2] // 3, figsize=(15, 5))
ax = axes.ravel()
for i in range(sense_prelim.shape[2]):
    ax[i].imshow(np.abs(sense_prelim[:, :, i]), cmap='gray')
    ax[i].set_title(f'Sensitivity Map Magnitude - Coil {i+1}')
    ax[i].axis('off')
    plt.colorbar(ax[i].images[0], ax=ax[i])
plt.show()
# %%
x_dim = fft_ivim.shape[0]
y_dim = fft_ivim.shape[1]
num_bvals = fft_ivim.shape[-1]
sens_prelim_fix = np.zeros((x_dim, y_dim, num_bvals), dtype=np.complex128)
sens_prelim_b_delta_1_fix = np.zeros_like(sens_prelim_fix)
print(sens_prelim_fix[..., 0].shape)
print(fft_ivim[..., 0].shape)
print(sens_maps_expand.shape)

for i in range(num_bvals):
    # sens_prelim_fix[..., i] = bart(1, 'pics -S -l2 -r0.001 -i 10', fft_ivim[...,i], sens_maps)
    sens_prelim_fix[..., i] = bart(1, 'pics -S -l2 -r0.001 -i 10', fft_ivim[...,i], sens_maps_expand)
    sens_prelim_b_delta_1_fix[..., i] = bart(1, 'pics -S -l2 -r0.001 -i 10', fft_ivim_b_delta_1[...,i], sens_maps_b_delta_1_expand)


def show_demo(x):
    if x.ndim == 4:
        x = x[:, :, 0, 0]
    elif x.ndim == 3:
        x = x[:, :, 0]
    plt.imshow(np.abs(x), cmap='gray')
    plt.axis('off')
    plt.colorbar()
    plt.show()


show_demo(sens_prelim_fix)
# %%
def show_15_bvals(x):
    print(x.shape)
    fix, axes = plt.subplots(3, 5, figsize=(15, 9))
    axes_flat = axes.ravel()
    last_im = None
    for i, ax in enumerate(axes_flat):
        last_im = ax.imshow(np.abs(x[:, :, i]), cmap='gray')
        ax.set_title(f'Prelim Sens Map Magnitude - Bval {bvals[i]}')
        ax.axis('off')
    # Add a single shared colorbar for the grid
    if last_im is not None:
        cax = fix.add_axes([axes[-1, -1].get_position().x1 + 0.02,
                           axes[-1, -1].get_position().y0,
                           0.01,
                           axes[0, -1].get_position().y1 - axes[-1, -1].get_position().y0])
        fix.colorbar(last_im, cax=cax)
    plt.tight_layout()
    plt.show()
show_15_bvals(sens_prelim_fix)
# %%
show_15_bvals(sens_prelim_b_delta_1_fix)
# %%
sens_prelim_fix2 = np.zeros_like(sens_prelim_fix)
for i in range(num_bvals):
    sens_prelim_fix2[..., i] = bart(1, 'pics -e -S -l2 -r0.001 -i 10', fft_ivim[...,i], sens_maps_expand)
# %%
show_15_bvals(sens_prelim_fix2)
show_15_bvals(composite_ivim)

# # %%
# sens_prelim_fix3 = np.zeros_like(sens_prelim_fix)
# for i in range(num_bvals):
#     sens_prelim_fix3[..., i] = bart(1, 'pics -e -d 5 -i 100 -S -R L:3:3:0.001 -R W:3:0:0.001', fft_ivim[...,i], sens_maps_expand)
# # %%
# show_15_bvals(sens_prelim_fix3)
# # %%
show = True
# if show:
#     import matplotlib.pyplot as plt

#     fig, axes = plt.subplots(3, 5, figsize=(15, 9))
#     axes_flat = axes.ravel()
#     for i, ax in enumerate(axes_flat):
#         ax.imshow(np.abs(sens_prelim_fix2[:, :, i]), cmap='gray')
#         ax.set_title(f'Prelim Sens Map Magnitude - Bval {bvals[i]}')
#         ax.axis('off')
#     plt.tight_layout()
#     plt.show()

# %%
sense_prelim = sens_prelim_fix2
sense_prelim_b_delta_1 = sens_prelim_b_delta_1_fix
# %%
print(sense_prelim[:, 0, 0])
if show:
    plt.hist(np.abs(sense_prelim).ravel(), bins=50)
# %%
phase_removal = lowres_phaseremoval(sense_prelim)
print(phase_removal.shape)
show_15_bvals(phase_removal)

phase_removal_b_delta_1 = lowres_phaseremoval(sense_prelim_b_delta_1)
print(phase_removal_b_delta_1.shape)
show_15_bvals(phase_removal_b_delta_1)
# %%
composite_sens, _ = get_composite_sens(sense_prelim, sens_maps_expand, visualize="True")  # 164 x 164 x 16 x 15 x 1
print(composite_sens.shape)
show_15_bvals(composite_sens[:,:,5,:, 0])
show_15_bvals(composite_sens[:,:,:,6, 0])
composite_sens = np.expand_dims(np.transpose(composite_sens, (0, 1, 4, 2, 3)), axis=4)
print(composite_sens.shape)

composite_sens_b_delta_1, _ = get_composite_sens(sense_prelim_b_delta_1, sens_maps_b_delta_1_expand, visualize="True")  # 164 x 164 x 16 x 15 x 1
composite_sens_b_delta_1 = np.expand_dims(np.transpose(composite_sens_b_delta_1, (0, 1, 4, 2, 3)), axis=4)
# %% Load Basis
standard_file_dir = '../Standard_Files_dtd'

def load_bases(base_dir, base_name, names):
    """Load multiple bases from a directory given a list of suffix numbers."""
    loaded = {}
    for n in names:
        loaded[n] = cfl.readcfl(os.path.join(base_dir, f'{base_name}_{n}'))
    return loaded

def plot_bases_grid(bases_dict, bvals, figsize=(10, 8)):
    import matplotlib.pyplot as plt
    names = list(bases_dict.keys())
    rows = int(np.ceil(len(names) / 2))
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_2d(axes)
    for idx, name in enumerate(names):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        ax.plot(bvals, bases_dict[name].squeeze()[:, :], label=f"IVIM Basis {name}")
        ax.set_title(f"IVIM Basis {name}")
        ax.set_xlabel('b-values')
        ax.set_ylabel('Signal Intensity')
        ax.legend()
    # hide any unused subplots
    for idx in range(len(names), rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis('off')
    plt.tight_layout()

basis_numbers = [2, 3, 4, 5]
bases = load_bases(standard_file_dir, "ivim_basis", basis_numbers)
for n, arr in bases.items():
    print(f"basis{n} shape: {arr.shape}")
plot_bases_grid(bases, bvals)
# todo tmp
basis2, basis3, basis4, basis5 = bases[2], bases[3], bases[4], bases[5]
# %%
bases_dtd_bdelta1 = load_bases(standard_file_dir, "dtd_bdelta1_basis", basis_numbers)
for n, arr in bases_dtd_bdelta1.items():
    print(f"dtd_bdelta1_basis{n} shape: {arr.shape}")
plot_bases_grid(bases_dtd_bdelta1, bvals)

# %%
print(fft_ivim.shape)
# %%
# recon_fmac2.shape
# 0      1      2      3      4      5      6      7      8      9    10   11   12   13   14
# RO     PH1    PH2    CHA    MAPS   TE     COEFF  COEFF2 ITER
# 164    164    1      16     1      15
fft_ivim = np.expand_dims(fft_ivim, axis=4)
print(fft_ivim.shape)
# %%
fft_ivim_b_delta_1_expand = np.expand_dims(fft_ivim_b_delta_1, axis=4)
# %%
def print_info(x, name="Variable"):
    print(f"{name} shape: {x.shape}, dtype: {x.dtype}")
print_info(fft_ivim, "fft_ivim")
print_info(composite_sens, "composite_sens")
print_info(basis2, "basis2")

# %%
def llr_recon_with_retry(
    fft_ivim,
    composite_sens,
    basis,
    use_basis=True,
    R=2,
    lambda1=0.001,
    lambda2=0.001,
    max_retries=5,
    delay_seconds=0.5,
):
    """
    Attempt llr_recon up to max_retries times before failing.
    """
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            return llr_recon(
                fft_ivim,
                composite_sens,
                basis,
                use_basis=use_basis,
                R=R,
                lambda1=lambda1,
                lambda2=lambda2,
            )
        except Exception as e:
            last_exc = e
            print(f"llr_recon attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(delay_seconds)
    raise last_exc

# %% helper: plotting recon vs ivim with shared colors
def plot_recon_vs_ivim(
    recon,
    ivim,
    bvals,
    points,
    recon_label="FMac",
    ivim_scale=1.0,
    figure_kwargs=None,
):
    import matplotlib.pyplot as plt
    arr = np.real(recon.squeeze())
    if figure_kwargs is None:
        figure_kwargs = {}
    plt.figure(**figure_kwargs)
    colors = plt.cm.tab10(np.linspace(0, 1, len(points)))
    legend_entries = []
    for (i, j), c in zip(points, colors):
        plt.plot(bvals, arr[i, j, :], marker="o", linestyle="-", color=c)
        plt.plot(bvals, ivim[i, j, :], marker="x", linestyle="--", color=c)
        legend_entries.append(f"{recon_label} ({i},{j})")
        legend_entries.append(f"dtd ({i},{j})")
    plt.xlabel("b-values (s/mm$^2$)")
    plt.ylabel("Signal Intensity")
    plt.title(f"{recon_label} Reconstructed Signal vs. Ground Truth IVIM Signal")
    plt.legend(legend_entries)

def plot_points_on_image(img, points, colors=None, title="Points on image"):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(np.abs(img), cmap='gray')
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(points)))
    for (i, j), c in zip(points, colors):
        plt.plot(j, i, marker='o', color=c, markersize=6)
        plt.text(j + 2, i - 2, f"({i},{j})", color=c, fontsize=8)
    plt.title(title)
    plt.axis('off')

# %%
recon, recon_fmac2 = llr_recon_with_retry(
    fft_ivim,
    composite_sens,
    basis2,
    use_basis=True,
    R=2,
    lambda1=0.001,
    lambda2=0.001,
)
# %% save recon fmac2 to mat
import scipy.io as sio
sio.savemat(os.path.join(args.outdir + '/phan_cyan', 'recon_fmac2.mat'), {'recon_fmac2': recon_fmac2})
# %% plot recon vs ivim using helper
plot_recon_vs_ivim(recon_fmac2, dtd_gamma, bvals, points, recon_label="2 basis", ivim_scale=1.0)
# visualize points on the phantom image (use b0 ivim magnitude)
plot_points_on_image(dtd_gamma[:, :, 6], points, title="Selected points on phantom (b0)")
plot_points_on_image(recon_fmac2.squeeze()[:, :, 6], points, title="Selected points on recon_fmac2 (b0)")
# can you plot the points on the phantom image?

# %%
print_info(recon, "recon")
print_info(recon_fmac2, "recon_fmac2")
print_info(recon_fmac2.squeeze(), "recon_fmac2.squeeze()")
plt.imshow(abs(recon.squeeze()[:, :, 0]), cmap='gray')
show_15_bvals(np.real(recon_fmac2.squeeze()))
# %%
recon, recon_fmac3 = llr_recon_with_retry(
    fft_ivim,
    composite_sens,
    basis3,
    use_basis=True,
    R=2,
    lambda1=0.001,
    lambda2=0.001,
)
#%%
plot_recon_vs_ivim(recon_fmac3, dtd_gamma, bvals, points, recon_label="3 basis", ivim_scale=1.0)
# %%
recon, recon_fmac4 = llr_recon_with_retry(
    fft_ivim,
    composite_sens,
    basis4,
    use_basis=True,
    R=2,
    lambda1=0.001,
    lambda2=0.001,
)

recon, recon_fmac5 = llr_recon_with_retry(
    fft_ivim,
    composite_sens,
    basis5,
    use_basis=True,
    R=2,
    lambda1=0.001,
    lambda2=0.001,
)
# %%
recon_b_delta_1, recon_fmac2_b_delta_1 = llr_recon_with_retry(
    fft_ivim_b_delta_1_expand,
    composite_sens_b_delta_1,
    bases_dtd_bdelta1[2],
    use_basis=True,
    R=2,
    lambda1=0.001,
    lambda2=0.001,
)
recon_b_delta_1, recon_fmac3_b_delta_1 = llr_recon_with_retry(
    fft_ivim_b_delta_1_expand,
    composite_sens_b_delta_1,
    bases_dtd_bdelta1[3],
    use_basis=True,
    R=2,
    lambda1=0.001,
    lambda2=0.001,
)
# %%
recon_b_delta_1, recon_fmac4_b_delta_1 = llr_recon_with_retry(
    fft_ivim_b_delta_1_expand,
    composite_sens_b_delta_1,
    bases_dtd_bdelta1[4],
    use_basis=True,
    R=2,
    lambda1=0.001,
    lambda2=0.001,
)
recon_b_delta_1, recon_fmac5_b_delta_1 = llr_recon_with_retry(
    fft_ivim_b_delta_1_expand,
    composite_sens_b_delta_1,
    bases_dtd_bdelta1[5],
    use_basis=True,
    R=2,
    lambda1=0.001,
    lambda2=0.001,
)
# %%
sio.savemat(os.path.join(args.outdir + '/phan_cyan', 'recon_all.mat'), {
    'recon': recon,   
    'recon_fmac2': recon_fmac2,
    'recon_fmac3': recon_fmac3,
    'recon_fmac4': recon_fmac4,
    'recon_fmac5': recon_fmac5,
    'recon_b_delta_1': recon_b_delta_1,
    'recon_fmac2_b_delta_1': recon_fmac2_b_delta_1,
    'recon_fmac3_b_delta_1': recon_fmac3_b_delta_1,
    'recon_fmac4_b_delta_1': recon_fmac4_b_delta_1,
    'recon_fmac5_b_delta_1': recon_fmac5_b_delta_1,
}) 
 
try:
    save_nifti(np.real(recon_fmac2.squeeze()[:,:,np.newaxis,:]), os.path.join(outdir_nifti, 'recon_fmac2.nii.gz'))
    save_nifti(np.real(recon_fmac3.squeeze()[:,:,np.newaxis,:]), os.path.join(outdir_nifti, 'recon_fmac3.nii.gz'))
    save_nifti(np.real(recon_fmac4.squeeze()[:,:,np.newaxis,:]), os.path.join(outdir_nifti, 'recon_fmac4.nii.gz'))
    save_nifti(np.real(recon_fmac5.squeeze()[:,:,np.newaxis,:]), os.path.join(outdir_nifti, 'recon_fmac5.nii.gz'))

    save_nifti(np.real(recon_fmac2_b_delta_1.squeeze()[:,:,np.newaxis,:]), os.path.join(outdir_nifti, 'recon_fmac2_b_delta_1.nii.gz'))
    save_nifti(np.real(recon_fmac3_b_delta_1.squeeze()[:,:,np.newaxis,:]), os.path.join(outdir_nifti, 'recon_fmac3_b_delta_1.nii.gz'))
    save_nifti(np.real(recon_fmac4_b_delta_1.squeeze()[:,:,np.newaxis,:]), os.path.join(outdir_nifti, 'recon_fmac4_b_delta_1.nii.gz'))
    save_nifti(np.real(recon_fmac5_b_delta_1.squeeze()[:,:,np.newaxis,:]), os.path.join(outdir_nifti, 'recon_fmac5_b_delta_1.nii.gz'))
except Exception as e:
    print(f"Failed to save NIfTI files: {e}")
#%%
plot_recon_vs_ivim(recon_fmac4, dtd_gamma, bvals, points, recon_label="4 basis", ivim_scale=1.0)
plot_recon_vs_ivim(recon_fmac5, dtd_gamma, bvals, points, recon_label="5 basis", ivim_scale=1.0)
# %%
show_15_bvals(np.real(recon_fmac3.squeeze()))
show_15_bvals(np.real(recon_fmac4.squeeze()))
show_15_bvals(np.real(recon_fmac5.squeeze()))
# %%
plot_recon_vs_ivim(recon_fmac2_b_delta_1, ivim_b_delta_1, bvals, points, recon_label="2 basis b_delta_1", ivim_scale=1.0)
plot_recon_vs_ivim(recon_fmac3_b_delta_1, ivim_b_delta_1, bvals, points, recon_label="3 basis b_delta_1", ivim_scale=1.0)
show_15_bvals(np.real(recon_fmac2_b_delta_1.squeeze()))
show_15_bvals(np.real(recon_fmac3_b_delta_1.squeeze()))
# %%
plot_recon_vs_ivim(recon_fmac4_b_delta_1, ivim_b_delta_1, bvals, points, recon_label="4 basis b_delta_1", ivim_scale=1.0)
plot_recon_vs_ivim(recon_fmac5_b_delta_1, ivim_b_delta_1, bvals, points, recon_label="5 basis b_delta_1", ivim_scale=1.0)
# %%
recon_fmac2 = np.real(recon_fmac2.squeeze())
# %%
recon_fmac3 = np.real(recon_fmac3.squeeze())
recon_fmac4 = np.real(recon_fmac4.squeeze())
recon_fmac5 = np.real(recon_fmac5.squeeze())

# %%
filtered = recon_fmac2
# %%
filtered3 = recon_fmac3
filtered4 = recon_fmac4
filtered5 = recon_fmac5
# %%
filtered_sens = abs(sense_prelim)
filtered_sens_real = np.real(phase_removal)

_, mask = median_otsu(np.abs(recon_fmac2.squeeze()[..., 0]), median_radius=6, numpass=2)
mask = mask.astype(np.uint8)
start = datetime.datetime.now()

with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    tmp = filtered
    args2 = [(tmp[i, j] / tmp[i, j, 0], mask[i, j]) for i in range(tmp.shape[0])
                for j in range(tmp.shape[1])]
    results = pool.starmap(ivim_fit_segmented, args2)

    results = np.rot90(np.asarray(results).reshape((164, 164, 3)))
    Dtsub2, Dpsub2, Fpsub2 = results[..., 0], results[..., 1], results[..., 2]

    tmp = filtered3
    args2 = [(tmp[i, j] / tmp[i, j, 0], mask[i, j]) for i in range(tmp.shape[0])
                for j in range(tmp.shape[1])]
    results = pool.starmap(ivim_fit_segmented, args2)
    results = np.rot90(np.asarray(results).reshape((164, 164, 3)))
    Dtsub3, Dpsub3, Fpsub3 = results[..., 0], results[..., 1], results[..., 2]

    tmp = filtered4
    args2 = [(tmp[i, j] / tmp[i, j, 0], mask[i, j]) for i in range(tmp.shape[0])
                for j in range(tmp.shape[1])]
    results = pool.starmap(ivim_fit_segmented, args2)
    results = np.rot90(np.asarray(results).reshape((164, 164, 3)))
    Dtsub4, Dpsub4, Fpsub4 = results[..., 0], results[..., 1], results[..., 2]

    tmp = filtered5
    args2 = [(tmp[i, j] / tmp[i, j, 0], mask[i, j]) for i in range(tmp.shape[0])
                for j in range(tmp.shape[1])]
    results = pool.starmap(ivim_fit_segmented, args2)
    results = np.rot90(np.asarray(results).reshape((164, 164, 3)))
    Dtsub5, Dpsub5, Fpsub5 = results[..., 0], results[..., 1], results[..., 2]

    tmp = filtered_sens
    args2 = [(tmp[i, j] / tmp[i, j, 0], mask[i, j]) for i in range(tmp.shape[0])
                for j in range(tmp.shape[1])]
    results = pool.starmap(ivim_fit_segmented, args2)
    results = np.rot90(np.asarray(results).reshape((164, 164, 3)))
    Dtsens, Dpsens, Fpsens = results[..., 0], results[..., 1], results[..., 2]

    tmp = filtered_sens_real
    args2 = [(tmp[i, j] / tmp[i, j, 0], mask[i, j]) for i in range(tmp.shape[0])
                for j in range(tmp.shape[1])]
    results = pool.starmap(ivim_fit_segmented, args2)
    results = np.rot90(np.asarray(results).reshape((164, 164, 3)))
    Dtlowres, Dplowres, Fplowres = results[..., 0], results[..., 1], results[..., 2]

print(datetime.datetime.now() - start)
CSFmask = np.rot90(mean_diff.copy())
CSFmask[CSFmask < 0.0025] = 0
CSFmask[CSFmask >= 0.0025] = 1
CSFmask[Dtsub2 > 0.003] = 1

Dtsub2, Fpsub2, Dpsub2 = Dtsub2 * abs((1 - CSFmask)), Fpsub2 * abs((1 - CSFmask)), Dpsub2 * abs((1 - CSFmask))
Dtsub3, Fpsub3, Dpsub3 = Dtsub3 * abs((1 - CSFmask)), Fpsub3 * abs((1 - CSFmask)), Dpsub3 * abs((1 - CSFmask))
Dtsub4, Fpsub4, Dpsub4 = Dtsub4 * abs((1 - CSFmask)), Fpsub4 * abs((1 - CSFmask)), Dpsub4 * abs((1 - CSFmask))
Dtsub5, Fpsub5, Dpsub5 = Dtsub5 * abs((1 - CSFmask)), Fpsub5 * abs((1 - CSFmask)), Dpsub5 * abs((1 - CSFmask))
Dtsens, Fpsens, Dpsens = Dtsens * abs((1 - CSFmask)), Fpsens * abs((1 - CSFmask)), Dpsens * abs((1 - CSFmask))
Dtlowres, Fplowres, Dplowres = Dtlowres * abs((1 - CSFmask)), Fplowres * abs((1 - CSFmask)), Dplowres * abs(
    (1 - CSFmask))
mean_diff, var_iso, var_aniso = np.rot90(mean_diff) * abs((1 - CSFmask)), np.rot90(var_iso) * abs((1 - CSFmask)), np.rot90(var_aniso) * abs((1 - CSFmask))

GTs = [mean_diff, var_iso, var_aniso]
mags = [Dtsens, Fpsens, Dpsens]
sub2 = [Dtsub2, Fpsub2, Dpsub2]
sub3 = [Dtsub3, Fpsub4, Dpsub3]
sub4 = [Dtsub4, Fpsub5, Dpsub4]
sub5 = [Dtsub5, Fpsub5, Dpsub5]
lowres = [Dtlowres, Fplowres, Dplowres]

# %%

def pub_figure(basis2:list, basis3:list, basis4:list,  basis5:list, lowres:list, mags:list, GTs:list, dki=False):
    from Phantom_utils import calc_rmse

    fig, axes = plt.subplots(3, 7, figsize=(10, 10))
    labels = ['MD', 'V_I', 'V_A']
    titles=['Ground-Truth', '2 Bases', '3 Bases', '4 Bases', '5 Bases', 'Phase Removal', 'Conventional']
    cmap='inferno'

    for row in range(axes.shape[0]):
        clim_ = [(0.0003, 0.003), (0.0, 2.5), (0.0, 0.99), (0, 1.5)]
        for col in range(axes.shape[1]):
            images = GTs if col == 0 else basis2 if col == 1 else basis3 if col == 2 else \
                basis4 if col == 3 else basis5 if col == 4 else lowres if col == 5 else mags
            im = axes[row, col].imshow(images[row],
                                       cmap=cmap)
            # im.set_clim(clim_[row])

            if col != 0:
                nrmse, ssim = calc_rmse(images[row], GTs[row], nrsme=True)

                text_to_add = "NRMSE: {}%\nSSIM: {}".format(nrmse, ssim)
                axes[row, col].text(82, 25, text_to_add, color='white', fontweight='bold', ha='center')
            if row == 0:
                axes[row, col].set_title("{}".format(titles[col]),
                                         fontweight='bold')
            if col == 0:
                axes[row,col].set_ylabel(labels[row], fontweight='bold')

            axes[row,col].set_xticks([]), axes[row,col].set_yticks([])

            if col == 6:
                cbar_axes = fig.add_axes([axes[row, col].get_position().x1 + 0.02, axes[row, col].get_position().y0,
                                          0.01, axes[row,col].get_position().y1 - axes[row, col].get_position().y0])

                cbar = plt.colorbar(im, cax=cbar_axes)

    plt.subplots_adjust(wspace=0, hspace=0)

    return None

pub_figure(sub2, sub3, sub4, sub5, lowres, mags, GTs)
bland_altman_image(sub2, sub3, sub4, sub5, mags, GTs, masks)

# %%


def load_dtd_gamma(directory, names=("dtd_gamma_Vi", "dtd_gamma_Va", "dtd_gamma_MD")):
	"""Load selected dtd_gamma_* nifti files into a dict."""
	data = {}
	for name in names:
		candidates = [f"{name}.nii", f"{name}.nii.gz"]
		for fname in candidates:
			path = os.path.join(directory, fname)
			if os.path.exists(path):
				img = nib.load(path)
				data[name] = img.get_fdata()
				break
	return data


directory = '/data/users/cyang/repos/HIFIVIM/Phantom/phan_cyan/processed/brain'
data_sel = load_dtd_gamma(directory)
print("selected dtd_gamma loaded:")
for k, v in data_sel.items():
    print(f"{k}: shape={v.shape}, dtype={v.dtype}")

sub2_dt_gamma = [data_sel['dtd_gamma_MD'], data_sel['dtd_gamma_Vi'], data_sel['dtd_gamma_Va']]

GTs = [mean_diff, var_iso, var_aniso]  # define GTs again for clarity
pub_figure(sub2_dt_gamma, GTs, GTs, GTs, GTs, GTs, GTs)
# %%
print(f"mean diffusivity shape: {mean_diff.shape}")
print("mean diffusivity at selected points:")
for x, y in points:
    print(f"mean diffusivity: GT: {mean_diff[x, y]}, fit: {sub2_dt_gamma[0][x, y]}")
    print(f"var isotropic:    GT: {var_iso[x, y]}, fit: {sub2_dt_gamma[1][x, y]}")
    print(f"var anisotropic:  GT: {var_aniso[x, y]}, fit: {sub2_dt_gamma[2][x, y]}")
# %%
