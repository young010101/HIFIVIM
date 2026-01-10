# %% run ../../../src/bart/startup.py
import os
import numpy as np
import multiprocessing
import datetime
from argparse import ArgumentParser
import cfl
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
        b-values (s/mm^2)
    s0 : float
        Baseline signal
    d_iso : float
        Mean diffusivity (MD)
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

    bvals = np.asarray(bvals) * 1e-3  # convert to s/um^2

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

    Dt, Fp, Dp = np.zeros_like(phantom), np.zeros_like(phantom), np.zeros_like(phantom)
    phantom_data = np.zeros((phantom.shape[0], phantom.shape[1], bvals.shape[0]))
    phantom_data_b_delta_1 = np.zeros_like(phantom_data)
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


    for i in range(phantom.shape[0]):
        for j in range(phantom.shape[1]):
            if WMH_mask1[i,j] == 1:
                Dt[i, j] = 0.0012
                Fp[i, j] = 0.96
                Dp[i, j] = 0.925
            elif WMH_mask2[i, j] == 1:
                Dt[i, j] = 0.0014
                Fp[i, j] = 0.97
                Dp[i, j] = 0.928
            elif WMH_mask3[i, j] == 1:
                Dt[i, j] = 0.0013
                Fp[i, j] = 0.965
                Dp[i, j] = 0.927
            else:
                if j < 101 and j > 65 and i > 62 and i < 103:
                    Dt[i, j] = 0.0006 if phantom[i, j] > 2.5 else 0.0005 if phantom[i, j] > 1.7 else 0.003 if phantom[
                                                                                                                  i, j] > 0 else 0
                    Fp[i, j] = 0.7 if phantom[i, j] > 2.5 else 0.6 if phantom[i, j] > 1.7 else 2.5 if phantom[
                                                                                                            i, j] > 0 else 0
                    Dp[i, j] = 0.45 if phantom[i, j] > 2.5 else 0.55 if phantom[i, j] > 1.7 else 0.2 if phantom[ i, j] > 0 else 0
                else:
                    Dt[i, j] = 0.0006 if phantom[i, j] > 2.35 else 0.0009 if phantom[i, j] > 1.7 else 0.003 if phantom[
                                                                                                                  i, j] > 0 else 0
                    Fp[i, j] = 0.7 if phantom[i, j] > 2.35 else 1.4 if phantom[i, j] > 1.7 else 2 if phantom[
                                                                                                            i, j] > 0 else 0
                    Dp[i, j] = 0.45 if phantom[i, j] > 2.35 else 0.3 if phantom[i, j] > 1.7 else 0.2 if phantom[
                                                                                                              i, j] > 0 else 0
            if Dt[i,j] == 0.0006:
                WMmask[i,j] = 1
            elif Dt[i, j] == 0.0005:
                BGmask[i, j] = 1
            elif Dt[i, j] == 0.0009:
                GMmask[i, j] = 1
            elif Dt[i, j] == 0.003:
                CSFmask[i, j] = 1


            phantom_data[i,j] = ivim_model(10, Fp[i,j], Dt[i,j], Dp[i,j], bvals)
            phantom_data[i,j][Dt[i,j] == 0] =0
            phantom_data_b_delta_1[i,j] = ivim_model(10, Fp[i,j], Dt[i,j], Dp[i,j], bvals, b_delta=np.ones_like(bvals))
            phantom_data_b_delta_1[i,j][Dt[i,j] == 0] =0


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

        im = axes[1].imshow(np.rot90(Dt), cmap=cmap)
        axes[1].set_xticks([]), axes[1].set_yticks([])
        axes[1].set_title('MD', fontsize=16, fontweight='bold'), im.set_clim(0.0003, 0.0015)
        cax = fig.add_axes([axes[1].get_position().x1 + 0.005,
                            axes[1].get_position().y0, 0.01, axes[1].get_position().height])
        cbar = plt.colorbar(axes[1].images[0], cax=cax)


        im = axes[2].imshow(np.rot90(Fp), cmap=cmap)
        axes[2].set_xticks([]), axes[2].set_yticks([])
        axes[2].set_title('$V_I$', fontsize=16, fontweight='bold'), im.set_clim()
        cax = fig.add_axes([axes[2].get_position().x1 + 0.005,
                            axes[2].get_position().y0, 0.01, axes[2].get_position().height])
        cbar = plt.colorbar(axes[2].images[0], cax=cax)

        im = axes[3].imshow(np.rot90(Dp), cmap=cmap)
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
            plt.xlabel('b-values (s/mm$^2$)', fontsize=14, fontweight='bold')
            plt.ylabel('Signal Intensity', fontsize=14, fontweight='bold')
            plt.title('Signal Curve at ({},{})'.format(i,j), fontsize=14, fontweight='bold')
            plt.grid()
        

    return Dt, Fp, Dp, phantom_data, phantom_data_b_delta_1, (np.rot90(WMmask), np.rot90(GMmask), np.rot90(CSFmask),
                                      np.rot90(BGmask), np.rot90(WMH_mask1), np.rot90(WMH_mask2), np.rot90(WMH_mask3))


Dt, Fp, Dp, ivim, ivim_b_delta_1, masks = make_phantom(args, show=True)  # 164 x 164 x 15 for ivim
ivim = ivim
# %%
print(ivim.shape)
if True:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15,3))
    for i in range(5):
        plt.subplot(1,5,i+1)
        plt.imshow(np.rot90(ivim[...,(i+3)*2]), cmap='gray'), plt.clim(), plt.axis('off')
        plt.title("b = {} s/mm$^2$".format(bvals[(i+3)*2]), fontsize=14, fontweight='bold')
        plt.colorbar()

    plt.show()
# %%
tmp_list = [Dt, Fp, Dp]
WMmask, GMmask, CSFmask, BGmask, WMH_mask1, WMH_mask2, WMH_mask3 = masks
masks = [WMmask, GMmask, BGmask, WMH_mask1, WMH_mask2, WMH_mask3]
# WMmask.shape, WMmask.max()
# GMmask.shape, GMmask.max()
# BGmask.shape, BGmask.max()
# WMH_mask1.shape, WMH_mask1.max(), plt.hist(WMH_mask1.ravel(), bins=2), plt.imshow(WMH_mask1)
# WMH_mask2.shape, WMH_mask2.max()
# WMH_mask3.shape, WMH_mask3.max()

# %%
composite_ivim = add_phase(args, ivim,
                            show=True)  # 164 x 164 x 15 - but now with different phase for each b-value.
print(composite_ivim.shape)

# %%
composite_ivim = add_noise(composite_ivim, 20, return_img=False)
# %%
composite_ivim_sens, sens_maps = add_sens_maps(args, composite_ivim, show=True)  # add_sens_maps... 164x164x16x15
sens_maps_expand = np.expand_dims(sens_maps, axis=2)
# %%
fft_ivim = np.expand_dims(get_fft(args, composite_ivim_sens, show=True), axis=2)  # 164x164x1x16x15
# %% sens_maps.shape
sens_maps = np.expand_dims(sens_maps, axis=2)

# %% sense_prelim.shape
sense_prelim = get_initial_sens(fft_ivim, sens_maps)  # 164x164x15
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
print(sens_prelim_fix[..., 0].shape)
print(fft_ivim[..., 0].shape)
print(sens_maps_expand.shape)

for i in range(num_bvals):
    # sens_prelim_fix[..., i] = bart(1, 'pics -S -l2 -r0.001 -i 10', fft_ivim[...,i], sens_maps)
    sens_prelim_fix[..., i] = bart(1, 'pics -S -l2 -r0.001 -i 10', fft_ivim[...,i], sens_maps_expand)


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
    for i, ax in enumerate(axes_flat):
        ax.imshow(np.abs(x[:, :, i]), cmap='gray')
        ax.set_title(f'Prelim Sens Map Magnitude - Bval {bvals[i]}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()
show_15_bvals(sens_prelim_fix)
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
# %%
print(sense_prelim[:, 0, 0])
if show:
    plt.hist(np.abs(sense_prelim).ravel(), bins=50)
# %%
phase_removal = lowres_phaseremoval(sense_prelim)
print(phase_removal.shape)
show_15_bvals(phase_removal)
# %%
composite_sens, _ = get_composite_sens(sense_prelim, sens_maps, visualize="True")  # 164 x 164 x 16 x 15 x 1
print(composite_sens.shape)
show_15_bvals(composite_sens[:,:,5,:, 0])
show_15_bvals(composite_sens[:,:,:,6, 0])
composite_sens = np.expand_dims(np.transpose(composite_sens, (0, 1, 4, 2, 3)), axis=4)
print(composite_sens.shape)

# %% Load Basis
standard_file_dir = '../Standard_Files_dtd'
basis2 = cfl.readcfl(standard_file_dir + '/ivim_basis_2')
basis3 = cfl.readcfl(standard_file_dir + '/ivim_basis_3')
basis4 = cfl.readcfl(standard_file_dir + '/ivim_basis_4')
basis5 = cfl.readcfl(standard_file_dir + '/ivim_basis_5')
# %%
print(basis2.shape)
print(basis3.shape)
print(basis4.shape)
print(basis5.shape)
# %%
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
def plot_basis(ax, title="", basis=None, bvals=bvals):
    ax.plot(bvals, basis.squeeze()[:, :], label=title)
    ax.set_title(title)
    ax.set_xlabel('b-values')
    ax.set_ylabel('Signal Intensity')
    ax.legend()
plot_basis(axes[0, 0], title="IVIM Basis 2", basis=basis2)
plot_basis(axes[0, 1], title="IVIM Basis 3", basis=basis3)
plot_basis(axes[1, 0], title="IVIM Basis 4", basis=basis4)
plot_basis(axes[1, 1], title="IVIM Basis 5", basis=basis5)

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
def print_info(x, name="Variable"):
    print(f"{name} shape: {x.shape}, dtype: {x.dtype}")
print_info(fft_ivim, "fft_ivim")
print_info(composite_sens, "composite_sens")
print_info(basis2, "basis2")

# %%
recon, recon_fmac2 = llr_recon(fft_ivim, composite_sens, basis2,
                                use_basis=True, R=2, lambda1=0.001, lambda2=0.001)  # check different lambdas after
# %% save recon fmac2 to mat
import scipy.io as sio
sio.savemat(os.path.join(args.outdir + '/phan_cyan', 'recon_fmac2.mat'), {'recon_fmac2': recon_fmac2})
# %% plot
recon_fmac2_squeezed = np.real(recon_fmac2.squeeze())
plt.plot(bvals, recon_fmac2_squeezed[82, 82, :], 'o-')
plt.plot(bvals, recon_fmac2_squeezed[50, 50, :], 'o-')
# plot use ivim and dtd model
plt.plot(bvals, ivim[82, 82, :] * 170, 'x--')
plt.plot(bvals, ivim[50, 50, :] * 170, 'x--')
plt.xlabel('b-values (s/mm$^2$)')
plt.ylabel('Signal Intensity')
plt.title('FMac2 Reconstructed Signal vs. Ground Truth IVIM Signal')
plt.legend(['FMac2 (82,82)', 'FMac2 (50,50)', 'IVIM (82,82)', 'IVIM (50,50)'])
# %%
print_info(recon, "recon")
print_info(recon_fmac2, "recon_fmac2")
print_info(recon_fmac2.squeeze(), "recon_fmac2.squeeze()")
plt.imshow(abs(recon.squeeze()[:, :, 0]), cmap='gray')
show_15_bvals(np.real(recon_fmac2.squeeze()))
# %%
recon, recon_fmac3 = llr_recon(fft_ivim, composite_sens, basis3,
                                use_basis=True, R=2, lambda1=0.001, lambda2=0.001)  # check different lambdas after
# %%
recon, recon_fmac4 = llr_recon(fft_ivim, composite_sens, basis4,
                                use_basis=True, R=2, lambda1=0.001, lambda2=0.001)  # check different lambdas after

recon, recon_fmac5 = llr_recon(fft_ivim, composite_sens, basis5,
                                use_basis=True, R=2, lambda1=0.001, lambda2=0.001)  # check different lambdas after
# %%
show_15_bvals(np.real(recon_fmac3.squeeze()))
show_15_bvals(np.real(recon_fmac4.squeeze()))
show_15_bvals(np.real(recon_fmac5.squeeze()))
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
CSFmask = np.rot90(Dt.copy())
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
Dt, Fp, Dp = np.rot90(Dt) * abs((1 - CSFmask)), np.rot90(Fp) * abs((1 - CSFmask)), np.rot90(Dp) * abs((1 - CSFmask))

GTs = [Dt, Fp, Dp]
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
    labels = ['D', 'f', 'D*']
    titles=['Ground-Truth', '2 Bases', '3 Bases', '4 Bases', '5 Bases', 'Phase Removal', 'Conventional']
    cmap='inferno'

    for row in range(axes.shape[0]):
        # clim = (0.0003, 0.0015) if row == 0 else (0.04, 0.25) if row == 1 else (0.02, 0.06) if row == 2 else (0, 1.5)
        for col in range(axes.shape[1]):
            images = GTs if col == 0 else basis2 if col == 1 else basis3 if col == 2 else \
                basis4 if col == 3 else basis5 if col == 4 else lowres if col == 5 else mags
            im = axes[row, col].imshow(images[row],
                                       cmap=cmap)
            # im.set_clim(clim)

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
