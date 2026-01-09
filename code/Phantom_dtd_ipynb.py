# %% run ../../../src/bart/startup.py
import os
import numpy as np
import multiprocessing
import datetime
from argparse import ArgumentParser
import cfl
from Phantom_utils import make_phantom, add_phase, add_sens_maps, get_fft, \
    pub_figure, bland_altman_image
from utils import add_noise, get_initial_sens, lowres_phaseremoval, \
    get_composite_sens, ivim_fit_segmented, llr_recon, median_otsu


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

bvals = [0, 5, 7, 10, 15, 20, 30, 40, 50, 60, 100, 200, 400, 700, 1000]

# %% make phantom
Dt, Fp, Dp, ivim, masks = make_phantom(args, show=True)  # 164 x 164 x 15 for ivim
ivim = ivim * 1000
# %%
print(ivim.shape)
if True:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15,3))
    for i in range(5):
        plt.subplot(1,5,i+1)
        plt.imshow(np.rot90(ivim[...,(i+3)*2]), cmap='gray'), plt.clim(0.2,1), plt.axis('off')
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
print(fft_ivim.shape)
print(sens_maps.shape)
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
standard_file_dir = '../Standard_Files'
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
def plot_basis(ax, title="", basis=None):
    ax.plot(basis.squeeze()[:, :], label=title)
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
pub_figure(sub2, sub3, sub4, sub5, lowres, mags, GTs)
bland_altman_image(sub2, sub3, sub4, sub5, mags, GTs, masks)

# %%
