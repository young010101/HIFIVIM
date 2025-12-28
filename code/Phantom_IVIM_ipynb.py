# %%
import os
import numpy as np
import multiprocessing
import datetime
from argparse import ArgumentParser
import cfl
from Phantom_utils import make_phantom, add_phase, add_sens_maps, get_fft, \
    pub_figure, bland_altman_image
from utils import *


def parser():
    parse = ArgumentParser()
    parse.add_argument('--phantom_dir', type=str, default='phantom_1.0mm_normal_fuzzy')
    parse.add_argument('--outdir', type=str)
    return parse.parse_args()


# %%
args = parser()
if os.path.exists(args.outdir):
    pass
else:
    os.mkdir(args.outdir)

bvals = [0, 5, 7, 10, 15, 20, 30, 40, 50, 60, 100, 200, 400, 700, 1000]

Dt, Fp, Dp, ivim, masks = make_phantom(args, show=False)  # 164 x 164 x 15 for ivim
tmp_list = [Dt, Fp, Dp]
WMmask, GMmask, CSFmask, BGmask, WMH_mask1, WMH_mask2, WMH_mask3 = masks
masks = [WMmask, GMmask, BGmask, WMH_mask1, WMH_mask2, WMH_mask3]

composite_ivim = add_phase(args, ivim,
                            show=False)  # 164 x 164 x 15 - but now with different phase for each b-value.
composite_ivim = add_noise(composite_ivim, 20, return_img=False)
composite_ivim_sens, sens_maps = add_sens_maps(args, composite_ivim, show=False)  # add_sens_maps... 164x164x16x15
fft_ivim = np.expand_dims(get_fft(args, composite_ivim_sens, show=False), axis=2)  # 164x164x1x16x15
sens_maps = np.expand_dims(sens_maps, axis=2)

sense_prelim = get_initial_sens(fft_ivim, sens_maps)  # 164x164x15
phase_removal = lowres_phaseremoval(sense_prelim)
composite_sens, _ = get_composite_sens(sense_prelim, sens_maps, visualize="False")  # 164 x 164 x 16 x 15 x 1
composite_sens = np.expand_dims(np.transpose(composite_sens, (0, 1, 4, 2, 3)), axis=4)

# Load Basis
basis2 = cfl.readcfl('Standard_Files/ivim_basis_2')
basis3 = cfl.readcfl('Standard_Files/ivim_basis_3')
basis4 = cfl.readcfl('Standard_Files/ivim_basis_4')
basis5 = cfl.readcfl('Standard_Files/ivim_basis_5')

fft_ivim = np.expand_dims(fft_ivim, axis=4)
recon, recon_fmac2 = llr_recon(fft_ivim, composite_sens, basis2,
                                use_basis=True, R=2, lambda1=0.001, lambda2=0.001)  # check different lambdas after

recon, recon_fmac3 = llr_recon(fft_ivim, composite_sens, basis3,
                                use_basis=True, R=2, lambda1=0.001, lambda2=0.001)  # check different lambdas after

recon, recon_fmac4 = llr_recon(fft_ivim, composite_sens, basis4,
                                use_basis=True, R=2, lambda1=0.001, lambda2=0.001)  # check different lambdas after

recon, recon_fmac5 = llr_recon(fft_ivim, composite_sens, basis5,
                                use_basis=True, R=2, lambda1=0.001, lambda2=0.001)  # check different lambdas after

recon_fmac2 = np.real(recon_fmac2.squeeze())
recon_fmac3 = np.real(recon_fmac3.squeeze())
recon_fmac4 = np.real(recon_fmac4.squeeze())
recon_fmac5 = np.real(recon_fmac5.squeeze())

filtered = recon_fmac2
filtered3 = recon_fmac3
filtered4 = recon_fmac4
filtered5 = recon_fmac5
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

pub_figure(sub2, sub3, sub4, sub5, lowres, mags, GTs)
bland_altman_image(sub2, sub3, sub4, sub5, mags, GTs, masks)
