# %% import region
from types import SimpleNamespace
# import importlib
import cyan_utils
import numpy as np
import sigpy as sp
import sigpy.mri as mr
import sigpy.plot as pl
import matplotlib.pyplot as plt
from utils import BVALS, get_composite_sens
from bart import bart
import utils
import dict_gen_dtd
import plot_utils
import nibabel as nib
import os
import Phantom_utils

# %% global parameters
args = SimpleNamespace()
args.outdir = '../Phantom/'
ps = SimpleNamespace() 
ps.bp = '../Phantom/phan_cyan'  # <- base path
ps.ip = os.path.join(ps.bp, 'DATA', 'phan_brain', 'NII')  # <- actual input data
ps.op = os.path.join(ps.bp, 'processed', 'phan_brain')  # <- store output here
ps.zp = os.path.join(ps.bp, 'tmp')  # <- store temporary files here

POINTS = [(82, 82), (50, 50), (30, 130), (100, 60), (45, 90)]
num_bvals = len(BVALS)

# %% define some useful functions


def simulate_coil_ksp(img_xyb, num_coils=16, device=sp.cpu_device):
    """
    Parameters
    ----------
    img_xyb :   Nx, Ny, Nb

    Returns
    ------
    ksp :       Nc, Nx, Ny, Nb 
    mps_cxy :   Nc, Nx, Ny
    """
    img_shape = img_xyb.shape[:2]

    with sp.Device(device):
        mps_cxy = mr.birdcage_maps((num_coils, *img_shape))
        img = sp.to_device(img_xyb, device)
        coil_imgs = img[None, ...] * mps_cxy[..., None]
        ksp = sp.fft(coil_imgs, axes=(1, 2))

    return ksp, mps_cxy


def help_show_imgs(imgs, cmap='gray'):
    utils.show_imgs(np.abs(imgs.transpose(2,0,1)), cmap=cmap)


def save_nifti(img_data, ps, filename=None, ref_path=None, debug=False) -> None:
    out_path=ps.op
    # affine = nib.load(ref_path).affine
    ref_path = os.path.join(args.outdir, 'Phantom_T1.nii.gz') if ref_path is None else ref_path
    
    affine = nib.load(ref_path).affine 
    img = nib.Nifti1Image(img_data, affine)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    nib.save(img, os.path.join(out_path, f'{filename}.nii.gz'))
    # save bvals
    np.savetxt(os.path.join(out_path, f'{filename}.bval'), BVALS.reshape(1, len(BVALS)), fmt='%d', delimiter=' ')
    # save unit bvecs
    bvecs = np.zeros((3, len(BVALS)))
    bvecs[0, :] = 1  # x direction
    np.savetxt(os.path.join(out_path, f'{filename}.bvec'), bvecs, fmt='%d', delimiter=' ')

    if debug:
        print(os.listdir(out_path))


# %% generate phantom
mean_diff, var_iso, var_aniso, _dtd_gamma_bdelta_0, _dtd_gamma_bdelta_1, masks = cyan_utils.make_phantom(args, show=True, points=POINTS)
dtd_gamma_bdelta_0 = np.rot90(_dtd_gamma_bdelta_0, k=1)
x_dim = _dtd_gamma_bdelta_0.shape[0]
y_dim = _dtd_gamma_bdelta_0.shape[1]
if False:
    cyan_utils.show_15_bvals(dtd_gamma_bdelta_0)

composite_dtd = np.asarray(dtd_gamma_bdelta_0, dtype=np.complex128).copy()
if False:
    composite_dtd = Phantom_utils.add_phase(args, composite_dtd,
                            show=True)  # 164 x 164 x 15 - but now with different phase for each b-value.
if False:
    composite_dtd = utils.add_noise(composite_dtd, 20, return_img=False)

ksp_bdelta_0, mps_true = simulate_coil_ksp(composite_dtd)
fft_bdelta_0 = ksp_bdelta_0.transpose(1, 2, 0, 3)[:,:,None,:,:] # Nx, Ny, 1, coils, bvals
    
# %% estimate sensitivity maps from bdelta=0 data
ksp_calib = np.mean(ksp_bdelta_0, axis=-1)
app = mr.app.EspiritCalib(ksp_calib, calib_width=24, device=sp.Device(0))
mps_estimated = app.run()
mps_estimated = sp.to_device(mps_estimated, sp.cpu_device)
sens_maps_expand = np.moveaxis(mps_estimated, 0, -1)[..., None, :]  # Nx, Ny, 1, coils 

if False:
    pl.ImagePlot(mps_estimated, title='Estimated Sensitivity Maps')

# %% pics reconstruction with estimated sensitivity maps
sense_prelim = np.zeros((x_dim, y_dim, num_bvals), dtype=np.complex128)
for i in range(num_bvals):
    # sens_prelim_fix[..., i] = bart(1, 'pics -S -l2 -r0.001 -i 10', fft_ivim[...,i], sens_maps)
    sense_prelim[..., i] = bart(1, 'pics -S -l2 -r0.001 -i 10', fft_bdelta_0[...,i], sens_maps_expand)

if False:
    plt.figure(figsize=(15,3))
    for idx, (i, j) in enumerate(POINTS):
        ax = plt.subplot(1, len(POINTS), idx + 1)
        ax.plot(BVALS, dtd_gamma_bdelta_0[i, j], 'o-', label=r'$b_{\Delta}=0$')
        # ax.plot(BVALS, dtd_gamma_bdelta_1[i, j], 'x--', label=r'$b_{\Delta}=1$')
        ax.plot(BVALS, abs(sense_prelim[i, j]), 's-.', label='Sens. Recon.')
        ax.legend()
# %% get composite sensitivity maps: sensitivity maps + phase estimation from hamming windowed
_composite_sens, _ = get_composite_sens(sense_prelim, sens_maps_expand, visualize="True")  # 164 x 164 x 16 x 15 x 1
composite_sens = np.expand_dims(np.transpose(_composite_sens, (0, 1, 4, 2, 3)), axis=4)
if False:
    print('Composite Sens shape:', composite_sens.shape)  # Nx, Ny, 1, coils, 1, bvals

# %% gen dict
basis = dict_gen_dtd.basis_pipeline(bvals=BVALS, num_basis=5, debug=True)
print('Basis shape:', basis.shape)  # Nb, num_basis
# %%
fft_bdelta_0_expand = np.expand_dims(fft_bdelta_0, axis=4)  # Nx, Ny, 1, coils, 1, bvals
basis2 = basis[..., :2]  # Nb, 2
recon, recon_fmac_basis = utils.llr_recon_with_retry(
    fft_bdelta_0_expand,
    composite_sens,
    basis2,
    use_basis=True,
    lambda1=0.001,
    lambda2=0.001,
)
print(f"coeff shape: {recon.shape}, basis shape: {basis2.shape}")  # Nx, Ny, 1, 1, 1, 1, num_coeff
print(f'Reconstructed FMAC basis shape: {recon_fmac_basis.shape}, max {np.max(np.abs(recon_fmac_basis))}')  # Nx, Ny, 1, 1, 1, num_basis
print(f'{dtd_gamma_bdelta_0.shape}')

help_show_imgs(recon_fmac_basis.squeeze())
help_show_imgs(dtd_gamma_bdelta_0)
help_show_imgs(np.abs(recon_fmac_basis.squeeze()) - np.abs(dtd_gamma_bdelta_0), cmap='bwr')
help_show_imgs(recon_fmac_basis.squeeze() - dtd_gamma_bdelta_0)
print('MSE:', np.mean((np.abs(recon_fmac_basis.squeeze()) - dtd_gamma_bdelta_0)**2))
plot_utils.plot_recon_vs_ivim(recon_fmac_basis, dtd_gamma_bdelta_0, BVALS, POINTS, recon_label="2 basis", ivim_scale=1.0)


# %%
save_nifti(recon_fmac_basis.squeeze(), ps=ps, filename='phan_dtd_recon_2basis', ref_path=None, debug=True)
