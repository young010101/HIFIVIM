# %% import region
from types import SimpleNamespace
# import importlib
import cyan_utils
import numpy as np
import sigpy as sp
import sigpy.mri as mr
import sigpy.plot as pl
import matplotlib.pyplot as plt
from utils import BVALS
from bart import bart

# %% global parameters
args = SimpleNamespace()
args.outdir = '../Phantom/'
POINTS = [(82, 82), (50, 50), (30, 130), (100, 60), (45, 90)]
num_bvals = len(BVALS)

# %% define some useful functions

# %% generate phantom
mean_diff, var_iso, var_aniso, dtd_gamma_bdelta_0, dtd_gamma_bdelta_1, masks = cyan_utils.make_phantom(args, show=True, points=POINTS)
x_dim = dtd_gamma_bdelta_0.shape[0]
y_dim = dtd_gamma_bdelta_0.shape[1]
dtd_gamma_bdelta_0_rot90 = np.rot90(dtd_gamma_bdelta_0, k=1)
if False:
    cyan_utils.show_15_bvals(dtd_gamma_bdelta_0_rot90)

def get_coil_ksp(img_xyb, num_coils=16, device=sp.cpu_device):
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

ksp_bdelta_0, mps_true = get_coil_ksp(dtd_gamma_bdelta_0_rot90)
fft_bdelta_0 = ksp_bdelta_0.transpose(1, 2, 0, 3)[:,:,None,:,:] # Nx, Ny, 1, coils, bvals
    
# %% estimate sensitivity maps from bdelta=0 data
ksp_calib = np.mean(ksp_bdelta_0, axis=-1)
app = mr.app.EspiritCalib(ksp_calib, calib_width=24, device=sp.Device(0))
mps_estimated = app.run()
mps_estimated = sp.to_device(mps_estimated, sp.cpu_device)
sens_maps_expand = np.moveaxis(mps_estimated, 0, -1)[..., None, :]  # Nx, Ny, 1, coils 

if False:
    pl.ImagePlot(mps_estimated, title='Estimated Sensitivity Maps')

# %% todo
sens_prelim_fix = np.zeros((x_dim, y_dim, num_bvals), dtype=np.complex128)
for i in range(num_bvals):
    # sens_prelim_fix[..., i] = bart(1, 'pics -S -l2 -r0.001 -i 10', fft_ivim[...,i], sens_maps)
    sens_prelim_fix[..., i] = bart(1, 'pics -S -l2 -r0.001 -i 10', fft_bdelta_0[...,i], sens_maps_expand)

if False:
    plt.figure(figsize=(15,3))
    for idx, (i, j) in enumerate(POINTS):
        ax = plt.subplot(1, len(POINTS), idx + 1)
        ax.plot(BVALS, dtd_gamma_bdelta_0_rot90[i, j], 'o-', label=r'$b_{\Delta}=0$')
        # ax.plot(BVALS, dtd_gamma_bdelta_1[i, j], 'x--', label=r'$b_{\Delta}=1$')
        ax.plot(BVALS, abs(sens_prelim_fix[i, j]), 's-.', label='Sens. Recon.')
        ax.legend()
# %%
