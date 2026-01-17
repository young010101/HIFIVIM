# %% import region
from types import SimpleNamespace
import importlib
import cyan_utils
import numpy as np
import sigpy as sp
import sigpy.mri as mr
import sigpy.plot as pl

# %% global parameters
args = SimpleNamespace()
args.outdir = '../Phantom/'
POINTS = [(82, 82), (50, 50), (30, 130), (100, 60), (45, 90)]

# %% define some useful functions

# %% generate phantom
mean_diff, var_iso, var_aniso, dtd_gamma_bdelta_0, dtd_gamma_bdelta_1, masks = cyan_utils.make_phantom(args, show=True, points=POINTS)
# %%
dtd_gamma_bdelta_0_rot90 = np.rot90(dtd_gamma_bdelta_0, k=1)
cyan_utils.show_15_bvals(dtd_gamma_bdelta_0_rot90)

def get_ksp(img_xyb, num_coils=16, device=sp.cpu_device):
    img_shape = img_xyb.shape[:2]

    with sp.Device(device):
        mps_cxy = mr.birdcage_maps((num_coils, *img_shape))
        img = sp.to_device(img_xyb, device)
        coil_imgs = img[None, ...] * mps_cxy[..., None]
        ksp = sp.fft(coil_imgs, axes=(1, 2))

    return ksp, mps_cxy

ksp_bdelta_0, mps_true = get_ksp(dtd_gamma_bdelta_0_rot90)
    
# %% estimate sensitivity maps from bdelta=0 data
ksp_calib = np.mean(ksp_bdelta_0, axis=-1)
app = mr.app.EspiritCalib(ksp_calib, calib_width=24, device=sp.Device(0))
mps_estimated = app.run()
mps_estimated = sp.to_device(mps_estimated, sp.cpu_device)

pl.ImagePlot(mps_estimated, title='Estimated Sensitivity Maps')

# %%
