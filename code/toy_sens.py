import numpy as np
import sigpy as sp
import sigpy.mri as mr
import sigpy.plot as pl

# 1. Setup simulation parameters
img_shape = [256, 256]
num_coils = 32

# 2. Generate a phantom image (ground truth)
phantom = sp.shepp_logan(img_shape).astype(np.complex128)

# 3. Simulate "Birdcage" coil sensitivity maps
# These simulate physical coil profiles
mps_true = mr.birdcage_maps((num_coils, *img_shape))

# 4. Generate k-space data (Image * Sensitivity -> FFT)
# This mimics raw data coming off a scanner
ksp = sp.fft(phantom * mps_true, axes=(-2, -1))

# 5. Estimate sensitivity maps from k-space using ESPIRiT
# calib_width defines the central 'auto-calibration' region (ACS)
app = mr.app.EspiritCalib(ksp, calib_width=24, device=sp.cpu_device)
mps_estimated = app.run()

# 6. Visualize the results
# This will show the magnitude and phase of each coil's sensitivity
pl.ImagePlot(mps_estimated, title='Estimated Sensitivity Maps (ESPIRiT)')
