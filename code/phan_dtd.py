# %% import region
from types import SimpleNamespace
from cyan_utils import make_phantom


# %% global parameters
args = SimpleNamespace()
args.outdir = '../Phantom/'
POINTS = [(82, 82), (50, 50), (30, 130), (100, 60), (45, 90)]

# %% define some useful functions

# %% generate phantom
mean_diff, var_iso, var_aniso, dtd_gamma_bdelta_0, dtd_gamma_bdelta_1, masks = make_phantom(args, show=True, points=POINTS)
# %%
